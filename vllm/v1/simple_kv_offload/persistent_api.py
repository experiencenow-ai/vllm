# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External persistent cache API for SimpleCPUOffloadConnector blocks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests
import torch

from vllm.logger import init_logger
from vllm.v1.simple_kv_offload.persistent_disk import (
    FORMAT_VERSION,
    PersistedBlock,
    _short_hash,
)

logger = init_logger(__name__)

API_URL_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_URL"
API_TOKEN_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_TOKEN"
API_TIMEOUT_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_TIMEOUT"


class PersistentSimpleOffloadAPIClient:
    """Client for a node-local durable KV cache service.

    The API is intentionally a small control plane. Tensor payload movement uses
    node-local file handles returned by the service, not JSON tensor blobs.
    """

    def __init__(
        self,
        *,
        api_url: str,
        role: str,
        rank_key: str,
        model_key: str,
        num_cpu_blocks: int,
        strict: bool,
        tensor_names: list[str] | None = None,
        timeout: float = 5.0,
        token: str | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.role = role
        self.rank_key = rank_key
        self.model_key = model_key
        self.num_cpu_blocks = int(num_cpu_blocks)
        self.strict = strict
        self.tensor_names = tensor_names or []
        self.timeout = float(timeout)
        self.token = token
        self._session = requests.Session()

    def load_scheduler_entries(self, num_cpu_blocks: int) -> list[PersistedBlock]:
        # A scalable API backend should not enumerate lifetime cache metadata at
        # startup. The scheduler does request-scoped lookup instead.
        return []

    def load_worker_entries(self, num_cpu_blocks: int) -> list[PersistedBlock]:
        # Worker startup indexes nothing locally; materialization is lazy.
        return []

    def lookup_block_hashes(
        self,
        block_hashes: list[str],
        limit: int,
        cache_ref: str | None = None,
    ) -> list[str]:
        if not block_hashes or limit <= 0:
            return []
        payload = {
            **self._base_payload(),
            "block_hashes": block_hashes,
            "limit": int(limit),
        }
        if cache_ref is not None:
            payload["cache_ref"] = cache_ref
        response = self._post_json("/v1/kv/lookup", payload)
        hits = _parse_hash_hits(response.get("hits", []))
        wanted = set(block_hashes)
        return [hash_hex for hash_hex in hits if hash_hex in wanted][:limit]

    def save_scheduler_blocks(
        self, cpu_block_ids: list[int], block_hashes: list[str]
    ) -> None:
        if not cpu_block_ids:
            return
        self._post_json(
            "/v1/kv/scheduler/commit",
            {
                **self._base_payload(),
                "blocks": _block_records(cpu_block_ids, block_hashes),
            },
        )

    def ensure_worker_blocks(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
        known_by_cpu_id: dict[int, str],
        cache_refs: list[str | None] | None = None,
    ) -> dict[int, str]:
        restored: dict[int, str] = {}
        pairs = [
            (int(cpu_id), hash_hex, cache_ref)
            for cpu_id, hash_hex, cache_ref in zip(
                cpu_block_ids,
                block_hashes,
                cache_refs or [None] * len(cpu_block_ids),
            )
            if known_by_cpu_id.get(int(cpu_id)) != hash_hex
        ]
        if not pairs:
            return restored
        response = self._post_json(
            "/v1/kv/materialize",
            {
                **self._base_payload(),
                "blocks": [
                    _block_record(cpu_id, hash_hex, cache_ref)
                    for cpu_id, hash_hex, cache_ref in pairs
                ],
            },
        )
        payload_paths = _parse_payload_paths(response)
        for cpu_id, hash_hex, _ in pairs:
            path = payload_paths.get(hash_hex)
            if path is None:
                self._fail(
                    "cache service did not materialize block "
                    f"{_short_hash(hash_hex)}"
                )
                continue
            try:
                self._restore_worker_block(
                    cpu_kv_caches, cpu_id, hash_hex, Path(path)
                )
                restored[cpu_id] = hash_hex
            except Exception as exc:
                self._fail(
                    f"failed to materialize CPU offload block {_short_hash(hash_hex)}",
                    exc,
                )
        return restored

    def persist_worker_blocks(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
    ) -> None:
        if not cpu_block_ids:
            return
        blocks = _block_records(cpu_block_ids, block_hashes)
        response = self._post_json(
            "/v1/kv/store/prepare",
            {**self._base_payload(), "blocks": blocks},
        )
        payload_paths = _parse_payload_paths(response)
        committed: list[dict[str, Any]] = []
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            path = payload_paths.get(hash_hex)
            if path is None:
                self._fail(
                    "cache service did not provide a staging path for block "
                    f"{_short_hash(hash_hex)}"
                )
                continue
            try:
                payload = {
                    "format": "ds4-vllm-simple-cpu-offload-block-v1",
                    "version": FORMAT_VERSION,
                    "model": self.model_key,
                    "rank": self.rank_key,
                    "cpu_block_id": int(cpu_block_id),
                    "hash": hash_hex,
                    "tensors": {
                        name: tensor[int(cpu_block_id)].detach().cpu().clone()
                        for name, tensor in cpu_kv_caches.items()
                    },
                }
                self._torch_save_atomic(Path(path), payload)
                committed.append(
                    {
                        "cpu_block_id": int(cpu_block_id),
                        "hash": hash_hex,
                        "path": str(path),
                    }
                )
            except Exception as exc:
                self._fail(
                    f"failed to stage CPU offload block {_short_hash(hash_hex)}",
                    exc,
                )
        if committed:
            self._post_json(
                "/v1/kv/store/commit",
                {**self._base_payload(), "blocks": committed},
            )

    def validate_loaded_blocks(
        self,
        cpu_block_ids: list[int],
        block_hashes: list[str],
        known_by_cpu_id: dict[int, str],
    ) -> None:
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            actual = known_by_cpu_id.get(int(cpu_block_id))
            if actual != hash_hex:
                self._fail(
                    "cache API load is missing materialized tensor data "
                    f"for cpu_block={cpu_block_id} hash={_short_hash(hash_hex)} "
                    f"actual={_short_hash(actual)}"
                )

    def _restore_worker_block(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_id: int,
        hash_hex: str,
        path: Path,
    ) -> None:
        payload = self._torch_load(path)
        if payload.get("hash") != hash_hex:
            raise ValueError("block hash mismatch")
        tensors = payload.get("tensors")
        if not isinstance(tensors, dict):
            raise ValueError("block payload has no tensor map")
        for name, dst in cpu_kv_caches.items():
            if name not in tensors:
                raise ValueError(f"block payload missing tensor {name}")
            src = tensors[name]
            target = dst[cpu_block_id]
            if tuple(src.shape) != tuple(target.shape):
                raise ValueError(
                    f"tensor {name} shape mismatch "
                    f"{tuple(src.shape)} != {tuple(target.shape)}"
                )
            target.copy_(src)

    def _base_payload(self) -> dict[str, Any]:
        return {
            "format": "ds4-vllm-simple-cpu-offload-api-v1",
            "version": FORMAT_VERSION,
            "model": self.model_key,
            "rank": self.rank_key,
            "role": self.role,
            "num_cpu_blocks": self.num_cpu_blocks,
            "tensor_names": self.tensor_names,
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self._session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("cache API response is not a JSON object")
            return data
        except Exception as exc:
            self._fail(f"cache API request failed: {path}", exc)
            return {}

    def _torch_save_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _torch_load(self, path: Path) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _fail(self, message: str, exc: Exception | None = None) -> None:
        if self.strict:
            raise RuntimeError(message) from exc
        logger.warning(
            "DS4 persistent SimpleCPUOffload API warning: %s",
            message,
            exc_info=exc is not None,
        )


def _block_records(
    cpu_block_ids: list[int], block_hashes: list[str]
) -> list[dict[str, Any]]:
    return [
        _block_record(int(cpu_block_id), hash_hex)
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes)
    ]


def _block_record(
    cpu_block_id: int,
    hash_hex: str,
    cache_ref: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"cpu_block_id": int(cpu_block_id), "hash": hash_hex}
    if cache_ref is not None:
        record["cache_ref"] = cache_ref
    return record


def _parse_hash_hits(raw_hits: Any) -> list[str]:
    hits: list[str] = []
    if not isinstance(raw_hits, list):
        return hits
    for item in raw_hits:
        if isinstance(item, str):
            hits.append(item)
        elif isinstance(item, dict) and isinstance(item.get("hash"), str):
            hits.append(item["hash"])
    return hits


def _parse_payload_paths(response: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key in ("blocks", "hits", "payloads"):
        raw_blocks = response.get(key)
        if not isinstance(raw_blocks, list):
            continue
        for item in raw_blocks:
            if not isinstance(item, dict):
                continue
            hash_hex = item.get("hash")
            path = item.get("path") or item.get("payload_path")
            if isinstance(hash_hex, str) and isinstance(path, str):
                paths[hash_hex] = path
    raw_map = response.get("payload_paths")
    if isinstance(raw_map, dict):
        for hash_hex, path in raw_map.items():
            if isinstance(hash_hex, str) and isinstance(path, str):
                paths[hash_hex] = path
    return paths
