# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk persistence for SimpleCPUOffloadConnector CPU blocks."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ROOT_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT"
STRICT_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT"
RANK_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK"
NAMESPACE_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_NAMESPACE"
API_URL_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_URL"
API_TOKEN_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_TOKEN"
API_TIMEOUT_ENV = "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_TIMEOUT"
BUNDLES_ENV = "VLLM_DS4_SIMPLE_KV_BUNDLES"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class PersistedBlock:
    cpu_block_id: int
    hash_hex: str
    block_hash: bytes
    cache_refs: tuple[str, ...] = ()


class PersistentSimpleOffloadStore:
    def __init__(
        self,
        *,
        root: Path,
        rank_key: str,
        model_key: str,
        num_cpu_blocks: int,
        strict: bool,
        tensor_names: list[str] | None = None,
    ) -> None:
        self.root = root
        self.rank_key = rank_key
        self.model_key = model_key
        self.num_cpu_blocks = int(num_cpu_blocks)
        self.strict = strict
        self.tensor_names = tensor_names or []
        self.worker_dir = self.root / "workers" / self.rank_key
        self.blocks_dir = self.worker_dir / "blocks"
        self.bundles_dir = self.worker_dir / "cache_ref_bundles"
        self.scheduler_index = self.root / "scheduler_index.json"
        self.worker_index = self.worker_dir / "worker_index.json"
        self.bundle_index = self.worker_dir / "cache_ref_bundle_index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        *,
        role: str,
        vllm_config: Any,
        num_cpu_blocks: int,
        tensor_names: list[str] | None = None,
    ) -> Any | None:
        rank_key = (
            os.getenv(RANK_ENV) or os.getenv("VLLM_HOST_IP") or socket.gethostname()
        )
        model_key = _model_key(vllm_config)
        strict = _env_bool(STRICT_ENV, True)
        api_url = os.getenv(API_URL_ENV)
        if api_url:
            from vllm.v1.simple_kv_offload.persistent_api import (
                PersistentSimpleOffloadAPIClient,
            )

            store = PersistentSimpleOffloadAPIClient(
                api_url=api_url,
                role=role,
                rank_key=_safe_key(rank_key),
                model_key=model_key,
                num_cpu_blocks=int(num_cpu_blocks),
                strict=strict,
                tensor_names=tensor_names,
                timeout=float(os.getenv(API_TIMEOUT_ENV, "5.0")),
                token=os.getenv(API_TOKEN_ENV),
            )
            logger.info(
                "DS4 persistent SimpleCPUOffload %s API enabled at %s "
                "rank=%s strict=%s",
                role,
                api_url,
                store.rank_key,
                store.strict,
            )
            return store
        root = os.getenv(ROOT_ENV)
        if not root:
            return None
        store = cls(
            root=Path(root),
            rank_key=_safe_key(rank_key),
            model_key=model_key,
            num_cpu_blocks=int(num_cpu_blocks),
            strict=strict,
            tensor_names=tensor_names,
        )
        logger.info(
            "DS4 persistent SimpleCPUOffload %s store enabled at %s rank=%s strict=%s",
            role,
            store.root,
            store.rank_key,
            store.strict,
        )
        return store

    def load_scheduler_entries(self, num_cpu_blocks: int) -> list[PersistedBlock]:
        return self._entries_from_index(
            self._read_json(self.scheduler_index), num_cpu_blocks
        )

    def load_worker_entries(self, num_cpu_blocks: int) -> list[PersistedBlock]:
        return self._entries_from_index(
            self._read_json(self.worker_index), num_cpu_blocks
        )

    def save_scheduler_blocks(
        self,
        cpu_block_ids: list[int],
        block_hashes: list[str],
        cache_refs: list[str | None] | None = None,
    ) -> None:
        self._upsert_index(
            self.scheduler_index, cpu_block_ids, block_hashes, cache_refs
        )

    def lookup_block_hashes(
        self,
        block_hashes: list[str],
        limit: int,
        cache_ref: str | None = None,
    ) -> list[str]:
        if not block_hashes or limit <= 0:
            return []
        available: set[str] = set()
        for path in (self.scheduler_index, self.worker_index):
            data = self._read_json(path)
            if not data:
                continue
            for hash_hex, raw in (data.get("blocks") or {}).items():
                if cache_ref is not None and cache_ref not in _entry_cache_refs(raw):
                    continue
                available.add(hash_hex)
        hits: list[str] = []
        for hash_hex in block_hashes:
            if hash_hex in available:
                hits.append(hash_hex)
                if len(hits) >= limit:
                    break
        return hits

    def persist_worker_blocks(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
        cache_refs: list[str | None] | None = None,
    ) -> None:
        self._validate_pairs(cpu_block_ids, block_hashes)
        refs = cache_refs or [None] * len(cpu_block_ids)
        for cpu_block_id, hash_hex, cache_ref in zip(
            cpu_block_ids, block_hashes, refs
        ):
            try:
                payload = {
                    "format": "ds4-vllm-simple-cpu-offload-block-v1",
                    "version": FORMAT_VERSION,
                    "model": self.model_key,
                    "rank": self.rank_key,
                    "cpu_block_id": int(cpu_block_id),
                    "hash": hash_hex,
                    "cache_ref": cache_ref,
                    "tensors": {
                        name: tensor[int(cpu_block_id)].detach().cpu().clone()
                        for name, tensor in cpu_kv_caches.items()
                    },
                }
                self._torch_save_atomic(self._block_path(hash_hex), payload)
            except Exception as exc:
                self._fail(
                    f"failed to persist CPU offload block {_short_hash(hash_hex)}", exc
                )
        self._upsert_index(self.worker_index, cpu_block_ids, block_hashes, cache_refs)
        self._persist_cache_ref_bundles(
            cpu_kv_caches, cpu_block_ids, block_hashes, refs
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
        self._validate_pairs(cpu_block_ids, block_hashes)
        refs = cache_refs or [None] * len(cpu_block_ids)
        if _bundle_enabled():
            restored.update(
                self._restore_worker_blocks_from_bundles(
                    cpu_kv_caches, cpu_block_ids, block_hashes, refs, known_by_cpu_id
                )
            )
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            cpu_id = int(cpu_block_id)
            if restored.get(cpu_id) == hash_hex:
                continue
            if known_by_cpu_id.get(cpu_id) == hash_hex:
                continue
            path = self._block_path(hash_hex)
            if not path.exists():
                self._fail(
                    "persistent CPU offload block missing: "
                    f"{_short_hash(hash_hex)} at {path}"
                )
                continue
            try:
                self._restore_worker_block(cpu_kv_caches, cpu_id, hash_hex, path)
                restored[cpu_id] = hash_hex
            except Exception as exc:
                self._fail(
                    f"failed to restore CPU offload block {_short_hash(hash_hex)}",
                    exc,
                )
        return restored

    def _persist_cache_ref_bundles(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
        cache_refs: list[str | None],
    ) -> None:
        if not _bundle_enabled():
            return
        by_ref: dict[str, list[int]] = {}
        for idx, cache_ref in enumerate(cache_refs):
            if not cache_ref:
                continue
            by_ref.setdefault(str(cache_ref), []).append(idx)
        if not by_ref:
            return

        index = self._read_json(self.bundle_index) or self._empty_bundle_index()
        refs = index.setdefault("refs", {})
        for cache_ref, positions in by_ref.items():
            safe_ref = _safe_key(cache_ref)
            ref_dir = self.bundles_dir / safe_ref
            ref_dir.mkdir(parents=True, exist_ok=True)
            bundle_name = f"{time.time_ns()}-{os.getpid()}.pt"
            bundle_path = ref_dir / bundle_name
            ids = [int(cpu_block_ids[pos]) for pos in positions]
            hashes = [block_hashes[pos] for pos in positions]
            try:
                payload = {
                    "format": "ds4-vllm-simple-cpu-offload-bundle-v1",
                    "version": FORMAT_VERSION,
                    "model": self.model_key,
                    "rank": self.rank_key,
                    "cache_ref": cache_ref,
                    "hashes": hashes,
                    "tensors": {
                        name: tensor[ids].detach().cpu().clone()
                        for name, tensor in cpu_kv_caches.items()
                    },
                }
                self._torch_save_atomic(bundle_path, payload)
                rel = str(bundle_path.relative_to(self.worker_dir))
                ref_entry = refs.setdefault(
                    safe_ref,
                    {
                        "cache_ref": cache_ref,
                        "blocks": {},
                    },
                )
                ref_entry["cache_ref"] = cache_ref
                blocks = ref_entry.setdefault("blocks", {})
                for row, hash_hex in enumerate(hashes):
                    blocks[hash_hex] = {
                        "bundle": rel,
                        "row": row,
                        "updated_at": time.time(),
                    }
            except Exception as exc:
                self._fail(
                    f"failed to persist CPU offload bundle for cache_ref={cache_ref!r}",
                    exc,
                )
        index["updated_at"] = time.time()
        self._write_json_atomic(self.bundle_index, index)

    def _restore_worker_blocks_from_bundles(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
        cache_refs: list[str | None],
        known_by_cpu_id: dict[int, str],
    ) -> dict[int, str]:
        index = self._read_json(self.bundle_index)
        if not index:
            return {}
        restored: dict[int, str] = {}
        loads: dict[tuple[str, str], list[tuple[int, str, int]]] = {}
        refs = index.get("refs")
        if not isinstance(refs, dict):
            return {}
        for cpu_block_id, hash_hex, cache_ref in zip(
            cpu_block_ids, block_hashes, cache_refs
        ):
            cpu_id = int(cpu_block_id)
            if known_by_cpu_id.get(cpu_id) == hash_hex:
                continue
            if not cache_ref:
                continue
            ref_entry = refs.get(_safe_key(str(cache_ref)))
            if not isinstance(ref_entry, dict):
                continue
            if ref_entry.get("cache_ref") != str(cache_ref):
                continue
            blocks = ref_entry.get("blocks")
            if not isinstance(blocks, dict):
                continue
            block_entry = blocks.get(hash_hex)
            if not isinstance(block_entry, dict):
                continue
            bundle_rel = block_entry.get("bundle")
            row = block_entry.get("row")
            if not isinstance(bundle_rel, str) or not isinstance(row, int):
                continue
            loads.setdefault((str(cache_ref), bundle_rel), []).append(
                (cpu_id, hash_hex, row)
            )

        for (cache_ref, bundle_rel), rows in loads.items():
            path = self.worker_dir / bundle_rel
            try:
                payload = self._torch_load(path)
                self._restore_bundle_rows(cpu_kv_caches, cache_ref, path, payload, rows)
                for cpu_id, hash_hex, _ in rows:
                    restored[cpu_id] = hash_hex
            except Exception as exc:
                self._fail(
                    f"failed to restore CPU offload bundle {path} "
                    f"for cache_ref={cache_ref!r}",
                    exc,
                )
        if restored:
            logger.info(
                "DS4 persistent SimpleCPUOffload restored %d blocks from "
                "%d cache-ref bundle files",
                len(restored),
                len(loads),
            )
        return restored

    def _restore_bundle_rows(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cache_ref: str,
        path: Path,
        payload: dict[str, Any],
        rows: list[tuple[int, str, int]],
    ) -> None:
        if payload.get("format") != "ds4-vllm-simple-cpu-offload-bundle-v1":
            raise ValueError("bundle payload format mismatch")
        if int(payload.get("version", 0)) != FORMAT_VERSION:
            raise ValueError("bundle payload version mismatch")
        if payload.get("model") != self.model_key:
            raise ValueError("bundle payload model mismatch")
        if payload.get("rank") != self.rank_key:
            raise ValueError("bundle payload rank mismatch")
        if payload.get("cache_ref") != cache_ref:
            raise ValueError("bundle payload cache_ref mismatch")
        hashes = payload.get("hashes")
        tensors = payload.get("tensors")
        if not isinstance(hashes, list) or not isinstance(tensors, dict):
            raise ValueError("bundle payload missing hashes/tensors")
        for name, dst in cpu_kv_caches.items():
            src_rows = tensors.get(name)
            if src_rows is None:
                raise ValueError(f"bundle {path} missing tensor {name}")
            for cpu_id, hash_hex, row in rows:
                if row < 0 or row >= len(hashes):
                    raise ValueError(f"bundle row out of range: {row}")
                if hashes[row] != hash_hex:
                    raise ValueError("bundle row hash mismatch")
                src = src_rows[row]
                target = dst[cpu_id]
                if tuple(src.shape) != tuple(target.shape):
                    raise ValueError(
                        f"tensor {name} shape mismatch "
                        f"{tuple(src.shape)} != {tuple(target.shape)}"
                    )
                target.copy_(src)

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

    def validate_loaded_blocks(
        self,
        cpu_block_ids: list[int],
        block_hashes: list[str],
        known_by_cpu_id: dict[int, str],
    ) -> None:
        self._validate_pairs(cpu_block_ids, block_hashes)
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            actual = known_by_cpu_id.get(int(cpu_block_id))
            if actual != hash_hex:
                self._fail(
                    "persistent CPU offload load is missing restored tensor data "
                    f"for cpu_block={cpu_block_id} hash={_short_hash(hash_hex)} "
                    f"actual={_short_hash(actual)}"
                )

    def _entries_from_index(
        self, data: dict[str, Any] | None, num_cpu_blocks: int
    ) -> list[PersistedBlock]:
        if not data:
            return []
        if int(data.get("version", 0)) != FORMAT_VERSION:
            self._fail(
                f"unsupported persistent offload index version in {data.get('format')}"
            )
            return []
        if data.get("model") not in (None, self.model_key):
            self._fail(
                f"persistent offload model mismatch: "
                f"{data.get('model')} != {self.model_key}"
            )
            return []
        entries: list[PersistedBlock] = []
        seen_cpu_ids: set[int] = set()
        for hash_hex, raw in sorted(
            (data.get("blocks") or {}).items(),
            key=lambda item: int(item[1].get("cpu_block_id", -1)),
        ):
            try:
                cpu_block_id = int(raw["cpu_block_id"])
                if cpu_block_id < 0 or cpu_block_id >= num_cpu_blocks:
                    continue
                if cpu_block_id in seen_cpu_ids:
                    continue
                entries.append(
                    PersistedBlock(
                        cpu_block_id=cpu_block_id,
                        hash_hex=hash_hex,
                        block_hash=bytes.fromhex(hash_hex),
                        cache_refs=tuple(_entry_cache_refs(raw)),
                    )
                )
                seen_cpu_ids.add(cpu_block_id)
            except Exception as exc:
                self._fail(f"invalid persistent offload index entry {hash_hex}", exc)
        return entries

    def _upsert_index(
        self,
        path: Path,
        cpu_block_ids: list[int],
        block_hashes: list[str],
        cache_refs: list[str | None] | None = None,
    ) -> None:
        self._validate_pairs(cpu_block_ids, block_hashes)
        refs = cache_refs or [None] * len(cpu_block_ids)
        data = self._read_json(path) or self._empty_index()
        blocks = data.setdefault("blocks", {})
        stale_block_hashes: list[str] = []
        for cpu_block_id, hash_hex, cache_ref in zip(
            cpu_block_ids, block_hashes, refs
        ):
            cpu_id = int(cpu_block_id)
            for old_hash, old_entry in list(blocks.items()):
                if (
                    int(old_entry.get("cpu_block_id", -1)) == cpu_id
                    and old_hash != hash_hex
                ):
                    blocks.pop(old_hash, None)
                    stale_block_hashes.append(old_hash)
            existing_refs = set(_entry_cache_refs(blocks.get(hash_hex)))
            if cache_ref:
                existing_refs.add(str(cache_ref))
            blocks[hash_hex] = {
                "cpu_block_id": cpu_id,
                "updated_at": time.time(),
                "cache_refs": sorted(existing_refs),
            }
        data["updated_at"] = time.time()
        self._write_json_atomic(path, data)
        if path == self.worker_index:
            for old_hash in stale_block_hashes:
                if old_hash not in blocks:
                    self._unlink_stale_block(old_hash)

    def _validate_pairs(
        self, cpu_block_ids: list[int], block_hashes: list[str]
    ) -> None:
        if len(cpu_block_ids) != len(block_hashes):
            self._fail(
                f"CPU block/hash length mismatch "
                f"{len(cpu_block_ids)} != {len(block_hashes)}"
            )
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            if int(cpu_block_id) < 0 or int(cpu_block_id) >= self.num_cpu_blocks:
                self._fail(f"CPU block id out of range: {cpu_block_id}")
            try:
                bytes.fromhex(hash_hex)
            except ValueError as exc:
                self._fail(f"invalid block hash hex: {hash_hex}", exc)

    def _empty_index(self) -> dict[str, Any]:
        return {
            "format": "ds4-vllm-simple-cpu-offload-index-v1",
            "version": FORMAT_VERSION,
            "model": self.model_key,
            "rank": self.rank_key,
            "num_cpu_blocks": self.num_cpu_blocks,
            "tensor_names": self.tensor_names,
            "created_at": time.time(),
            "updated_at": time.time(),
            "blocks": {},
        }

    def _empty_bundle_index(self) -> dict[str, Any]:
        return {
            "format": "ds4-vllm-simple-cpu-offload-bundle-index-v1",
            "version": FORMAT_VERSION,
            "model": self.model_key,
            "rank": self.rank_key,
            "created_at": time.time(),
            "updated_at": time.time(),
            "refs": {},
        }

    def _block_path(self, hash_hex: str) -> Path:
        digest = hashlib.sha256(hash_hex.encode("ascii")).hexdigest()
        return self.blocks_dir / f"{digest}.pt"

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            self._fail(f"failed to read persistent offload index {path}", exc)
            return None

    def _write_json_atomic(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, indent=2))
        os.replace(tmp, path)

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

    def _unlink_stale_block(self, hash_hex: str) -> None:
        try:
            self._block_path(hash_hex).unlink(missing_ok=True)
        except Exception as exc:
            self._fail(f"failed to remove stale CPU offload block {hash_hex}", exc)

    def _fail(self, message: str, exc: Exception | None = None) -> None:
        if self.strict:
            raise RuntimeError(message) from exc
        logger.warning(
            "DS4 persistent SimpleCPUOffload warning: %s",
            message,
            exc_info=exc is not None,
        )


def _model_key(vllm_config: Any) -> str:
    model_config = getattr(vllm_config, "model_config", None)
    model = getattr(model_config, "model", None) or getattr(
        model_config, "served_model_name", None
    )
    namespace = os.getenv(NAMESPACE_ENV)
    if namespace:
        return _safe_key(f"{model or 'unknown-model'}__{namespace}")
    return _safe_key(str(model or "unknown-model"))


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _entry_cache_refs(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    refs = raw.get("cache_refs")
    if isinstance(refs, list):
        return [str(ref) for ref in refs if isinstance(ref, str) and ref]
    ref = raw.get("cache_ref")
    if isinstance(ref, str) and ref:
        return [ref]
    return []


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _bundle_enabled() -> bool:
    return _env_bool(BUNDLES_ENV, True)


def _short_hash(value: str | None) -> str:
    if value is None:
        return "None"
    if len(value) <= 32:
        return value
    return f"{value[:16]}...{value[-16:]}(len={len(value)})"
