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
FORMAT_VERSION = 1


@dataclass(frozen=True)
class PersistedBlock:
    cpu_block_id: int
    hash_hex: str
    block_hash: bytes


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
        self.scheduler_index = self.root / "scheduler_index.json"
        self.worker_index = self.worker_dir / "worker_index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        *,
        role: str,
        vllm_config: Any,
        num_cpu_blocks: int,
        tensor_names: list[str] | None = None,
    ) -> "PersistentSimpleOffloadStore | None":
        root = os.getenv(ROOT_ENV)
        if not root:
            return None
        rank_key = (
            os.getenv(RANK_ENV) or os.getenv("VLLM_HOST_IP") or socket.gethostname()
        )
        store = cls(
            root=Path(root),
            rank_key=_safe_key(rank_key),
            model_key=_model_key(vllm_config),
            num_cpu_blocks=int(num_cpu_blocks),
            strict=_env_bool(STRICT_ENV, True),
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
        self, cpu_block_ids: list[int], block_hashes: list[str]
    ) -> None:
        self._upsert_index(self.scheduler_index, cpu_block_ids, block_hashes)

    def persist_worker_blocks(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
    ) -> None:
        self._validate_pairs(cpu_block_ids, block_hashes)
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
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
                self._torch_save_atomic(self._block_path(hash_hex), payload)
            except Exception as exc:
                self._fail(
                    f"failed to persist CPU offload block {_short_hash(hash_hex)}", exc
                )
        self._upsert_index(self.worker_index, cpu_block_ids, block_hashes)

    def ensure_worker_blocks(
        self,
        cpu_kv_caches: dict[str, torch.Tensor],
        cpu_block_ids: list[int],
        block_hashes: list[str],
        known_by_cpu_id: dict[int, str],
    ) -> dict[int, str]:
        restored: dict[int, str] = {}
        self._validate_pairs(cpu_block_ids, block_hashes)
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            cpu_id = int(cpu_block_id)
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
                    )
                )
                seen_cpu_ids.add(cpu_block_id)
            except Exception as exc:
                self._fail(f"invalid persistent offload index entry {hash_hex}", exc)
        return entries

    def _upsert_index(
        self, path: Path, cpu_block_ids: list[int], block_hashes: list[str]
    ) -> None:
        self._validate_pairs(cpu_block_ids, block_hashes)
        data = self._read_json(path) or self._empty_index()
        blocks = data.setdefault("blocks", {})
        stale_block_hashes: list[str] = []
        for cpu_block_id, hash_hex in zip(cpu_block_ids, block_hashes):
            cpu_id = int(cpu_block_id)
            for old_hash, old_entry in list(blocks.items()):
                if (
                    int(old_entry.get("cpu_block_id", -1)) == cpu_id
                    and old_hash != hash_hex
                ):
                    blocks.pop(old_hash, None)
                    stale_block_hashes.append(old_hash)
            blocks[hash_hex] = {"cpu_block_id": cpu_id, "updated_at": time.time()}
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
    return _safe_key(str(model or "unknown-model"))


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _short_hash(value: str | None) -> str:
    if value is None:
        return "None"
    if len(value) <= 32:
        return value
    return f"{value[:16]}...{value[-16:]}(len={len(value)})"
