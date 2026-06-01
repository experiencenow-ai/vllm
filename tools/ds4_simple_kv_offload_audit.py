#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static DS4 checks for SimpleCPUOffload distributed block contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _require(name: str, rel: str, needles: list[str]) -> None:
    text = _read(rel)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(
            f"FAIL: {name}: missing {missing!r} in {rel}"
        )
    print(f"PASS: {name}")


def main() -> None:
    _require(
        "shared logical CPU block helper exists",
        "vllm/v1/simple_kv_offload/capacity.py",
        ["derive_logical_cpu_block_count", "shared by scheduler and workers"],
    )
    _require(
        "scheduler uses shared logical CPU block helper",
        "vllm/v1/simple_kv_offload/manager.py",
        ["derive_logical_cpu_block_count("],
    )
    _require(
        "worker allocates scheduler-compatible logical id space",
        "vllm/v1/simple_kv_offload/worker.py",
        [
            "derive_logical_cpu_block_count(",
            "logical CPU block id space",
            "scheduler-emitted",
        ],
    )
    _require(
        "worker grows CPU tensors when scheduler metadata exceeds local stage estimate",
        "vllm/v1/simple_kv_offload/worker.py",
        [
            "_ensure_cpu_block_capacity",
            "_required_cpu_blocks",
            "growing CPU KV offload block id space",
        ],
    )
    _require(
        "persistent offload cache key includes deployment namespace",
        "vllm/v1/simple_kv_offload/persistent_disk.py",
        ["VLLM_SIMPLE_KV_OFFLOAD_PERSIST_NAMESPACE", "__{namespace}"],
    )
    _require(
        "persistent offload stores cache refs with block indexes",
        "vllm/v1/simple_kv_offload/persistent_disk.py",
        ["cache_refs", "_entry_cache_refs", "cache_ref not in _entry_cache_refs"],
    )
    _require(
        "scheduler only stores DS4-marked cache requests",
        "vllm/v1/simple_kv_offload/manager.py",
        [
            "_request_wants_store",
            "and _request_wants_store(request)",
            "VLLM_DS4_SIMPLE_KV_STORE_UNMARKED",
        ],
    )
    _require(
        "persistent cache startup restore can be disabled for DS4 services",
        "vllm/v1/simple_kv_offload/manager.py",
        [
            "_startup_restore_enabled",
            "persistent startup restore disabled",
            "VLLM_DS4_SIMPLE_KV_STARTUP_RESTORE",
        ],
    )
    _require(
        "store metadata carries cache refs to workers",
        "vllm/v1/simple_kv_offload/metadata.py",
        ["store_cache_refs"],
    )
    _require(
        "worker persists cache refs with tensor payloads",
        "vllm/v1/simple_kv_offload/worker.py",
        ["metadata.store_cache_refs", "persist_worker_blocks("],
    )
    for rel in (
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "tools/ds4_launch_dsv4_flash_pp4_tp2_ep.sh",
    ):
        _require(
            f"{rel} partitions persistent offload namespace",
            rel,
            [
                "DSV4_OFFLOAD_PARTITION_KEY",
                "VLLM_SIMPLE_KV_OFFLOAD_PERSIST_NAMESPACE",
                "part_${DSV4_OFFLOAD_PARTITION_KEY}",
                "VLLM_DS4_SIMPLE_KV_STARTUP_RESTORE",
                "VLLM_DS4_SIMPLE_KV_STORE_UNMARKED",
            ],
        )


if __name__ == "__main__":
    main()
