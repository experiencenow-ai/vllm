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
        "persistent offload cache key includes deployment namespace",
        "vllm/v1/simple_kv_offload/persistent_disk.py",
        ["VLLM_SIMPLE_KV_OFFLOAD_PERSIST_NAMESPACE", "__{namespace}"],
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
            ],
        )


if __name__ == "__main__":
    main()
