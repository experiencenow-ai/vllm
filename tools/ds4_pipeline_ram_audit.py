#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static DS4 audit for cache-primary pipeline launchers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    (
        "Qwen NVFP4 PP launcher disables async by default",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "ASYNC_SCHEDULING_ARGS=(--no-async-scheduling)",
    ),
    (
        "Qwen BF16 PP launcher disables async by default",
        "tools/ds4_launch_qwen27_pp8.sh",
        "ASYNC_SCHEDULING_ARGS=(--no-async-scheduling)",
    ),
    (
        "Qwen NVFP4 async requires experimental opt-in",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_ENABLE_ASYNC_SCHEDULING_EXPERIMENTAL",
    ),
    (
        "Qwen BF16 async requires experimental opt-in",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_ENABLE_ASYNC_SCHEDULING_EXPERIMENTAL",
    ),
    (
        "Qwen NVFP4 PP launcher defaults PP globals to NCCL",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        'VLLM_DS4_PP_ONLY_GLOBAL_BACKEND="${VLLM_DS4_PP_ONLY_GLOBAL_BACKEND:-nccl}"',
    ),
    (
        "Qwen BF16 PP launcher defaults PP globals to NCCL",
        "tools/ds4_launch_qwen27_pp8.sh",
        'VLLM_DS4_PP_ONLY_GLOBAL_BACKEND="${VLLM_DS4_PP_ONLY_GLOBAL_BACKEND:-nccl}"',
    ),
    (
        "Qwen NVFP4 default layer split offsets last-rank head work",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        'QWEN27_PP_LAYER_PARTITION="9,9,9,8,8,8,8,5"',
    ),
    (
        "Qwen BF16 default layer split offsets last-rank head work",
        "tools/ds4_launch_qwen27_pp8.sh",
        'QWEN27_PP_LAYER_PARTITION="9,9,9,8,8,8,8,5"',
    ),
    (
        "Qwen NVFP4 resident3 KV cap is compact",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_KV_CACHE_MEMORY_BYTES:=4294967296",
    ),
    (
        "Qwen BF16 resident3 KV cap is compact",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_KV_CACHE_MEMORY_BYTES:=4294967296",
    ),
    (
        "Qwen NVFP4 resident3 local CPU LMCache cap is compact",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "LMCACHE_MAX_LOCAL_CPU_SIZE:=0.5",
    ),
    (
        "Qwen BF16 resident3 local CPU LMCache cap is compact",
        "tools/ds4_launch_qwen27_pp8.sh",
        "LMCACHE_MAX_LOCAL_CPU_SIZE:=0.5",
    ),
    (
        "Qwen NVFP4 launcher constrains graph capture sizes",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8",
    ),
    (
        "Qwen BF16 launcher constrains graph capture sizes",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8",
    ),
    (
        "Qwen NVFP4 preserves rank-zero LMCache lookup",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "lookup_server_worker_ids: [0]",
    ),
    (
        "Qwen BF16 preserves rank-zero LMCache lookup",
        "tools/ds4_launch_qwen27_pp8.sh",
        "lookup_server_worker_ids: [0]",
    ),
    (
        "Qwen NVFP4 reasoning parser is opt-in",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        'QWEN27_REASONING_PARSER:-none',
    ),
    (
        "Qwen BF16 reasoning parser is opt-in",
        "tools/ds4_launch_qwen27_pp8.sh",
        'QWEN27_REASONING_PARSER:-none',
    ),
    (
        "Qwen3Next uses PPMissingLayer for non-owned embeddings/head",
        "vllm/model_executor/models/qwen3_next.py",
        "self.lm_head = PPMissingLayer()",
    ),
    (
        "Qwen3Next trace helper is outside Dynamo",
        "vllm/model_executor/models/qwen3_next.py",
        "@_compile_disabled",
    ),
]


def main() -> int:
    failed = False
    for description, relative_path, needle in CHECKS:
        path = ROOT / relative_path
        text = path.read_text()
        if needle in text:
            print(f"PASS: {description}")
        else:
            failed = True
            print(f"FAIL: {description}: missing {needle!r} in {relative_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
