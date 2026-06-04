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
        "Qwen NVFP4 async opt-in is refused",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "Qwen PP async scheduling is disabled until the sync PP pipeline is stable",
    ),
    (
        "Qwen BF16 async opt-in is refused",
        "tools/ds4_launch_qwen27_pp8.sh",
        "Qwen PP async scheduling is disabled until the sync PP pipeline is stable",
    ),
    (
        "Qwen NVFP4 validates scheduled input ids before H2D",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "VLLM_DS4_VALIDATE_INPUT_IDS",
    ),
    (
        "Qwen BF16 validates scheduled input ids before H2D",
        "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_DS4_VALIDATE_INPUT_IDS",
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
        "Qwen NVFP4 resident3 exposes 256K context",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_MAX_MODEL_LEN:=262144",
    ),
    (
        "Qwen BF16 resident3 exposes 256K context",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_MAX_MODEL_LEN:=262144",
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
        "Qwen NVFP4 resident3 max seqs allows 12-way batch",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_MAX_NUM_SEQS:=12",
    ),
    (
        "Qwen BF16 resident3 max seqs allows 12-way batch",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_MAX_NUM_SEQS:=12",
    ),
    (
        "Qwen NVFP4 launcher constrains graph capture sizes",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8,12",
    ),
    (
        "Qwen NVFP4 launcher keeps graph max consistent with 12-way capture",
        "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_MAX_CUDAGRAPH_CAPTURE_SIZE:=12",
    ),
    (
        "Qwen BF16 launcher constrains graph capture sizes",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8,12",
    ),
    (
        "Qwen BF16 launcher keeps graph max consistent with 12-way capture",
        "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_MAX_CUDAGRAPH_CAPTURE_SIZE:=12",
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
    (
        "DSV4 PP launcher defaults to multi-resident RAM profile",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        'DS4_DSV4_PIPELINE_RAM_PROFILE="${DS4_DSV4_PIPELINE_RAM_PROFILE:-resident3}"',
    ),
    (
        "DSV4 resident3 KV cap is compact",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_KV_CACHE_MEMORY_BYTES:=4294967296",
    ),
    (
        "DSV4 resident3 max batched tokens are bounded",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_MAX_NUM_BATCHED_TOKENS:=4096",
    ),
    (
        "DSV4 resident3 offload cap is compact",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_KV_OFFLOADING_SIZE:=2",
    ),
    (
        "DSV4 resident128 keeps compact KV while admitting eval-sized cohorts",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "resident128|RESIDENT128|resident-128|RESIDENT-128",
    ),
    (
        "DSV4 resident128 KV cap is bounded",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_KV_CACHE_MEMORY_BYTES:=8589934592",
    ),
    (
        "DSV4 resident128 batched tokens are bounded",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_MAX_NUM_BATCHED_TOKENS:=16384",
    ),
    (
        "DSV4 PP default derives balanced layer partition",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "base=$((43 / NNODES))",
    ),
    (
        "DSV4 PP default spreads partition remainder across early stages",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "rem=$((43 % NNODES))",
    ),
    (
        "DSV4 PP launcher bounds explicit GPU reservation before launch",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_MAX_EXPLICIT_GPU_RESERVATION_BYTES",
    ),
    (
        "DSV4 throughput profiles cannot silently fall back to auto partition",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "max-throughput|MAX-THROUGHPUT|batch512|BATCH512)",
    ),
    (
        "DSV4 validates custom PP layer partition",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_FLASH_PP_LAYER_PARTITION must sum to 43 DSV4 decoder layers",
    ),
    (
        "DSV4 prefix caching is an explicit launch choice",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "DSV4_ENABLE_PREFIX_CACHING",
    ),
    (
        "DSV4 prefix caching can be disabled for correctness isolation",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "--no-enable-prefix-caching",
    ),
    (
        "DSV4 launch banner records prefix caching state",
        "tools/ds4_launch_dsv4_flash_pp8.sh",
        "prefix_caching=$DSV4_ENABLE_PREFIX_CACHING",
    ),
    (
        "DSV4 skips MTP hidden buffer when speculation is disabled",
        "vllm/models/deepseek_v4/nvidia/model.py",
        "vllm_config.speculative_config is not None",
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
