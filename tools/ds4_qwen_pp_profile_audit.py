#!/usr/bin/env python3
"""Static DS4 Qwen PP bring-up audit."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    (
        "env exposes VLLM_QWEN_GDN_PROFILE_WARMUP",
        ROOT / "vllm/envs.py",
        "VLLM_QWEN_GDN_PROFILE_WARMUP",
    ),
    (
        "env exposes VLLM_DS4_QWEN_GDN_FIXED_TRITON_CONFIGS",
        ROOT / "vllm/envs.py",
        "VLLM_DS4_QWEN_GDN_FIXED_TRITON_CONFIGS",
    ),
    (
        "Qwen GDN warmup is skippable",
        ROOT / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "Skipping Qwen GDN prefill profile warmup",
    ),
    (
        "env exposes VLLM_DS4_PROFILE_RUN_MAX_TOKENS",
        ROOT / "vllm/envs.py",
        "VLLM_DS4_PROFILE_RUN_MAX_TOKENS",
    ),
    (
        "Qwen profile run token cap is applied",
        ROOT / "vllm/v1/worker/gpu_model_runner.py",
        "profile_num_tokens = min(profile_num_tokens, ds4_profile_run_max_tokens)",
    ),
    (
        "Qwen profile watchdog is available",
        ROOT / "vllm/v1/worker/gpu_worker.py",
        "ds4_profile_watchdog_context",
    ),
    (
        "Qwen PP layer trace is compile-safe",
        ROOT / "vllm/model_executor/models/qwen3_next.py",
        "@_compile_disabled",
    ),
    (
        "Qwen PP layer trace exists",
        ROOT / "vllm/model_executor/models/qwen3_next.py",
        "DS4 Qwen PP rank %d %s layer %d",
    ),
    (
        "Qwen3Next PP embed is not duplicated",
        ROOT / "vllm/model_executor/models/qwen3_next.py",
        "config.tie_word_embeddings and get_pp_group().is_last_rank",
    ),
    (
        "Qwen3Next lm_head is not duplicated",
        ROOT / "vllm/model_executor/models/qwen3_next.py",
        "self.lm_head = PPMissingLayer()",
    ),
    (
        "BF16 Qwen PP has explicit KV cap",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "--kv-cache-memory-bytes",
    ),
    (
        "BF16 Qwen PP enables GDN profile warmup by default",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_QWEN_GDN_PROFILE_WARMUP:-1",
    ),
    (
        "BF16 Qwen PP pins fixed GDN Triton configs by default",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_DS4_QWEN_GDN_FIXED_TRITON_CONFIGS:-1",
    ),
    (
        "BF16 Qwen PP bounds profile run by default",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_DS4_PROFILE_RUN_MAX_TOKENS:-512",
    ),
    (
        "BF16 Qwen PP skips DeepGEMM warmup by default",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_DEEP_GEMM_WARMUP:-skip",
    ),
    (
        "BF16 Qwen PP can disable LMCache for compute benchmarks",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_ENABLE_LMCACHE:-1",
    ),
    (
        "BF16 Qwen PP can disable hybrid KV manager for LMCache layout A/B tests",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "QWEN27_DISABLE_HYBRID_KV_CACHE_MANAGER:-0",
    ),
    (
        "BF16 Qwen PP uses LMCache V3 connector for grouped hybrid KV",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "LMCACHE_USE_GPU_CONNECTOR_V3:-true",
    ),
    (
        "BF16 Qwen PP uses deterministic LMCache chunk hashes",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "LMCACHE_PRE_CACHING_HASH_ALGORITHM:-sha256_cbor",
    ),
    (
        "BF16 Qwen PP uses rank-0 LMCache lookup oracle for multi-node IPC",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "lookup_server_worker_ids: [0]",
    ),
    (
        "BF16 Qwen PP forces bounded GDN backend",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "--gdn-prefill-backend",
    ),
    (
        "BF16 Qwen PP refuses async scheduling",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "Qwen PP async scheduling is disabled until the sync PP pipeline is stable",
    ),
    (
        "BF16 Qwen PP validates scheduled input ids before H2D",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_DS4_VALIDATE_INPUT_IDS",
    ),
    (
        "NVFP4 Qwen PP launcher exists",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "--quantization modelopt",
    ),
    (
        "NVFP4 Qwen PP disables MTP by default",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_NVFP4_ENABLE_MTP:-0",
    ),
    (
        "NVFP4 Qwen PP bounds profile run by default",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "VLLM_DS4_PROFILE_RUN_MAX_TOKENS:-512",
    ),
    (
        "NVFP4 Qwen PP skips DeepGEMM warmup by default",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "VLLM_DEEP_GEMM_WARMUP:-skip",
    ),
    (
        "NVFP4 Qwen PP can disable LMCache for compute benchmarks",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_ENABLE_LMCACHE:-1",
    ),
    (
        "NVFP4 Qwen PP can disable hybrid KV manager for LMCache layout A/B tests",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_DISABLE_HYBRID_KV_CACHE_MANAGER:-0",
    ),
    (
        "NVFP4 Qwen PP uses LMCache V3 connector for grouped hybrid KV",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "LMCACHE_USE_GPU_CONNECTOR_V3:-true",
    ),
    (
        "NVFP4 Qwen PP uses deterministic LMCache chunk hashes",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "LMCACHE_PRE_CACHING_HASH_ALGORITHM:-sha256_cbor",
    ),
    (
        "NVFP4 Qwen PP uses rank-0 LMCache lookup oracle for multi-node IPC",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "lookup_server_worker_ids: [0]",
    ),
    (
        "NVFP4 Qwen PP bounds CUDA graph capture sizes by default",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "QWEN27_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8,12",
    ),
    (
        "NVFP4 Qwen PP refuses async scheduling",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "Qwen PP async scheduling is disabled until the sync PP pipeline is stable",
    ),
    (
        "NVFP4 Qwen PP validates scheduled input ids before H2D",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        "VLLM_DS4_VALIDATE_INPUT_IDS",
    ),
]


def main() -> int:
    failed = False
    for description, path, needle in CHECKS:
        text = path.read_text() if path.exists() else ""
        if needle in text:
            print(f"PASS: {description}")
        else:
            failed = True
            print(f"FAIL: {description}: missing {needle!r} in {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
