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
        "BF16 Qwen PP disables GDN profile warmup by default",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "VLLM_QWEN_GDN_PROFILE_WARMUP:-0",
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
        "BF16 Qwen PP forces bounded GDN backend",
        ROOT / "tools/ds4_launch_qwen27_pp8.sh",
        "--gdn-prefill-backend",
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
        "NVFP4 Qwen PP disables torch compile by default",
        ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
        'QWEN27_COMPILATION_CONFIG:-\'{"mode":0,"cudagraph_mode":"NONE"}\'',
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
