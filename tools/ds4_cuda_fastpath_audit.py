#!/usr/bin/env python3
"""Static checks for DS4 CUDA fast-path changes.

This is intentionally source-text based. It catches accidental reversion of the
high-throughput SM12x sparse-MQA path before a Spark run spends time loading
models.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(path: str, *needles: str) -> None:
    text = (ROOT / path).read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(
            f"FAIL: {path} missing required fast-path markers: {missing}")
    print(f"PASS: {path}")


require(
    "vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py",
    "logits_out: torch.Tensor | None = None",
    "logits = logits_out[:num_rows, :token_count]",
    "logits = logits_out[:num_q, :seq_len_kv]",
)

require(
    "vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py",
    "_ds4_gather_values_i32_indices_kernel",
    "_ds4_gather_values_and_indices_i32_kernel",
    "VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT",
    "chunk_logits,",
    "_gather_values_and_indices_i32(",
    "logits_out=chunk_logits_buf",
)

require(
    "vllm/entrypoints/openai/completion/serving.py",
    "RequestOutputKind",
    "VLLM_DS4_FINAL_ONLY_NONSTREAMING",
    "sampling_params.output_kind = RequestOutputKind.FINAL_ONLY",
)

require(
    "vllm/v1/engine/core.py",
    "DS4 PP iteration timing",
    "future_wait_ms",
    "VLLM_DS4_ITERATION_TIMING",
)

require(
    "tools/ds4_launch_dsv4_flash_pp8.sh",
    "VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT",
    "VLLM_DS4_FINAL_ONLY_NONSTREAMING",
)

print("PASS: DS4 CUDA fast-path audit")
