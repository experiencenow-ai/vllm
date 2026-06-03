#!/usr/bin/env python3
"""Static checks for DS4 sparse-attention CUDA graph support.

The DSV4 sparse prefill path builds dynamic indexer/MQA metadata. It may use
CUDA graphs for single-token decode, but must not keep PIECEWISE mixed/prefill
graphs when the attention backend only advertises single-token decode support.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(path: str, *needles: str) -> None:
    text = (ROOT / path).read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(
            f"FAIL: {path} missing CUDA graph support markers: {missing}")
    print(f"PASS: {path}")


require(
    "vllm/config/compilation.py",
    "cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE",
    "min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value",
    "setting cudagraph_mode=FULL_DECODE_ONLY",
)

require(
    "vllm/v1/attention/backends/mla/flashmla_sparse.py",
    "AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE",
)

require(
    "vllm/v1/attention/backends/mla/sparse_swa.py",
    "AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE",
)

require(
    "vllm/models/deepseek_v4/nvidia/ops/attention.py",
    "_slice_slot_mapping_to_q_rows",
    "graph-padded slot_mapping must be 1-D",
    "slot_mapping = _slice_slot_mapping_to_q_rows",
)

print("PASS: DS4 CUDA graph support audit")
