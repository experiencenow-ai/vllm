#!/usr/bin/env python3
"""Static audit for DSV4 MXFP4 SwiGLU activation parameters."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
flashinfer = (
    root / "vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py"
).read_text()
gpt_oss_method = (
    root / "vllm/model_executor/layers/quantization/mxfp4.py"
).read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "FlashInfer MXFP4 defaults missing SwiGLU alpha to standard DSV4 value",
        "if quant_config.gemm1_alpha is not None" in flashinfer
        and "else 1.0" in flashinfer
        and "[gemm1_alpha] * self.num_experts" in flashinfer,
    ),
    (
        "FlashInfer MXFP4 defaults missing SwiGLU beta to standard value",
        "if quant_config.gemm1_beta is not None" in flashinfer
        and "[gemm1_beta] * self.num_experts" in flashinfer,
    ),
    (
        "GPT-OSS MXFP4 still passes explicit OpenAI SwiGLU alpha",
        "class GptOssMxfp4MoEMethod" in gpt_oss_method
        and "gemm1_alpha=1.702" in gpt_oss_method
        and "gemm1_beta=1.0" in gpt_oss_method,
    ),
    (
        "DSV4 MXFP4 method does not hard-code GPT-OSS SwiGLU alpha",
        "class Mxfp4MoEMethod" in gpt_oss_method
        and "swiglu_limit = getattr(layer, \"swiglu_limit\", None)" in gpt_oss_method
        and "swiglu_limit=swiglu_limit" in gpt_oss_method,
    ),
    (
        "relaunch build validates DSV4 MXFP4 SwiGLU audit",
        "tools/ds4_dsv4_mxfp4_swiglu_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
