#!/usr/bin/env python3
"""Static guardrail for GB10/SM12x native FP4/MXFP4 builds.

This does not replace live GPU startup checks. It prevents the specific
regressions where Blackwell auto-selection contains Marlin candidates, strict
rejection depends only on an environment variable, or Qwen ModelOpt NVFP4 can
directly instantiate a Marlin W4A16 path on GB10.
"""

from pathlib import Path
import sys


root = Path(__file__).resolve().parents[1]
mxfp4 = (root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py").read_text()
linear = (root / "vllm/model_executor/kernels/linear/__init__.py").read_text()
modelopt = (root / "vllm/model_executor/layers/quantization/modelopt.py").read_text()

checks = [
    (
        "Blackwell MXFP4 native list exists",
        "def _get_native_blackwell_mxfp4_backends" in mxfp4,
    ),
    (
        "Blackwell MXFP4 auto list is native-only",
        "if _is_cuda_blackwell():\n        return _get_native_blackwell_mxfp4_backends()" in mxfp4,
    ),
    (
        "generic MXFP4 selector avoids GPT-OSS fallback list on Blackwell",
        "_get_priority_backends()\n        if _is_cuda_blackwell()" in mxfp4,
    ),
    (
        "MXFP4 rejection is fail-closed on Blackwell without env requirement",
        "and not _is_cuda_blackwell()" in mxfp4,
    ),
    (
        "NVFP4 linear rejection is fail-closed on Blackwell without env requirement",
        "current_platform.is_device_capability_blackwell()" in linear
        and "not envs.VLLM_DS4_STRICT_NATIVE_FP4" in linear,
    ),
    (
        "Qwen ModelOpt W4A16_NVFP4 Marlin path is forbidden on Blackwell",
        "current_platform.is_cuda()" in modelopt
        and "current_platform.is_device_capability_blackwell()" in modelopt
        and "Native Blackwell FP4 mode rejected W4A16_NVFP4" in modelopt,
    ),
    (
        "explicit MXFP4 Marlin is forbidden on Blackwell",
        "VLLM_MXFP4_USE_MARLIN=1 is forbidden on CUDA Blackwell" in mxfp4,
    ),
]

failed = False
for name, passed in checks:
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
    failed |= not passed

sys.exit(1 if failed else 0)
