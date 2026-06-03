#!/usr/bin/env python3
"""Static checks for DS4 DSV4 HC head backend isolation."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
mhc = (root / "vllm/model_executor/layers/mhc.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()
probe = (root / "tools/ds4_mhc_correctness_probe.py").read_text()

checks = [
    (
        "HC head backend env is registered",
        "VLLM_DS4_DSV4_HC_HEAD_BACKEND: str = \"tilelang\"" in envs
        and "\"VLLM_DS4_DSV4_HC_HEAD_BACKEND\": lambda:" in envs,
    ),
    (
        "HCHeadOp reads DS4 backend env",
        "import vllm.envs as envs" in mhc
        and "backend = envs.VLLM_DS4_DSV4_HC_HEAD_BACKEND" in mhc,
    ),
    (
        "TileLang remains the production default",
        "if backend in (\"\", \"tilelang\"):" in mhc
        and "hc_head_fused_kernel_tilelang" in mhc,
    ),
    (
        "Triton head is explicit debug-only",
        "elif backend == \"triton-debug\":" in mhc
        and "torch.ops.vllm.hc_head_triton" in mhc,
    ),
    (
        "Unknown HC head backend fails closed",
        "Unsupported VLLM_DS4_DSV4_HC_HEAD_BACKEND" in mhc,
    ),
    (
        "HC head live reference check is env-gated",
        "VLLM_DS4_DSV4_HC_HEAD_REF_CHECK: bool = False" in envs
        and "\"VLLM_DS4_DSV4_HC_HEAD_REF_CHECK\": lambda:" in envs
        and "_maybe_check_hc_head_ref(" in mhc
        and "torch.cuda.is_current_stream_capturing()" in mhc
        and "DS4 DSV4 hc_head reference check failed" in mhc,
    ),
    (
        "MHC correctness probe does not require Triton head by default",
        "--include-triton-head" in probe
        and "DS4_MHC_CORRECTNESS_INCLUDE_TRITON_HEAD" in probe
        and "include_triton_head: bool" in probe,
    ),
    (
        "launcher exposes and logs HC head backend",
        'export VLLM_DS4_DSV4_HC_HEAD_BACKEND="${VLLM_DS4_DSV4_HC_HEAD_BACKEND:-tilelang}"'
        in launcher
        and "hc_head_backend=$VLLM_DS4_DSV4_HC_HEAD_BACKEND" in launcher,
    ),
    (
        "relaunch build validates HC head audit",
        "tools/ds4_dsv4_hc_head_backend_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
