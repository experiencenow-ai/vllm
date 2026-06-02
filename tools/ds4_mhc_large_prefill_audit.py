#!/usr/bin/env python3
"""Static checks for DS4 large-prefill mHC TileLang chunking."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
tilelang = (root / "vllm/model_executor/kernels/mhc/tilelang.py").read_text()
dsv4_pp8 = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
envs = (root / "vllm/envs.py").read_text()
deep_gemm = (root / "vllm/utils/deep_gemm.py").read_text()
preflight = (root / "tools/ds4_dsv4_native_preflight.py").read_text()

checks = [
    (
        "mHC TileLang max-token env is parsed",
        "VLLM_DS4_MHC_TILELANG_MAX_TOKENS" in tilelang
        and "def _ds4_mhc_tilelang_max_tokens() -> int:" in tilelang,
    ),
    (
        "mHC TileLang max-token env is registered with 8K default",
        "VLLM_DS4_MHC_TILELANG_MAX_TOKENS: int = 8192" in envs
        and 'os.environ.get("VLLM_DS4_MHC_TILELANG_MAX_TOKENS", "8192")' in envs,
    ),
    (
        "mHC pre path chunks large token slabs",
        "if max_tokens > 0 and num_tokens > max_tokens:" in tilelang
        and "chunk_post, chunk_comb, chunk_input = mhc_pre_tilelang(" in tilelang,
    ),
    (
        "mHC fused post/pre path chunks large token slabs",
        "chunk_residual," in tilelang
        and "mhc_fused_post_pre_tilelang(" in tilelang
        and "residual_cur[start:end].copy_(" in tilelang,
    ),
    (
        "DSV4 PP launcher sets chunk default",
        'export VLLM_DS4_MHC_TILELANG_MAX_TOKENS="${VLLM_DS4_MHC_TILELANG_MAX_TOKENS:-8192}"'
        in dsv4_pp8,
    ),
    (
        "DSV4 PP launcher logs chunk default",
        "mhc_tilelang_max_tokens=$VLLM_DS4_MHC_TILELANG_MAX_TOKENS" in dsv4_pp8,
    ),
    (
        "SM12x mHC prefers native DeepGEMM before Triton fallback",
        "if _tf32_hc_prenorm_gemm_impl is not None:" in deep_gemm
        and "return _tf32_hc_prenorm_gemm_impl(" in deep_gemm
        and "VLLM_DS4_MHC_ALLOW_TRITON_SM12X_FALLBACK" in deep_gemm,
    ),
    (
        "Triton mHC fallback is disabled by default in DSV4 PP launcher",
        'export VLLM_DS4_MHC_ALLOW_TRITON_SM12X_FALLBACK="${VLLM_DS4_MHC_ALLOW_TRITON_SM12X_FALLBACK:-0}"'
        in dsv4_pp8
        and "mhc_triton_fallback=$VLLM_DS4_MHC_ALLOW_TRITON_SM12X_FALLBACK"
        in dsv4_pp8,
    ),
    (
        "Native preflight probes DSV4-sized mHC slab",
        "hidden_size = 4096" in preflight
        and "VLLM_DS4_MHC_NATIVE_PREFLIGHT_TOKENS" in preflight
        and "compute_num_split(64, hc_hidden_size, cdiv(num_tokens, 64))"
        in preflight,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
