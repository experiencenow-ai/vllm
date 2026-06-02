#!/usr/bin/env python3
"""Static checks for DS4 large-prefill mHC TileLang chunking."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
tilelang = (root / "vllm/model_executor/kernels/mhc/tilelang.py").read_text()
dsv4_pp8 = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()

checks = [
    (
        "mHC TileLang max-token env is parsed",
        "VLLM_DS4_MHC_TILELANG_MAX_TOKENS" in tilelang
        and "def _ds4_mhc_tilelang_max_tokens() -> int:" in tilelang,
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
        'export VLLM_DS4_MHC_TILELANG_MAX_TOKENS="${VLLM_DS4_MHC_TILELANG_MAX_TOKENS:-65536}"'
        in dsv4_pp8,
    ),
    (
        "DSV4 PP launcher logs chunk default",
        "mhc_tilelang_max_tokens=$VLLM_DS4_MHC_TILELANG_MAX_TOKENS" in dsv4_pp8,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
