#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static audit for DS4 DSV4 K-cache gather/dequant policy."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def check(label: str, condition: bool) -> bool:
    print(("PASS" if condition else "FAIL") + f": {label}")
    return condition


def main() -> int:
    cache_utils = read("vllm/models/deepseek_v4/common/ops/cache_utils.py")
    envs = read("vllm/envs.py")
    pp8 = read("tools/ds4_launch_dsv4_flash_pp8.sh")
    tp2 = read("tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh")
    preflight = read("tools/ds4_dsv4_native_preflight.py")

    ok = True
    ok &= check(
        "DSV4 K gather backend env exists",
        "VLLM_DS4_DSV4_K_GATHER_BACKEND" in envs,
    )
    ok &= check(
        "DSV4 Triton gather debug opt-in env exists",
        "VLLM_DS4_DSV4_ALLOW_TRITON_GATHER_DEBUG" in envs,
    )
    ok &= check(
        "strict native auto resolves to cutedsl",
        "if envs.VLLM_DS4_STRICT_NATIVE_FP4" in cache_utils
        and 'return "cutedsl"' in cache_utils,
    )
    ok &= check(
        "missing CuteDSL raises instead of falling through",
        "DSV4 K-cache gather/dequant requires CuteDSL" in cache_utils
        and "Refusing to fall through" in cache_utils,
    )
    ok &= check(
        "Triton gather is debug-only under explicit opt-in",
        "triton-debug" in cache_utils
        and "VLLM_DS4_DSV4_ALLOW_TRITON_GATHER_DEBUG=1" in cache_utils,
    )
    ok &= check(
        "DSV4 PP8 launcher defaults to CuteDSL gather",
        'VLLM_DS4_DSV4_K_GATHER_BACKEND="${VLLM_DS4_DSV4_K_GATHER_BACKEND:-cutedsl}"'
        in pp8,
    )
    ok &= check(
        "DSV4 TP2 launcher defaults to CuteDSL gather",
        'VLLM_DS4_DSV4_K_GATHER_BACKEND="${VLLM_DS4_DSV4_K_GATHER_BACKEND:-cutedsl}"'
        in tp2,
    )
    ok &= check(
        "native preflight checks CuteDSL gather path",
        "check_cutedsl_gather_path" in preflight
        and "dequant_gather_k_cutedsl" in preflight,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
