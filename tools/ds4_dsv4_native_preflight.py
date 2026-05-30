#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast validation for DeepSeek-V4 native GB10 kernel coverage.

This checks the kernel families that matter before vLLM spends minutes loading
DeepSeek-V4-Flash. It intentionally treats missing native SM12x support as a
startup failure. Marlin/emulation fallback is not a production path.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable

REQUIRED_DEEPGEMM_SYMBOLS = (
    "tf32_hc_prenorm_gemm",
    "get_paged_mqa_logits_metadata",
    "fp8_fp4_paged_mqa_logits",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
)

REQUIRED_SM12X_MARKERS = (
    "sm120_tf32_hc_prenorm_gemm",
    "sm120_fp8_paged_mqa_logits",
)

REQUIRED_SM12X_LAYOUT_MARKERS = (
    "arch_major == 10 or arch_major == 12",
    "Unknown SF transformation",
)

INSTALL_HINT = """
Install the GB10/SM12x-capable DeepGEMM fork, then rebuild/restart vLLM:

  tools/ds4_install_deepgemm_gb10_native.sh

That script applies the DS4 SM12x DeepGEMM layout patch before building.
Without that patch, DeepGEMM can import and still fail at load time with:

  Unknown SF transformation

For a full vLLM wheel/container rebuild, export these before building:

  export DEEPGEMM_GIT_REPO=https://github.com/jasl/DeepGEMM.git
  export DEEPGEMM_GIT_REF=7a7a41a1
  export VLLM_DS4_PATCH_DEEPGEMM_SM12X=1
  export TORCH_CUDA_ARCH_LIST=12.1a

The installer applies the DS4 SM12x layout patch before building DeepGEMM.
""".strip()


def emit(label: str, value: object) -> None:
    print(f"{label}: {value}", flush=True)


def import_first_available(names: Iterable[str]):
    last_exc: BaseException | None = None
    for name in names:
        try:
            return importlib.import_module(name)
        except BaseException as exc:  # Import may raise RuntimeError from _C.
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ImportError("no module names supplied")


def module_search_roots(module) -> list[Path]:
    roots: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        roots.append(Path(module_file).resolve().parent)
    for root in list(roots):
        for extra in (root.parent, root.parent / "include", root / "include"):
            if extra.exists():
                roots.append(extra.resolve())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen and root.exists():
            seen.add(root)
            deduped.append(root)
    return deduped


def file_may_contain_marker(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size > 64 * 1024 * 1024:
        return False
    return path.suffix in {
        "",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".py",
        ".so",
        ".txt",
    }


def find_marker(roots: Iterable[Path], marker: str) -> Path | None:
    marker_bytes = marker.encode("utf-8")
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            try:
                if not file_may_contain_marker(path):
                    continue
                if marker_bytes in path.read_bytes():
                    return path
            except OSError:
                continue
    return None


def check_torch_cuda() -> bool:
    try:
        import torch
    except BaseException as exc:
        emit("torch_import", f"no: {exc}")
        return False

    emit("torch", getattr(torch, "__version__", "unknown"))
    emit("torch_cuda", getattr(torch.version, "cuda", None))
    if not torch.cuda.is_available():
        emit("cuda_available", "no")
        return False

    capability = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    emit("cuda_device", name)
    emit("cuda_capability", f"{capability[0]}.{capability[1]}")
    if capability[0] != 12:
        emit("sm12x", "no")
        return False
    emit("sm12x", "yes")
    return True


def check_vllm_platform() -> bool:
    try:
        from vllm.platforms import current_platform
    except BaseException as exc:
        emit("vllm_platform", f"no: {exc}")
        return False

    try:
        emit("vllm_is_cuda", current_platform.is_cuda())
        emit("vllm_is_blackwell", current_platform.is_device_capability_blackwell())
        emit("vllm_support_deep_gemm", current_platform.support_deep_gemm())
        return bool(
            current_platform.is_cuda()
            and current_platform.is_device_capability_blackwell()
            and current_platform.support_deep_gemm()
        )
    except BaseException as exc:
        emit("vllm_platform_probe", f"no: {exc}")
        return False


def check_deep_gemm_package() -> bool:
    try:
        module = import_first_available(("deep_gemm", "vllm.third_party.deep_gemm"))
    except BaseException as exc:
        emit("deep_gemm_import", f"no: {exc}")
        print(INSTALL_HINT, file=sys.stderr)
        return False

    emit("deep_gemm_module", module.__name__)
    emit("deep_gemm_file", getattr(module, "__file__", "unknown"))

    ok = True
    for symbol in REQUIRED_DEEPGEMM_SYMBOLS:
        has_symbol = hasattr(module, symbol)
        emit(f"deep_gemm_symbol_{symbol}", "yes" if has_symbol else "no")
        ok = ok and has_symbol

    roots = module_search_roots(module)
    emit("deep_gemm_search_roots", ":".join(str(root) for root in roots))
    for marker in REQUIRED_SM12X_MARKERS:
        hit = find_marker(roots, marker)
        emit(f"deep_gemm_marker_{marker}", hit if hit else "no")
        ok = ok and hit is not None

    for marker in REQUIRED_SM12X_LAYOUT_MARKERS:
        # Source markers are helpful when the wheel vendors headers, but some
        # DeepGEMM wheels do not ship csrc. The active CUDA layout probe below
        # is the authoritative check for this failure mode.
        hit = find_marker(roots, marker)
        emit(f"deep_gemm_layout_marker_{marker}", hit if hit else "not_in_wheel")

    if not ok:
        print(INSTALL_HINT, file=sys.stderr)
    return ok


def active_probe_sf_layout() -> bool:
    try:
        import torch
        from vllm.utils.deep_gemm import transform_sf_into_required_layout
    except BaseException as exc:
        emit("active_sf_layout_import", f"no: {exc}")
        return False

    try:
        device = torch.device("cuda:0")

        # DS4 MegaMoE weight-scale path: recipe=(1, 32), FP32 source scales,
        # E8M0 packing enabled. This catches DeepGEMM csrc/apis/layout.hpp
        # "Unknown SF transformation" before model load.
        sf_moe = torch.ones((1, 1, 1), device=device, dtype=torch.float32)
        transformed_moe = transform_sf_into_required_layout(
            sf_moe,
            1,
            32,
            (1, 32),
            1,
        )
        torch.cuda.synchronize()
        emit(
            "active_sf_layout_moe_recipe_1x32",
            f"yes:{tuple(transformed_moe.shape)}:{transformed_moe.dtype}",
        )

        # Generic FP8 BMM/GEMM post-process path used by fp8_utils.py.
        sf_fp8 = torch.ones((1, 1, 1), device=device, dtype=torch.float32)
        transformed_fp8 = transform_sf_into_required_layout(
            sf_fp8,
            1,
            128,
            (1, 128, 128),
            1,
            False,
        )
        torch.cuda.synchronize()
        emit(
            "active_sf_layout_fp8_recipe_1x128x128",
            f"yes:{tuple(transformed_fp8.shape)}:{transformed_fp8.dtype}",
        )
        return True
    except BaseException as exc:
        emit("active_sf_layout", f"no: {exc}")
        return False


def active_probe_mhc() -> bool:
    try:
        import torch
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    except BaseException as exc:
        emit("active_mhc_import", f"no: {exc}")
        return False

    try:
        device = torch.device("cuda:0")
        hc_mult = 4
        hidden_size = 1280
        num_tokens = 64
        hc_mult2 = hc_mult * hc_mult
        hc_mult3 = hc_mult * 2 + hc_mult2
        x = torch.randn(
            (num_tokens, hc_mult * hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )
        fn = torch.randn(
            (hc_mult3, hc_mult * hidden_size),
            device=device,
            dtype=torch.float32,
        )
        n_splits = 1
        out = torch.empty(
            (n_splits, num_tokens, hc_mult3),
            device=device,
            dtype=torch.float32,
        )
        sqrsum = torch.empty(
            (n_splits, num_tokens),
            device=device,
            dtype=torch.float32,
        )
        tf32_hc_prenorm_gemm(x, fn, out, sqrsum, n_splits)
        torch.cuda.synchronize()
        emit("active_mhc_tf32_hc_prenorm_gemm", "yes")
        return True
    except BaseException as exc:
        emit("active_mhc_tf32_hc_prenorm_gemm", f"no: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-active-layout-probe",
        action="store_true",
        help="Skip the small DeepGEMM SF layout CUDA probe.",
    )
    parser.add_argument(
        "--active-kernel-probe",
        action="store_true",
        help="Run a small tf32_hc_prenorm_gemm CUDA call.",
    )
    args = parser.parse_args()

    ok = True
    ok = check_torch_cuda() and ok
    ok = check_vllm_platform() and ok
    ok = check_deep_gemm_package() and ok
    if not args.skip_active_layout_probe:
        ok = active_probe_sf_layout() and ok
    if args.active_kernel_probe:
        ok = active_probe_mhc() and ok

    emit("ds4_dsv4_native_preflight", "pass" if ok else "fail")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
