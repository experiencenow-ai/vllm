#!/usr/bin/env python3
"""Print the DS4-relevant native Blackwell backend gates.

Run inside the same container/env that will start vLLM. This script does not
load a model; it only reports the hardware/library gates that decide whether
DeepSeek-V4 can reach native MXFP4/FP8 paths instead of Marlin.
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import (
    has_flashinfer,
    has_flashinfer_cutlass_fused_moe,
    has_flashinfer_trtllm_fused_moe,
)
from vllm.utils.import_utils import has_deep_gemm


def _bool(value: bool) -> str:
    return "yes" if value else "no"


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - diagnostic tool
        return f"not importable ({type(exc).__name__}: {exc})"
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else "importable"


def _is_true(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _probe_mhc_tilelang() -> tuple[bool, str]:
    try:
        import vllm.model_executor.layers.mhc  # noqa: F401

        torch.cuda.set_device(0)
        hidden_size = int(os.getenv("DS4_MHC_PROBE_HIDDEN_SIZE", "7168"))
        hc_mult = int(os.getenv("DS4_MHC_PROBE_HC_MULT", "2"))
        num_tokens = int(os.getenv("DS4_MHC_PROBE_TOKENS", "1"))
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        residual = torch.randn(
            num_tokens,
            hc_mult,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        fn = torch.randn(mix_hc, hc_dim, dtype=torch.float32, device="cuda")
        hc_scale = torch.ones(3, dtype=torch.float32, device="cuda")
        hc_base = torch.zeros(mix_hc, dtype=torch.float32, device="cuda")
        torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            1.0e-6,
            1.0e-6,
            1.0e-6,
            2.0,
            2,
            1,
            None,
            0.0,
        )
        torch.cuda.synchronize()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic preflight
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    strict_dsv4 = "--strict-dsv4" in sys.argv[1:]
    print("DS4 native Blackwell probe")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST', '<unset>')}")
    print(f"VLLM_DS4_STRICT_NATIVE_FP4: {os.getenv('VLLM_DS4_STRICT_NATIVE_FP4', '<unset>')}")
    print(f"VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X: {os.getenv('VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X', '<unset>')}")
    print(f"VLLM_DS4_ALLOW_DEEPGEMM_FP8_LINEAR_SM12X: {os.getenv('VLLM_DS4_ALLOW_DEEPGEMM_FP8_LINEAR_SM12X', '<unset>')}")
    print(f"VLLM_MXFP4_USE_MARLIN: {os.getenv('VLLM_MXFP4_USE_MARLIN', '<unset>')}")
    print(f"VLLM_USE_DEEP_GEMM: {os.getenv('VLLM_USE_DEEP_GEMM', '<unset>')}")
    print(f"VLLM_USE_DEEP_GEMM_E8M0: {os.getenv('VLLM_USE_DEEP_GEMM_E8M0', '<unset>')}")
    print()

    if strict_dsv4 and _is_true(os.getenv("VLLM_MXFP4_USE_MARLIN")):
        print(
            "strict_dsv4_error: VLLM_MXFP4_USE_MARLIN requests Marlin fallback",
            file=sys.stderr,
        )
        return 3

    if not torch.cuda.is_available():
        print("cuda_available: no")
        return 2

    capability = current_platform.get_device_capability()
    print(f"device_name: {current_platform.get_device_name()}")
    print(f"capability: {capability.as_version_str() if capability else '<unknown>'}")
    print(f"capability_int: {capability.to_int() if capability else '<unknown>'}")
    print(f"is_cuda: {_bool(current_platform.is_cuda())}")
    print(f"is_blackwell: {_bool(current_platform.is_device_capability_blackwell())}")
    print(f"family_100: {_bool(current_platform.is_device_capability_family(100))}")
    print(f"family_120: {_bool(current_platform.is_device_capability_family(120))}")
    print()

    print(f"deep_gemm module: {_module_version('deep_gemm')}")
    print(f"has_deep_gemm: {_bool(has_deep_gemm())}")
    print(f"is_deep_gemm_supported: {_bool(is_deep_gemm_supported())}")
    print(f"flashinfer module: {_module_version('flashinfer')}")
    print(f"has_flashinfer: {_bool(has_flashinfer())}")
    print(f"has_flashinfer_trtllm_fused_moe: {_bool(has_flashinfer_trtllm_fused_moe())}")
    print(f"has_flashinfer_cutlass_fused_moe: {_bool(has_flashinfer_cutlass_fused_moe())}")
    print()

    native_ready = (
        current_platform.is_cuda()
        and current_platform.is_device_capability_blackwell()
        and (is_deep_gemm_supported() or has_flashinfer_trtllm_fused_moe() or has_flashinfer_cutlass_fused_moe())
    )
    print(f"native_mxfp4_candidate_ready: {_bool(native_ready)}")
    if strict_dsv4 and not native_ready:
        print(
            "strict_dsv4_error: no native Blackwell MXFP4/FP8 candidate is ready",
            file=sys.stderr,
        )
        return 1

    mhc_ready, mhc_detail = _probe_mhc_tilelang()
    print(f"native_mhc_tilelang_ready: {_bool(mhc_ready)}")
    print(f"native_mhc_tilelang_detail: {mhc_detail}")
    if strict_dsv4 and not mhc_ready:
        print(
            "strict_dsv4_error: native MHC/TileLang/DeepGEMM preflight failed; "
            "this build would either fail during vLLM profiling or require a "
            "slower MHC fallback, so DS4 strict native startup is stopped",
            file=sys.stderr,
        )
        return 4

    return 0 if native_ready and mhc_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
