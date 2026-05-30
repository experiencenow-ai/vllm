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
cutlass = (root / "vllm/model_executor/kernels/linear/scaled_mm/cutlass.py").read_text()
modelopt = (root / "vllm/model_executor/layers/quantization/modelopt.py").read_text()
ds4_attention = (root / "vllm/models/deepseek_v4/nvidia/ops/attention.py").read_text()
ds4_fp8_einsum = (
    root / "vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py"
).read_text()
ds4_sm12x_fallbacks = (
    root / "vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py"
).read_text()
ds4_sm12x_mqa = (root / "vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py").read_text()
deep_gemm = (root / "vllm/utils/deep_gemm.py").read_text()
sparse_indexer = (
    root / "vllm/model_executor/layers/sparse_attn_indexer.py"
).read_text()
mla_indexer = (root / "vllm/v1/attention/backends/mla/indexer.py").read_text()
dsv4_tp2 = (root / "tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh").read_text()
dsv4_pp8 = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
qwen_pp8 = (root / "tools/ds4_launch_qwen27_pp8.sh").read_text()
guard = (root / "tools/ds4_200g_guard.sh").read_text()
triton_preflight = (root / "tools/ds4_triton_jit_preflight.py").read_text()

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
        "DeepGEMM MXFP4 SM12x opt-in env exists",
        "VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X" in mxfp4,
    ),
    (
        "FlashInfer TRTLLM MXFP4 SM12x opt-in env exists",
        "VLLM_DS4_ALLOW_FLASHINFER_TRTLLM_MXFP4_SM12X" in mxfp4,
    ),
    (
        "FlashInfer TRTLLM MXFP4 is disabled on Blackwell family-120 by default",
        "FlashInfer TRTLLM MXFP4 is disabled on CUDA Blackwell" in mxfp4
        and "family-120 by default because the GB10/SM121 runtime selected" in mxfp4
        and "sm100f TRTLLM batched GEMM runner" in mxfp4,
    ),
    (
        "Blackwell family-120 MXFP4 auto list prefers CUTLASS unless TRTLLM is opted in",
        "Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8" in mxfp4
        and "if _flashinfer_trtllm_mxfp4_allowed_on_current_device()" in mxfp4
        and "backends.insert(0, Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8)" in mxfp4,
    ),
    (
        "DeepGEMM MXFP4 is disabled on Blackwell family-120 by default",
        "DeepGEMM_MXFP4 is disabled on CUDA Blackwell family-120" in mxfp4
        and "csrc/apis/gemm.hpp:99" in mxfp4,
    ),
    (
        "Blackwell MXFP4 auto list does not force DeepGEMM",
        "backends.insert(1, Mxfp4MoeBackend.DEEPGEMM_MXFP4)" in mxfp4
        and "if _deepgemm_mxfp4_allowed_on_current_device()" in mxfp4,
    ),
    (
        "DeepGEMM FP8 linear SM12x opt-in guard exists",
        "VLLM_DS4_ALLOW_DEEPGEMM_FP8_LINEAR_SM12X" in linear
        and "csrc/apis/gemm.hpp:99" in linear,
    ),
    (
        "DeepGEMM FP8 linear kernels are rejected on Blackwell family-120",
        "DeepGemmFp8BlockScaledMMKernel" in linear
        and "FlashInferFp8DeepGEMMDynamicBlockScaledKernel" in linear
        and "_reject_deepgemm_fp8_linear_on_sm12x(kernel)" in linear,
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
    (
        "Cutlass FP8 block-scale path decodes E8M0 scales before stable ABI op",
        "def process_weights_after_loading(self, layer: torch.nn.Module)" in cutlass
        and "weight_scale.dtype != torch.float8_e8m0fnu" in cutlass
        and "_upcast_e8m0_to_fp32(weight_scale).contiguous()" in cutlass,
    ),
    (
        "SM12x DSV4 fp8_einsum uses DeepGEMM-supported FP32 scale recipe",
        "deepseek_v4_fp8_einsum_config(cap.major)" in ds4_attention
        and "def deepseek_v4_fp8_einsum_config(" in ds4_fp8_einsum
        and "return (1, 128, 128), False" in ds4_fp8_einsum,
    ),
    (
        "SM12x DSV4 fp8_einsum uses known-good Triton fallback",
        "def deepseek_v4_sm12x_fp8_einsum(" in ds4_fp8_einsum
        and "def _use_deepseek_v4_sm12x_triton_fp8_einsum(" in ds4_fp8_einsum
        and "deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, out)" in ds4_fp8_einsum,
    ),
    (
        "SM12x DSV4 MQA/HC fallbacks are present",
        "def fp8_paged_mqa_logits_triton(" in ds4_sm12x_mqa
        and "def tf32_hc_prenorm_gemm_triton(" in ds4_sm12x_mqa
        and "def fp8_fp4_paged_mqa_topk_indices(" in ds4_sm12x_fallbacks
        and "def _tf32_hc_prenorm_gemm_sm12x(" in ds4_sm12x_fallbacks,
    ),
    (
        "DeepGEMM wrapper routes SM12x MQA/HC through fallbacks",
        "def fp8_fp4_mqa_topk_indices(" in deep_gemm
        and "def fp8_fp4_paged_mqa_topk_indices(" in deep_gemm
        and "if current_platform.is_device_capability_family(120) and q[1] is None" in deep_gemm
        and "return _tf32_hc_prenorm_gemm_sm12x(x, fn, out, sqrsum, num_split)" in deep_gemm,
    ),
    (
        "Sparse indexer uses SM12x direct top-k and bounded logits",
        "fp8_fp4_mqa_topk_indices" in sparse_indexer
        and "fp8_fp4_paged_mqa_topk_indices" in sparse_indexer
        and "sparse_indexer_max_logits_bytes()" in sparse_indexer
        and "used_direct_topk = fp8_fp4_paged_mqa_topk_indices(" in sparse_indexer,
    ),
    (
        "MLA indexer avoids DeepGEMM scheduler metadata on SM12x",
        "def sparse_indexer_max_logits_bytes(" in mla_indexer
        and "def _uses_deep_gemm_scheduler_metadata(" in mla_indexer
        and "and not current_platform.is_device_capability_family(120)" in mla_indexer,
    ),

    (
        "DSV4 launchers prepare and preflight Triton JIT before serving",
        all(
            "ds4_prepare_triton_jit_environment" in script
            and "ds4_run_triton_jit_preflight" in script
            for script in (dsv4_tp2, dsv4_pp8)
        ),
    ),
    (
        "Qwen launcher inherits Triton JIT environment guard",
        "ds4_prepare_triton_jit_environment" in qwen_pp8
        and "ds4_run_triton_jit_preflight" in qwen_pp8,
    ),
    (
        "Triton JIT preflight checks gcc, Python.h, libcuda, and active launch",
        "check_python_headers" in triton_preflight
        and "check_libcuda_compile" in triton_preflight
        and "check_triton_active_jit" in triton_preflight
        and "_ds4_triton_launcher_probe" in triton_preflight,
    ),
    (
        "Triton JIT guard can use user-space Python dev headers",
        "ds4_prepare_python_include_environment" in guard
        and "DS4_PYTHON_INCLUDE_DIRS" in guard
        and "DS4_PYTHON_INCLUDE_DIRS" in triton_preflight,
    ),
    (
        "DSV4 launchers keep compilation config override as valid JSON",
        all(
            "DEFAULT_COMPILATION_CONFIG='" in script
            and 'DSV4_COMPILATION_CONFIG="${DSV4_COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"' in script
            and '--compilation-config "$DSV4_COMPILATION_CONFIG"' in script
            and 'DSV4_COMPILATION_CONFIG:-{\\"cudagraph_mode\\"' not in script
            for script in (dsv4_tp2, dsv4_pp8)
        ),
    ),
]

failed = False
for name, passed in checks:
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
    failed |= not passed

sys.exit(1 if failed else 0)
