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
dsv4_tp2_autotune = (
    root / "tools/ds4_launch_dsv4_flash_tp2_flashinfer_autotune.sh"
).read_text()
dsv4_pp8 = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
qwen_pp8 = (root / "tools/ds4_launch_qwen27_pp8.sh").read_text()
qwen_nvfp4_pp8 = (root / "tools/ds4_launch_qwen27_nvfp4_pp8.sh").read_text()
lmcache_adapter = (
    root
    / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"
).read_text()
guard = (root / "tools/ds4_200g_guard.sh").read_text()
triton_preflight = (root / "tools/ds4_triton_jit_preflight.py").read_text()
dual_pipeline_doc = (
    root / "docs/deployment/ds4-dual-8x-pipelines.md"
).read_text()

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
        "DeepSeek V4 CUTLASS MXFP4 converter handles DSV4 weight layout",
        "DSV4 loading gives contiguous [w1/gate, w3/up]" in mxfp4
        and "Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8" in mxfp4
        and "block_scale_interleave(" in mxfp4,
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
        "Qwen BF16 launcher is N-way and stage-local for LMCache",
        "--pipeline-parallel-size \"$NNODES\"" in qwen_pp8
        and "--distributed-executor-backend mp" in qwen_pp8
        and "QWEN27_PP_LAYER_PARTITION must sum to 64 Qwen decoder layers" in qwen_pp8
        and "DEFAULT_LMCACHE_ROOT=\"$HOME/ds4_lmcache/qwen27_bf16_pp${NNODES}/${DS4_NODE_ID}\"" in qwen_pp8,
    ),
    (
        "Qwen launchers hardfail without 200G/NCCL preflight",
        all(
            "ds4_require_200g_fabric" in script
            and "ds4_run_nccl_preflight \"$NNODES\"" in script
            for script in (qwen_pp8, qwen_nvfp4_pp8)
        ),
    ),
    (
        "Qwen NVFP4 cache-primary launcher uses ModelOpt FP4 without MTP by default",
        "--quantization modelopt" in qwen_nvfp4_pp8
        and "--linear-backend \"${QWEN27_LINEAR_BACKEND:-flashinfer-cutlass}\"" in qwen_nvfp4_pp8
        and "QWEN27_KV_CACHE_DTYPE=\"${QWEN27_KV_CACHE_DTYPE:-fp8}\"" in qwen_nvfp4_pp8
        and "QWEN27_ATTENTION_BACKEND=\"${QWEN27_ATTENTION_BACKEND:-TRITON_ATTN}\"" in qwen_nvfp4_pp8
        and "--attention-backend \"$QWEN27_ATTENTION_BACKEND\"" in qwen_nvfp4_pp8
        and "FlashAttention rejects fp8 KV cache" in qwen_nvfp4_pp8
        and "QWEN27_ALLOW_FLASHINFER_ATTENTION_EXPERIMENTAL" in qwen_nvfp4_pp8
        and "--kv-cache-dtype \"$QWEN27_KV_CACHE_DTYPE\"" in qwen_nvfp4_pp8
        and "QWEN27_NVFP4_ENABLE_MTP_EXPERIMENTAL" in qwen_nvfp4_pp8
        and "SPEC_ARGS=()" in qwen_nvfp4_pp8,
    ),
    (
        "Qwen NVFP4 launcher uses native LMCache HMA with stage-local roots",
        "LMCacheConnectorV1" in qwen_nvfp4_pp8
        and '"use_native":true' in qwen_nvfp4_pp8
        and '"lmcache_kv_cache_group_id":"auto"' in qwen_nvfp4_pp8
        and "DEFAULT_LMCACHE_ROOT=\"$HOME/ds4_lmcache/qwen27_nvfp4_pp${PP_SIZE}/${DS4_NODE_ID}\"" in qwen_nvfp4_pp8,
    ),
    (
        "Qwen launchers keep bounded coexistence KV defaults",
        "max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE:-2.0}" in qwen_pp8
        and "max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE:-2.0}" in qwen_nvfp4_pp8
        and "QWEN27_KV_CACHE_MEMORY_BYTES=\"${QWEN27_KV_CACHE_MEMORY_BYTES:-8589934592}\"" in qwen_pp8
        and "QWEN27_KV_CACHE_MEMORY_BYTES=\"${QWEN27_KV_CACHE_MEMORY_BYTES:-8589934592}\"" in qwen_nvfp4_pp8
        and "--kv-cache-memory-bytes \"$QWEN27_KV_CACHE_MEMORY_BYTES\"" in qwen_pp8
        and "--kv-cache-memory-bytes \"$QWEN27_KV_CACHE_MEMORY_BYTES\"" in qwen_nvfp4_pp8
        and "QWEN27_ENFORCE_EAGER" in qwen_pp8
        and "QWEN27_ENFORCE_EAGER" in qwen_nvfp4_pp8
        and "EAGER_ARGS=(--enforce-eager)" in qwen_pp8
        and "EAGER_ARGS=(--enforce-eager)" in qwen_nvfp4_pp8
        and '--gpu-memory-utilization "${QWEN27_GPU_MEMORY_UTILIZATION:-0.24}"' in qwen_nvfp4_pp8
        and "ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE" in qwen_pp8
        and "ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE" in qwen_nvfp4_pp8
        and "LMCACHE_MAX_LOCAL_CPU_SIZE=2.0" in dual_pipeline_doc
        and "QWEN27_KV_CACHE_MEMORY_BYTES=8589934592" in dual_pipeline_doc
        and "DS4_ENABLE_FLASHINFER_AUTOTUNE=0" in dual_pipeline_doc,
    ),
    (
        "DS4 service launchers fail closed on FlashInfer autotune",
        all(
            "ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE" in script
            and '"${FLASHINFER_AUTOTUNE_ARGS[@]}"' in script
            for script in (dsv4_tp2, dsv4_pp8, qwen_pp8, qwen_nvfp4_pp8)
        )
        and "production/validation launcher" in guard
        and "DS4_FLASHINFER_AUTOTUNE_TUNING_JOB" in guard,
    ),
    (
        "FlashInfer autotune is isolated to an explicit tuning wrapper",
        "DS4_FLASHINFER_AUTOTUNE_TUNING_JOB=1" in dsv4_tp2_autotune
        and "DS4_ENABLE_FLASHINFER_AUTOTUNE=1" in dsv4_tp2_autotune
        and "not a production service launcher" in dsv4_tp2_autotune,
    ),
    (
        "Launchers bound FlashInfer runtime JIT parallelism during bringup",
        "ds4_prepare_flashinfer_jit_environment()" in guard
        and 'DS4_FLASHINFER_JIT_MAX_JOBS:-1' in guard
        and "export MAX_JOBS=\"${MAX_JOBS:-$max_jobs}\"" in guard
        and all(
            "ds4_prepare_flashinfer_jit_environment" in script
            for script in (dsv4_tp2, dsv4_pp8, qwen_pp8, qwen_nvfp4_pp8)
        ),
    ),
    (
        "DSV4 launchers can disable MTP for memory bringup",
        all(
            "DSV4_DISABLE_MTP" in script
            and "SPECULATIVE_ARGS=()" in script
            and '"${SPECULATIVE_ARGS[@]}"' in script
            for script in (dsv4_tp2, dsv4_pp8)
        ),
    ),
    (
        "DSV4 PP8 launcher keeps bounded coexistence KV defaults",
        "DSV4_KV_CACHE_MEMORY_BYTES=\"${DSV4_KV_CACHE_MEMORY_BYTES:-12884901888}\"" in dsv4_pp8
        and "--kv-cache-memory-bytes \"$DSV4_KV_CACHE_MEMORY_BYTES\"" in dsv4_pp8
        and '--max-num-batched-tokens "${DSV4_MAX_NUM_BATCHED_TOKENS:-16384}"' in dsv4_pp8
        and "DSV4_KV_CACHE_MEMORY_BYTES=12884901888" in dual_pipeline_doc,
    ),
    (
        "LMCache lookup client/server receive LMCache metadata, not VllmConfig",
        "def _build_lmcache_metadata_from_vllm_config(" in lmcache_adapter
        and "LookupClientFactory.create_lookup_client(\n                config, lookup_metadata\n            )" in lmcache_adapter
        and "lookup_metadata = getattr(self.lmcache_engine, \"metadata\", None)" in lmcache_adapter
        and "LookupClientFactory.create_lookup_server(\n                self.lmcache_engine, lookup_metadata\n            )" in lmcache_adapter,
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
        "Triton JIT guard keeps LMCache IPC socket paths short",
        "default_ipc_tmp=\"/tmp/d4i/${MASTER_PORT:-0}_${NODE_RANK:-x}\"" in guard
        and "export TMPDIR=\"${TMPDIR:-$ipc_tmp}\"" in guard
        and "LMCache ZMQ IPC sockets require TMPDIR <= 29 chars" in guard,
    ),
    (
        "DS4 Spark recipes do not default to /mnt/nvme",
        "/mnt/nvme" not in guard
        and "/mnt/nvme" not in qwen_pp8
        and "/mnt/nvme" not in qwen_nvfp4_pp8
        and "/mnt/nvme" not in dsv4_pp8
        and "/mnt/nvme" not in dual_pipeline_doc,
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
