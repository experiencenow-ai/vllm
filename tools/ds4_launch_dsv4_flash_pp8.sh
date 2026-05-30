#!/usr/bin/env bash
set -euo pipefail

NNODES="${NNODES:-8}"
: "${NODE_RANK:?set NODE_RANK to the local pipeline rank}"
: "${HEAD_ADDR:?set HEAD_ADDR to the rank-0 Spark private IP or hostname}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ds4_200g_guard.sh"
MASTER_PORT="${MASTER_PORT:-29544}"
API_PORT="${API_PORT:-8102}"
MODEL="${DSV4_FLASH_MODEL:-/home/$USER/models/hf/deepseek-ai/DeepSeek-V4-Flash}"
RUNTIME_PYTHON="${DS4_VLLM_PYTHON:-/home/$USER/ds4-vllm-local/bin/python}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-/home/$USER/src/vllm}"
DEFAULT_SPECULATIVE_CONFIG="{\"model\":\"$MODEL\",\"num_speculative_tokens\":2,\"method\":\"deepseek_mtp\"}"
DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
DSV4_LINEAR_BACKEND="${DSV4_LINEAR_BACKEND:-auto}"
DSV4_MOE_BACKEND="${DSV4_MOE_BACKEND:-auto}"
DSV4_COMPILATION_CONFIG="${DSV4_COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"
DSV4_KV_CACHE_MEMORY_BYTES="${DSV4_KV_CACHE_MEMORY_BYTES:-12884901888}"
SPECULATIVE_ARGS=(--speculative-config "${DSV4_SPECULATIVE_CONFIG:-$DEFAULT_SPECULATIVE_CONFIG}")
if [[ "${DSV4_DISABLE_MTP:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  SPECULATIVE_ARGS=()
fi
ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_TRITON_MLA_SPARSE="${VLLM_TRITON_MLA_SPARSE:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"
export VLLM_DS4_STRICT_NATIVE_FP4="${VLLM_DS4_STRICT_NATIVE_FP4:-1}"
export VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X="${VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X:-0}"
export VLLM_DS4_ALLOW_DEEPGEMM_FP8_LINEAR_SM12X="${VLLM_DS4_ALLOW_DEEPGEMM_FP8_LINEAR_SM12X:-0}"
if [[ "${VLLM_MXFP4_USE_MARLIN:-}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  echo "DS4 strict native mode refuses VLLM_MXFP4_USE_MARLIN=$VLLM_MXFP4_USE_MARLIN" >&2
  exit 64
fi
export VLLM_MXFP4_USE_MARLIN=0
if [[ "${VLLM_TEST_FORCE_FP8_MARLIN:-}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  echo "DS4 strict native mode refuses VLLM_TEST_FORCE_FP8_MARLIN=$VLLM_TEST_FORCE_FP8_MARLIN" >&2
  exit 64
fi
export VLLM_TEST_FORCE_FP8_MARLIN=0
export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:-MarlinNvFp4LinearKernel,EmulationNvFp4LinearKernel,MarlinMxFp4LinearKernel,MarlinMxfp8LinearKernel,EmulationMxfp8LinearKernel,MarlinFP8ScaledMMLinearKernel}"
export DS4_200G_IFNAME="${DS4_200G_IFNAME:-enP2p1s0f0np0,enP2p1s0f1np1}"
export DS4_CONTROL_IFNAME="${DS4_CONTROL_IFNAME:-ds4ring0}"
export DS4_200G_ADVERTISE_LOOPBACK="${DS4_200G_ADVERTISE_LOOPBACK:-1}"
export DS4_200G_NCCL_TRANSPORT="${DS4_200G_NCCL_TRANSPORT:-socket}"
export VLLM_DS4_PP_ONLY_GLOBAL_BACKEND="${VLLM_DS4_PP_ONLY_GLOBAL_BACKEND:-gloo}"
export VLLM_DS4_SKIP_PYNCCL_WARMUP_ALLREDUCE="${VLLM_DS4_SKIP_PYNCCL_WARMUP_ALLREDUCE:-1}"
export DS4_NCCL_PREFLIGHT_MODE="${DS4_NCCL_PREFLIGHT_MODE:-nccl}"
if [[ "$NODE_RANK" == "0" ]]; then
  export DS4_200G_ALLOW_LOOPBACK_HEAD="${DS4_200G_ALLOW_LOOPBACK_HEAD:-1}"
fi
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export DS4_NATIVE_PREFLIGHT_ACTIVE="${DS4_NATIVE_PREFLIGHT_ACTIVE:-1}"
ds4_prepare_triton_jit_environment "dsv4-flash-pp${NNODES}"
ds4_prepare_flashinfer_jit_environment
ds4_require_200g_fabric
ds4_run_nccl_preflight "$NNODES"
ds4_run_dsv4_native_preflight
ds4_run_native_blackwell_preflight
ds4_run_triton_jit_preflight

export VLLM_USE_SIMPLE_KV_OFFLOAD="${VLLM_USE_SIMPLE_KV_OFFLOAD:-1}"
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT:-$HOME/ds4_hma_store/dsv4_flash_pp8/simple_cpu_offload}"
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT:-1}"
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK:-$(hostname)-dsv4-pp8-r${NODE_RANK}}"
mkdir -p "$VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT"

if [[ -n "${DSV4_FLASH_PP_LAYER_PARTITION:-}" ]]; then
  export VLLM_PP_LAYER_PARTITION="$DSV4_FLASH_PP_LAYER_PARTITION"
else
  unset VLLM_PP_LAYER_PARTITION
fi

KV_CACHE_MEMORY_ARGS=()
case "$DSV4_KV_CACHE_MEMORY_BYTES" in
  ""|0|auto|AUTO|none|NONE)
    ;;
  *)
    KV_CACHE_MEMORY_ARGS=(--kv-cache-memory-bytes "$DSV4_KV_CACHE_MEMORY_BYTES")
    ;;
esac

COMMON_ARGS=(
  -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name deepseek-v4-flash-pp${NNODES}
  --tensor-parallel-size 1
  --pipeline-parallel-size "$NNODES"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --max-model-len "${DSV4_MAX_MODEL_LEN:-65536}"
  --max-num-seqs "${DSV4_MAX_NUM_SEQS:-8}"
  --max-num-batched-tokens "${DSV4_MAX_NUM_BATCHED_TOKENS:-16384}"
  --gpu-memory-utilization "${DSV4_GPU_MEMORY_UTILIZATION:-0.82}"
  "${KV_CACHE_MEMORY_ARGS[@]}"
  "${FLASHINFER_AUTOTUNE_ARGS[@]}"
  --block-size 256
  --kv-cache-dtype fp8
  --enable-prefix-caching
  --kv-offloading-size "${DSV4_KV_OFFLOADING_SIZE:-8}"
  --kv-offloading-backend native
  --kv-cache-metrics
  --enable-logging-iteration-details
  "${SPECULATIVE_ARGS[@]}"
  --compilation-config "$DSV4_COMPILATION_CONFIG"
  --tokenizer-mode deepseek_v4
  --load-format safetensors
  --no-disable-hybrid-kv-cache-manager
)

if [[ "$DSV4_LINEAR_BACKEND" != "auto" ]]; then
  COMMON_ARGS+=(--linear-backend "$DSV4_LINEAR_BACKEND")
fi

if [[ "$DSV4_MOE_BACKEND" != "auto" ]]; then
  COMMON_ARGS+=(--moe-backend "$DSV4_MOE_BACKEND")
fi

if [[ "$NODE_RANK" == "0" ]]; then
  exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "$API_PORT"
fi

exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --headless
