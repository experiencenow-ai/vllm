#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0 on head or 1 on worker}"
: "${HEAD_ADDR:?set HEAD_ADDR to the rank-0 Spark private IP or hostname}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ds4_200g_guard.sh"

MASTER_PORT="${MASTER_PORT:-29501}"
API_PORT="${API_PORT:-8000}"
MODEL="${DSV4_FLASH_MODEL:-/home/$USER/models/hf/deepseek-ai/DeepSeek-V4-Flash}"
RUNTIME_PYTHON="${DS4_VLLM_PYTHON:-/home/$USER/ds4-vllm-local/bin/python}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-/home/$USER/src/vllm}"
DEFAULT_SPECULATIVE_CONFIG="{\"model\":\"$MODEL\",\"num_speculative_tokens\":2,\"method\":\"deepseek_mtp\"}"

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_TRITON_MLA_SPARSE="${VLLM_TRITON_MLA_SPARSE:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"
export VLLM_DS4_STRICT_NATIVE_FP4="${VLLM_DS4_STRICT_NATIVE_FP4:-1}"
if [[ "${VLLM_MXFP4_USE_MARLIN:-}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  echo "DS4 strict native mode refuses VLLM_MXFP4_USE_MARLIN=$VLLM_MXFP4_USE_MARLIN" >&2
  exit 64
fi
unset VLLM_MXFP4_USE_MARLIN
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
ds4_require_200g_fabric
ds4_run_nccl_preflight 2
ds4_run_native_blackwell_preflight

COMMON_ARGS=(
  -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name deepseek-v4-flash-tp2-native
  --trust-remote-code
  --tensor-parallel-size 2
  --enable-expert-parallel
  --nnodes 2
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --kv-cache-dtype fp8
  --block-size 256
  --enable-prefix-caching
  --max-model-len "${DSV4_MAX_MODEL_LEN:-200000}"
  --max-num-seqs "${DSV4_MAX_NUM_SEQS:-2}"
  --max-num-batched-tokens "${DSV4_MAX_NUM_BATCHED_TOKENS:-4096}"
  --gpu-memory-utilization "${DSV4_GPU_MEMORY_UTILIZATION:-0.85}"
  --speculative-config "${DSV4_SPECULATIVE_CONFIG:-$DEFAULT_SPECULATIVE_CONFIG}"
  --compilation-config "${DSV4_COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}}"
  --tokenizer-mode deepseek_v4
  --load-format safetensors
)

if [[ "$NODE_RANK" == "0" ]]; then
  exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "$API_PORT"
fi

exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --headless
