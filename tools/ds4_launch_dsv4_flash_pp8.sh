#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0..7 on each Spark}"
: "${HEAD_ADDR:?set HEAD_ADDR to the rank-0 Spark private IP or hostname}"

NNODES="${NNODES:-8}"
MASTER_PORT="${MASTER_PORT:-29544}"
API_PORT="${API_PORT:-8102}"
MODEL="${DSV4_FLASH_MODEL:-/home/$USER/models/hf/deepseek-ai/DeepSeek-V4-Flash}"
RUNTIME_PYTHON="${DS4_VLLM_PYTHON:-/home/$USER/ds4-vllm-local/bin/python}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-/home/$USER/src/vllm}"

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"
export VLLM_USE_SIMPLE_KV_OFFLOAD="${VLLM_USE_SIMPLE_KV_OFFLOAD:-1}"
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT:-/mnt/nvme/ds4_hma_store/dsv4_flash_pp8/simple_cpu_offload}"
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT:-1}"
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK:-$(hostname)-dsv4-pp8-r${NODE_RANK}}"
mkdir -p "$VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT"

if [[ -n "${DSV4_FLASH_PP_LAYER_PARTITION:-}" ]]; then
  export VLLM_PP_LAYER_PARTITION="$DSV4_FLASH_PP_LAYER_PARTITION"
else
  unset VLLM_PP_LAYER_PARTITION
fi

COMMON_ARGS=(
  -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name deepseek-v4-flash-pp8
  --tensor-parallel-size 1
  --pipeline-parallel-size 8
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --max-model-len "${DSV4_MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${DSV4_MAX_NUM_SEQS:-12}"
  --max-num-batched-tokens "${DSV4_MAX_NUM_BATCHED_TOKENS:-32768}"
  --gpu-memory-utilization "${DSV4_GPU_MEMORY_UTILIZATION:-0.40}"
  --block-size 256
  --kv-cache-dtype fp8
  --enable-prefix-caching
  --kv-offloading-size "${DSV4_KV_OFFLOADING_SIZE:-8}"
  --kv-offloading-backend native
  --kv-cache-metrics
  --enable-logging-iteration-details
  --speculative-config "${DSV4_SPECULATIVE_CONFIG:-{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":2}}"
  --no-disable-hybrid-kv-cache-manager
  --enforce-eager
)

if [[ "$NODE_RANK" == "0" ]]; then
  exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "$API_PORT"
fi

exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --headless
