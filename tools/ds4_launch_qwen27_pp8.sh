#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0..7 on each Spark}"
: "${HEAD_ADDR:?set HEAD_ADDR to the rank-0 Spark private IP or hostname}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ds4_200g_guard.sh"

NNODES="${NNODES:-8}"
MASTER_PORT="${MASTER_PORT:-29527}"
API_PORT="${API_PORT:-8101}"
MODEL="${QWEN27_BF16_MODEL:-/home/$USER/models/hf/Qwen/Qwen3.6-27B}"
RUNTIME_PYTHON="${DS4_VLLM_PYTHON:-/home/$USER/ds4-vllm-local/bin/python}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-/home/$USER/src/vllm}"

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_PP_LAYER_PARTITION="${QWEN27_PP_LAYER_PARTITION:-9,9,9,8,8,8,8,5}"
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/tmp/lmcache_qwen27_pp8.yaml}"
export LMCACHE_ROOT="${LMCACHE_ROOT:-/mnt/nvme/ds4_lmcache/qwen27_pp8}"
mkdir -p "$LMCACHE_ROOT"

ds4_prepare_triton_jit_environment "qwen27-bf16-pp${NNODES}"
if [[ "${DS4_QWEN_TRITON_JIT_PREFLIGHT:-1}" == "1" ]]; then
  ds4_run_triton_jit_preflight
fi

cat > "$LMCACHE_CONFIG_FILE" <<YAML
chunk_size: ${LMCACHE_CHUNK_SIZE:-784}
local_cpu: true
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE:-16.0}
local_disk: file://$LMCACHE_ROOT
max_local_disk_size: ${LMCACHE_MAX_LOCAL_DISK_SIZE:-2048.0}
YAML

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache_kv_cache_group_id":"auto","discard_partial_chunks":false}}'

ASYNC_SCHEDULING_ARGS=(--async-scheduling)
case "${QWEN27_ASYNC_SCHEDULING:-1}" in
  0|false|False|no|NO|off|OFF)
    ASYNC_SCHEDULING_ARGS=(--no-async-scheduling)
    ;;
esac

COMMON_ARGS=(
  -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name qwen27-bf16-pp8
  --trust-remote-code
  --tensor-parallel-size 1
  --pipeline-parallel-size 8
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --max-model-len "${QWEN27_MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${QWEN27_MAX_NUM_SEQS:-12}"
  --max-num-batched-tokens "${QWEN27_MAX_NUM_BATCHED_TOKENS:-32768}"
  --gpu-memory-utilization "${QWEN27_GPU_MEMORY_UTILIZATION:-0.50}"
  --dtype bfloat16
  --language-model-only
  --enable-chunked-prefill
  --enable-prefix-caching
  "${ASYNC_SCHEDULING_ARGS[@]}"
  --reasoning-parser qwen3
  --no-disable-hybrid-kv-cache-manager
  --mamba-cache-mode align
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
)

if [[ "$NODE_RANK" == "0" ]]; then
  exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "$API_PORT"
fi

exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --headless
