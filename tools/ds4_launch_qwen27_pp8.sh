#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to the local pipeline rank}"
: "${HEAD_ADDR:?set HEAD_ADDR to the rank-0 Spark private IP or hostname}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ds4_200g_guard.sh"

if [[ "${QWEN27_ENABLE_FLASHINFER_AUTOTUNE:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "QWEN27_ENABLE_FLASHINFER_AUTOTUNE is deprecated; use DS4_ENABLE_FLASHINFER_AUTOTUNE only from a dedicated tuning job" >&2
  exit 64
fi
ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE

NNODES="${NNODES:-8}"
MASTER_PORT="${MASTER_PORT:-29527}"
API_PORT="${API_PORT:-8101}"
MODEL="${QWEN27_BF16_MODEL:-/home/$USER/models/hf/Qwen/Qwen3.6-27B}"
RUNTIME_PYTHON="${DS4_VLLM_PYTHON:-/home/$USER/ds4-vllm-local/bin/python}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-/home/$USER/src/vllm}"
DS4_NODE_ID="${DS4_NODE_ID:-spark${NODE_RANK}}"
DS4_QWEN_PIPELINE_RAM_PROFILE="${DS4_QWEN_PIPELINE_RAM_PROFILE:-resident3}"
DEFAULT_LMCACHE_ROOT="$HOME/ds4_lmcache/qwen27_bf16_pp${NNODES}/${DS4_NODE_ID}"

if [[ -z "${QWEN27_PP_LAYER_PARTITION:-}" ]]; then
  if [[ "$NNODES" == "8" ]]; then
    QWEN27_PP_LAYER_PARTITION="9,9,9,8,8,8,8,5"
  else
    QWEN27_PP_LAYER_PARTITION="$($RUNTIME_PYTHON - "$NNODES" <<'PY'
import sys
layers = 64
stages = int(sys.argv[1])
if stages < 1 or stages > layers:
    raise SystemExit(f"invalid Qwen pipeline stage count {stages} for {layers} layers")
base, extra = divmod(layers, stages)
print(",".join(str(base + (1 if index < extra else 0)) for index in range(stages)))
PY
)"
  fi
fi

"$RUNTIME_PYTHON" - "$NNODES" "$QWEN27_PP_LAYER_PARTITION" <<'PY'
import sys
stages = int(sys.argv[1])
raw = sys.argv[2]
parts = [int(item) for item in raw.split(",") if item.strip()]
if len(parts) != stages:
    raise SystemExit(f"QWEN27_PP_LAYER_PARTITION has {len(parts)} stages but NNODES={stages}: {raw}")
if sum(parts) != 64:
    raise SystemExit(f"QWEN27_PP_LAYER_PARTITION must sum to 64 Qwen decoder layers, got {sum(parts)}: {raw}")
if any(part <= 0 for part in parts):
    raise SystemExit(f"QWEN27_PP_LAYER_PARTITION stages must all be positive: {raw}")
PY

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_PP_LAYER_PARTITION="$QWEN27_PP_LAYER_PARTITION"
export VLLM_QWEN_GDN_PROFILE_WARMUP="${VLLM_QWEN_GDN_PROFILE_WARMUP:-0}"
export VLLM_DS4_PROFILE_LAYER_TRACE="${VLLM_DS4_PROFILE_LAYER_TRACE:-0}"
export VLLM_DS4_PROFILE_DEBUG="${VLLM_DS4_PROFILE_DEBUG:-1}"
export VLLM_DS4_PROFILE_WATCHDOG_SECONDS="${VLLM_DS4_PROFILE_WATCHDOG_SECONDS:-120}"
export VLLM_DS4_PROFILE_ABORT_SECONDS="${VLLM_DS4_PROFILE_ABORT_SECONDS:-600}"
export VLLM_DS4_PROFILE_RUN_MAX_TOKENS="${VLLM_DS4_PROFILE_RUN_MAX_TOKENS:-512}"
export VLLM_DS4_VALIDATE_INPUT_IDS="${VLLM_DS4_VALIDATE_INPUT_IDS:-1}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export DS4_200G_IFNAME="${DS4_200G_IFNAME:-enP2p1s0f0np0,enP2p1s0f1np1}"
export DS4_CONTROL_IFNAME="${DS4_CONTROL_IFNAME:-ds4ring0}"
export DS4_200G_ADVERTISE_LOOPBACK="${DS4_200G_ADVERTISE_LOOPBACK:-1}"
export DS4_200G_NCCL_TRANSPORT="${DS4_200G_NCCL_TRANSPORT:-socket}"
export VLLM_DS4_PP_ONLY_GLOBAL_BACKEND="${VLLM_DS4_PP_ONLY_GLOBAL_BACKEND:-nccl}"
export VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR="${VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR:-0}"
export VLLM_DS4_PP_PYNCCL_TENSOR_DICT="${VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-0}"
export VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT="${VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT:-0}"
export VLLM_DS4_PP_TORCH_PG_TENSOR_DICT="${VLLM_DS4_PP_TORCH_PG_TENSOR_DICT:-1}"
export VLLM_DS4_PP_DIRECT_CUDA_MIN_BYTES="${VLLM_DS4_PP_DIRECT_CUDA_MIN_BYTES:-262144}"
export VLLM_DS4_PP_DEVICE_TENSOR_DICT_METADATA="${VLLM_DS4_PP_DEVICE_TENSOR_DICT_METADATA:-1}"
export VLLM_DS4_PP_SEND_BACKLOG="${VLLM_DS4_PP_SEND_BACKLOG:-${QWEN27_PP_SEND_BACKLOG:-2}}"
export VLLM_DS4_PP_OVERLAP_SEND="${VLLM_DS4_PP_OVERLAP_SEND:-1}"
export VLLM_DS4_PP_SEND_BUFFER_SLOTS="${VLLM_DS4_PP_SEND_BUFFER_SLOTS:-4}"
export VLLM_DS4_PP_SEND_BUFFER_MAX_BYTES="${VLLM_DS4_PP_SEND_BUFFER_MAX_BYTES:-1073741824}"
export VLLM_DS4_PP_GANTT_TRACE="${VLLM_DS4_PP_GANTT_TRACE:-0}"
export VLLM_DS4_PP_GANTT_TRACE_EVERY="${VLLM_DS4_PP_GANTT_TRACE_EVERY:-10}"
export VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES="${VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES:-8}"
export VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPE_MIN_BYTES="${VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPE_MIN_BYTES:-1048576}"
export VLLM_DS4_PP_PYNCCL_P2P_CREDIT="${VLLM_DS4_PP_PYNCCL_P2P_CREDIT:-1}"
export VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT="${VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT:-0}"
export VLLM_DS4_PP_STRIPED_NCCL_STRIPES="${VLLM_DS4_PP_STRIPED_NCCL_STRIPES:-$VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES}"
export VLLM_DS4_PP_STRIPED_NCCL_MIN_BYTES="${VLLM_DS4_PP_STRIPED_NCCL_MIN_BYTES:-262144}"
export VLLM_DS4_PP_STRIPED_NCCL_STREAMS="${VLLM_DS4_PP_STRIPED_NCCL_STREAMS:-1}"
export VLLM_DS4_SKIP_PYNCCL_WARMUP_ALLREDUCE="${VLLM_DS4_SKIP_PYNCCL_WARMUP_ALLREDUCE:-0}"
export DS4_NCCL_PREFLIGHT_MODE="${DS4_NCCL_PREFLIGHT_MODE:-nccl}"
export DS4_RAIL_TCP_PREFLIGHT_ACTIVE="${DS4_RAIL_TCP_PREFLIGHT_ACTIVE:-1}"
export DS4_RAIL_TCP_PREFLIGHT_STREAMS="${DS4_RAIL_TCP_PREFLIGHT_STREAMS:-16}"
export DS4_RAIL_TCP_PREFLIGHT_BYTES="${DS4_RAIL_TCP_PREFLIGHT_BYTES:-268435456}"
export DS4_RAIL_TCP_PREFLIGHT_MIN_GBIT_S="${DS4_RAIL_TCP_PREFLIGHT_MIN_GBIT_S:-10}"
export DS4_RAIL_TCP_PREFLIGHT_WARN_GBIT_S="${DS4_RAIL_TCP_PREFLIGHT_WARN_GBIT_S:-64}"
if [[ -z "${DS4_RAIL_TCP_PREFLIGHT_PAIRS:-}" && "$NNODES" == "8" ]]; then
  export DS4_RAIL_TCP_PREFLIGHT_PAIRS="0-1;1-2;2-3;3-4;4-5;5-6;6-7"
fi
if [[ "$NODE_RANK" == "0" ]]; then
  export DS4_200G_ALLOW_LOOPBACK_HEAD="${DS4_200G_ALLOW_LOOPBACK_HEAD:-1}"
fi
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/tmp/lmcache_qwen27_bf16_pp${NNODES}_${DS4_NODE_ID}.yaml}"
export LMCACHE_ROOT="${LMCACHE_ROOT:-$DEFAULT_LMCACHE_ROOT}"
mkdir -p "$LMCACHE_ROOT"

case "$DS4_QWEN_PIPELINE_RAM_PROFILE" in
  resident3|compact|COMPACT)
    : "${QWEN27_MAX_MODEL_LEN:=262144}"
    : "${QWEN27_KV_CACHE_MEMORY_BYTES:=4294967296}"
    : "${LMCACHE_MAX_LOCAL_CPU_SIZE:=0.5}"
    : "${QWEN27_MAX_NUM_SEQS:=12}"
    : "${QWEN27_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8,12}"
    : "${QWEN27_MAX_CUDAGRAPH_CAPTURE_SIZE:=12}"
    ;;
  balanced|BALANCED)
    : "${QWEN27_KV_CACHE_MEMORY_BYTES:=8589934592}"
    : "${LMCACHE_MAX_LOCAL_CPU_SIZE:=2.0}"
    ;;
  *)
    echo "Unsupported DS4_QWEN_PIPELINE_RAM_PROFILE=$DS4_QWEN_PIPELINE_RAM_PROFILE; expected resident3, compact, or balanced" >&2
    exit 1
    ;;
esac

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TRITON_CACHE_MAX_SIZE="${TRITON_CACHE_MAX_SIZE:-2147483648}"

QWEN27_KV_CACHE_DTYPE="${QWEN27_KV_CACHE_DTYPE:-fp8}"
QWEN27_ATTENTION_BACKEND="${QWEN27_ATTENTION_BACKEND:-TRITON_ATTN}"
case "$QWEN27_ATTENTION_BACKEND" in
  TRITON_ATTN)
    ;;
  FLASH_ATTN)
    case "$QWEN27_KV_CACHE_DTYPE" in
      fp8*)
        echo "Qwen BF16 PP cannot use FLASH_ATTN with QWEN27_KV_CACHE_DTYPE=$QWEN27_KV_CACHE_DTYPE." >&2
        echo "Reason: FlashAttention rejects fp8 KV cache in this vLLM path; use TRITON_ATTN or switch KV to auto/bfloat16." >&2
        exit 2
        ;;
    esac
    ;;
  FLASHINFER)
    echo "Qwen BF16 PP does not default to FLASHINFER attention on GB10." >&2
    echo "Reason: FlashInfer XQA failed dummy-run capture with a query/output dtype mismatch." >&2
    echo "Use QWEN27_ALLOW_FLASHINFER_ATTENTION_EXPERIMENTAL=1 only for a targeted bring-up run." >&2
    if [[ ! "${QWEN27_ALLOW_FLASHINFER_ATTENTION_EXPERIMENTAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      exit 2
    fi
    ;;
  *)
    echo "Unsupported QWEN27_ATTENTION_BACKEND=$QWEN27_ATTENTION_BACKEND; expected FLASH_ATTN, TRITON_ATTN, or guarded FLASHINFER" >&2
    exit 2
    ;;
esac

ds4_prepare_triton_jit_environment "qwen27-bf16-pp${NNODES}"
ds4_prepare_flashinfer_jit_environment
export VLLM_MQ_MAX_CHUNKS="${VLLM_MQ_MAX_CHUNKS:-64}"
ds4_require_200g_fabric
ds4_run_rail_tcp_preflight "$NNODES"
ds4_run_nccl_preflight "$NNODES"
if [[ "${DS4_QWEN_TRITON_JIT_PREFLIGHT:-1}" == "1" ]]; then
  ds4_run_triton_jit_preflight
fi

cat > "$LMCACHE_CONFIG_FILE" <<YAML
chunk_size: ${LMCACHE_CHUNK_SIZE:-784}
local_cpu: true
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
local_disk: file://$LMCACHE_ROOT
max_local_disk_size: ${LMCACHE_MAX_LOCAL_DISK_SIZE:-2048.0}
use_gpu_connector_v3: ${LMCACHE_USE_GPU_CONNECTOR_V3:-true}
pre_caching_hash_algorithm: ${LMCACHE_PRE_CACHING_HASH_ALGORITHM:-sha256_cbor}
lookup_server_worker_ids: [0]
YAML

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache_kv_cache_group_id":"auto","discard_partial_chunks":false}}'
KV_TRANSFER_ARGS=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
case "${QWEN27_ENABLE_LMCACHE:-1}" in
  0|false|False|no|NO|off|OFF)
    echo "Qwen BF16 PP LMCache/HMA disabled for this run (QWEN27_ENABLE_LMCACHE=0)." >&2
    KV_TRANSFER_ARGS=()
    ;;
esac
KV_CACHE_MEMORY_ARGS=()
case "$QWEN27_KV_CACHE_MEMORY_BYTES" in
  ""|0|auto|AUTO|none|NONE)
    ;;
  *)
    KV_CACHE_MEMORY_ARGS=(--kv-cache-memory-bytes "$QWEN27_KV_CACHE_MEMORY_BYTES")
    ;;
esac
EAGER_ARGS=()
if [[ "${QWEN27_ENFORCE_EAGER:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  EAGER_ARGS=(--enforce-eager)
fi
COMPILATION_ARGS=()
if [[ -n "${QWEN27_COMPILATION_CONFIG:-}" ]]; then
  case "$QWEN27_COMPILATION_CONFIG" in
    none|NONE|auto|AUTO)
      ;;
    *)
      COMPILATION_ARGS=(--compilation-config "$QWEN27_COMPILATION_CONFIG")
      ;;
  esac
elif [[ -n "${QWEN27_CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
  COMPILATION_ARGS=(--compilation-config "{\"cudagraph_capture_sizes\":[$QWEN27_CUDAGRAPH_CAPTURE_SIZES],\"max_cudagraph_capture_size\":${QWEN27_MAX_CUDAGRAPH_CAPTURE_SIZE:-8}}")
fi

ASYNC_SCHEDULING_ARGS=(--no-async-scheduling)
if [[ "${QWEN27_ENABLE_ASYNC_SCHEDULING_EXPERIMENTAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "Qwen PP async scheduling is disabled until the sync PP pipeline is stable; unset QWEN27_ENABLE_ASYNC_SCHEDULING_EXPERIMENTAL." >&2
  exit 64
fi

HYBRID_KV_CACHE_MANAGER_ARGS=(--no-disable-hybrid-kv-cache-manager)
case "${QWEN27_DISABLE_HYBRID_KV_CACHE_MANAGER:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    HYBRID_KV_CACHE_MANAGER_ARGS=(--disable-hybrid-kv-cache-manager)
    ;;
esac

REASONING_PARSER_ARGS=()
if [[ "${QWEN27_REASONING_PARSER:-none}" != "none" ]]; then
  REASONING_PARSER_ARGS=(--reasoning-parser "$QWEN27_REASONING_PARSER")
fi

COMMON_ARGS=(
  -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name "${QWEN27_SERVED_MODEL_NAME:-qwen27-bf16-pp${NNODES}}"
  --trust-remote-code
  --tensor-parallel-size 1
  --pipeline-parallel-size "$NNODES"
  --distributed-executor-backend mp
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --max-model-len "${QWEN27_MAX_MODEL_LEN:-65536}"
  --max-num-seqs "${QWEN27_MAX_NUM_SEQS:-8}"
  --max-num-batched-tokens "${QWEN27_MAX_NUM_BATCHED_TOKENS:-8192}"
  --gpu-memory-utilization "${QWEN27_GPU_MEMORY_UTILIZATION:-0.24}"
  "${KV_CACHE_MEMORY_ARGS[@]}"
  "${EAGER_ARGS[@]}"
  "${COMPILATION_ARGS[@]}"
  --dtype bfloat16
  --kv-cache-dtype "$QWEN27_KV_CACHE_DTYPE"
  --attention-backend "$QWEN27_ATTENTION_BACKEND"
  --gdn-prefill-backend "${QWEN27_GDN_PREFILL_BACKEND:-triton}"
  --language-model-only
  --enable-chunked-prefill
  --enable-prefix-caching
  "${ASYNC_SCHEDULING_ARGS[@]}"
  "${REASONING_PARSER_ARGS[@]}"
  "${HYBRID_KV_CACHE_MANAGER_ARGS[@]}"
  --mamba-cache-mode align
  "${KV_TRANSFER_ARGS[@]}"
  "${FLASHINFER_AUTOTUNE_ARGS[@]}"
)

if [[ "$NODE_RANK" == "0" ]]; then
  exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "$API_PORT"
fi

exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --headless
