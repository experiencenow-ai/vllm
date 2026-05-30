#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0..N-1 on each Spark}"
: "${HEAD_ADDR:?set HEAD_ADDR to the rank-0 Spark private IP or hostname}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ds4_200g_guard.sh"

if [[ "${QWEN27_ENABLE_FLASHINFER_AUTOTUNE:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "QWEN27_ENABLE_FLASHINFER_AUTOTUNE is deprecated; use DS4_ENABLE_FLASHINFER_AUTOTUNE only from a dedicated tuning job" >&2
  exit 64
fi
ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE

NNODES="${NNODES:-8}"
PP_SIZE="${QWEN27_PP_SIZE:-$NNODES}"
MASTER_PORT="${MASTER_PORT:-29537}"
API_PORT="${API_PORT:-8103}"
MODEL="${QWEN27_NVFP4_MODEL:-/home/$USER/models/hf/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP}"
RUNTIME_PYTHON="${DS4_VLLM_PYTHON:-/home/$USER/ds4-vllm-local/bin/python}"
SOURCE_ROOT="${DS4_VLLM_SOURCE_ROOT:-/home/$USER/src/vllm}"
DS4_NODE_ID="${DS4_NODE_ID:-spark${NODE_RANK}}"
DEFAULT_LMCACHE_ROOT="$HOME/ds4_lmcache/qwen27_nvfp4_pp${PP_SIZE}/${DS4_NODE_ID}"

if [[ "$PP_SIZE" != "$NNODES" ]]; then
  echo "Qwen PP launcher expects one PP rank per Spark: PP_SIZE=$PP_SIZE NNODES=$NNODES" >&2
  exit 2
fi

if [[ -z "${QWEN27_PP_LAYER_PARTITION:-}" ]]; then
  if [[ "$PP_SIZE" == "8" ]]; then
    QWEN27_PP_LAYER_PARTITION="9,9,9,8,8,8,8,5"
  else
    QWEN27_PP_LAYER_PARTITION="$($RUNTIME_PYTHON - "$PP_SIZE" <<'PY'
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

if [[ "${QWEN27_NVFP4_ENABLE_MTP:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "Qwen NVFP4 PP is cache-primary and currently forbids MTP by default." >&2
  echo "Reason: Qwen3.6 MTP draft/cache semantics under PP need a separate correctness pass." >&2
  echo "Use QWEN27_NVFP4_ENABLE_MTP_EXPERIMENTAL=1 only for a targeted bring-up run." >&2
  if [[ ! "${QWEN27_NVFP4_ENABLE_MTP_EXPERIMENTAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    exit 2
  fi
fi

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_PP_LAYER_PARTITION="$QWEN27_PP_LAYER_PARTITION"
export VLLM_DS4_STRICT_NATIVE_FP4="${VLLM_DS4_STRICT_NATIVE_FP4:-1}"
export VLLM_MXFP4_USE_MARLIN=0
export VLLM_TEST_FORCE_FP8_MARLIN=0
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
export LMCACHE_ROOT="${LMCACHE_ROOT:-$DEFAULT_LMCACHE_ROOT}"
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/tmp/lmcache_qwen27_nvfp4_pp${PP_SIZE}_${DS4_NODE_ID}.yaml}"
mkdir -p "$LMCACHE_ROOT"

QWEN27_KV_CACHE_DTYPE="${QWEN27_KV_CACHE_DTYPE:-fp8}"
QWEN27_ATTENTION_BACKEND="${QWEN27_ATTENTION_BACKEND:-TRITON_ATTN}"
case "$QWEN27_ATTENTION_BACKEND" in
  TRITON_ATTN)
    ;;
  FLASH_ATTN)
    case "$QWEN27_KV_CACHE_DTYPE" in
      fp8*)
        echo "Qwen NVFP4 PP cannot use FLASH_ATTN with QWEN27_KV_CACHE_DTYPE=$QWEN27_KV_CACHE_DTYPE." >&2
        echo "Reason: FlashAttention rejects fp8 KV cache in this vLLM path; use TRITON_ATTN or switch KV to auto/bfloat16." >&2
        exit 2
        ;;
    esac
    ;;
  FLASHINFER)
    echo "Qwen NVFP4 PP does not default to FLASHINFER attention on GB10." >&2
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

ds4_prepare_triton_jit_environment "qwen27-nvfp4-pp${PP_SIZE}"
ds4_prepare_flashinfer_jit_environment
ds4_require_200g_fabric
ds4_run_nccl_preflight "$NNODES"
if [[ "${DS4_QWEN_TRITON_JIT_PREFLIGHT:-1}" == "1" ]]; then
  ds4_run_triton_jit_preflight
fi

cat > "$LMCACHE_CONFIG_FILE" <<YAML
chunk_size: ${LMCACHE_CHUNK_SIZE:-784}
local_cpu: true
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE:-4.0}
local_disk: file://$LMCACHE_ROOT
max_local_disk_size: ${LMCACHE_MAX_LOCAL_DISK_SIZE:-2048.0}
YAML

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache_kv_cache_group_id":"auto","discard_partial_chunks":false}}'

SPEC_ARGS=()
if [[ "${QWEN27_NVFP4_ENABLE_MTP_EXPERIMENTAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  SPEC_ARGS=(--speculative-config "${QWEN27_SPECULATIVE_CONFIG:-{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}}")
fi

COMMON_ARGS=(
  -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name "${QWEN27_SERVED_MODEL_NAME:-qwen27-nvfp4-pp${PP_SIZE}}"
  --trust-remote-code
  --distributed-executor-backend mp
  --tensor-parallel-size 1
  --pipeline-parallel-size "$PP_SIZE"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_ADDR"
  --master-port "$MASTER_PORT"
  --max-model-len "${QWEN27_MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${QWEN27_MAX_NUM_SEQS:-24}"
  --max-num-batched-tokens "${QWEN27_MAX_NUM_BATCHED_TOKENS:-65536}"
  --gpu-memory-utilization "${QWEN27_GPU_MEMORY_UTILIZATION:-0.40}"
  --quantization modelopt
  --linear-backend "${QWEN27_LINEAR_BACKEND:-flashinfer-cutlass}"
  --attention-backend "$QWEN27_ATTENTION_BACKEND"
  --kv-cache-dtype "$QWEN27_KV_CACHE_DTYPE"
  --language-model-only
  --enable-chunked-prefill
  --enable-prefix-caching
  --async-scheduling
  --reasoning-parser qwen3
  --no-disable-hybrid-kv-cache-manager
  --mamba-cache-mode align
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
  "${FLASHINFER_AUTOTUNE_ARGS[@]}"
  "${SPEC_ARGS[@]}"
)

if [[ "$NODE_RANK" == "0" ]]; then
  exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --host "${API_HOST:-0.0.0.0}" --port "$API_PORT"
fi

exec "$RUNTIME_PYTHON" "${COMMON_ARGS[@]}" --headless
