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
DS4_DSV4_PIPELINE_RAM_PROFILE="${DS4_DSV4_PIPELINE_RAM_PROFILE:-resident3}"
DEFAULT_SPECULATIVE_CONFIG="{\"model\":\"$MODEL\",\"num_speculative_tokens\":2,\"method\":\"deepseek_mtp\"}"
DSV4_LINEAR_BACKEND="${DSV4_LINEAR_BACKEND:-auto}"
DSV4_MOE_BACKEND="${DSV4_MOE_BACKEND:-auto}"
DSV4_MTP_MODE="${DSV4_MTP_MODE:-off}"
case "$DS4_DSV4_PIPELINE_RAM_PROFILE" in
  resident3|compact|COMPACT)
    : "${DSV4_MAX_MODEL_LEN:=65536}"
    : "${DSV4_MAX_NUM_SEQS:=8}"
    : "${DSV4_MAX_NUM_BATCHED_TOKENS:=4096}"
    : "${DSV4_KV_CACHE_MEMORY_BYTES:=4294967296}"
    : "${DSV4_KV_OFFLOADING_SIZE:=2}"
    : "${DSV4_GPU_MEMORY_UTILIZATION:=0.20}"
    : "${DSV4_WORKSPACE_PREALLOC_BYTES:=268435456}"
    : "${DSV4_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8}"
    : "${DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE:=8}"
    if [[ "$NNODES" == "8" && -z "${DSV4_FLASH_PP_LAYER_PARTITION:-}" ]]; then
      DSV4_FLASH_PP_LAYER_PARTITION="3,4,5,6,7,7,6,5"
    fi
    ;;
  tight|TIGHT|resident3-tight)
    : "${DSV4_MAX_MODEL_LEN:=32768}"
    : "${DSV4_MAX_NUM_SEQS:=4}"
    : "${DSV4_MAX_NUM_BATCHED_TOKENS:=4096}"
    : "${DSV4_KV_CACHE_MEMORY_BYTES:=3221225472}"
    : "${DSV4_KV_OFFLOADING_SIZE:=1}"
    : "${DSV4_GPU_MEMORY_UTILIZATION:=0.18}"
    : "${DSV4_WORKSPACE_PREALLOC_BYTES:=134217728}"
    : "${DSV4_CUDAGRAPH_CAPTURE_SIZES:=1,2,4}"
    : "${DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE:=4}"
    if [[ "$NNODES" == "8" && -z "${DSV4_FLASH_PP_LAYER_PARTITION:-}" ]]; then
      DSV4_FLASH_PP_LAYER_PARTITION="3,4,5,6,7,7,6,5"
    fi
    ;;
  balanced|BALANCED|perf|PERF|performance|PERFORMANCE)
    : "${DSV4_MAX_MODEL_LEN:=65536}"
    : "${DSV4_MAX_NUM_SEQS:=8}"
    : "${DSV4_MAX_NUM_BATCHED_TOKENS:=8192}"
    : "${DSV4_KV_CACHE_MEMORY_BYTES:=12884901888}"
    : "${DSV4_KV_OFFLOADING_SIZE:=8}"
    : "${DSV4_GPU_MEMORY_UTILIZATION:=0.30}"
    : "${DSV4_WORKSPACE_PREALLOC_BYTES:=536870912}"
    : "${DSV4_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8}"
    : "${DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE:=8}"
    ;;
  throughput|THROUGHPUT|api-throughput|API-THROUGHPUT)
    : "${DSV4_MAX_MODEL_LEN:=65536}"
    : "${DSV4_MAX_NUM_SEQS:=128}"
    : "${DSV4_MAX_NUM_BATCHED_TOKENS:=32768}"
    : "${DSV4_KV_CACHE_MEMORY_BYTES:=34359738368}"
    : "${DSV4_KV_OFFLOADING_SIZE:=8}"
    : "${DSV4_GPU_MEMORY_UTILIZATION:=0.60}"
    : "${DSV4_WORKSPACE_PREALLOC_BYTES:=805306368}"
    : "${DSV4_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8,16,32,64,128}"
    : "${DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE:=128}"
    ;;
  max-throughput|MAX-THROUGHPUT|batch512|BATCH512)
    : "${DSV4_MAX_MODEL_LEN:=65536}"
    : "${DSV4_MAX_NUM_SEQS:=512}"
    : "${DSV4_MAX_NUM_BATCHED_TOKENS:=65536}"
    : "${DSV4_KV_CACHE_MEMORY_BYTES:=51539607552}"
    : "${DSV4_KV_OFFLOADING_SIZE:=8}"
    : "${DSV4_GPU_MEMORY_UTILIZATION:=0.70}"
    : "${DSV4_WORKSPACE_PREALLOC_BYTES:=1073741824}"
    : "${DSV4_CUDAGRAPH_CAPTURE_SIZES:=1,2,4,8,16,32,64,128,256,512}"
    : "${DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE:=512}"
    ;;
  *)
    echo "Unsupported DS4_DSV4_PIPELINE_RAM_PROFILE=$DS4_DSV4_PIPELINE_RAM_PROFILE; expected resident3, tight, balanced, perf, throughput, or max-throughput" >&2
    exit 1
    ;;
esac
DEFAULT_COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[$DSV4_CUDAGRAPH_CAPTURE_SIZES],\"max_cudagraph_capture_size\":$DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE}"
DSV4_COMPILATION_CONFIG="${DSV4_COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"
ASYNC_SCHEDULING_ARGS=(--no-async-scheduling)
if [[ "${DSV4_ENABLE_ASYNC_SCHEDULING_EXPERIMENTAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  ASYNC_SCHEDULING_ARGS=(--async-scheduling)
fi
SPECULATIVE_ARGS=()
DSV4_MTP_REQUESTED=0
if [[ "$DSV4_MTP_MODE" != "off" && "$DSV4_MTP_MODE" != "OFF" && "$DSV4_MTP_MODE" != "none" && "$DSV4_MTP_MODE" != "NONE" ]]; then
  DSV4_MTP_REQUESTED=1
fi
if [[ "${DSV4_ENABLE_MTP:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  DSV4_MTP_REQUESTED=1
fi
if [[ "${DSV4_DISABLE_MTP:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  DSV4_MTP_REQUESTED=0
fi
if [[ "$DSV4_MTP_REQUESTED" == "1" ]]; then
  case "$DSV4_MTP_MODE" in
    chat_single|single_chat|latency_chat)
      ;;
    *)
      echo "DSV4 MTP is reserved for single-request chat latency mode." >&2
      echo "Set DSV4_MTP_MODE=chat_single and DSV4_MAX_NUM_SEQS=1 to enable it." >&2
      exit 2
      ;;
  esac
  if [[ "$DSV4_MAX_NUM_SEQS" != "1" ]]; then
    echo "DSV4 MTP requires DSV4_MAX_NUM_SEQS=1; got $DSV4_MAX_NUM_SEQS." >&2
    echo "Batch/throughput PP profiles must keep MTP off." >&2
    exit 2
  fi
  SPECULATIVE_ARGS=(--speculative-config "${DSV4_SPECULATIVE_CONFIG:-$DEFAULT_SPECULATIVE_CONFIG}")
fi
ds4_set_flashinfer_autotune_args DS4_ENABLE_FLASHINFER_AUTOTUNE

export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$RUNTIME_PYTHON"):$PATH"

export VLLM_DS4_PROFILE_DEBUG="${VLLM_DS4_PROFILE_DEBUG:-0}"
export VLLM_DS4_PROFILE_WATCHDOG_SECONDS="${VLLM_DS4_PROFILE_WATCHDOG_SECONDS:-120}"
export VLLM_DS4_PROFILE_ABORT_SECONDS="${VLLM_DS4_PROFILE_ABORT_SECONDS:-600}"
export VLLM_DS4_PROFILE_RUN_MAX_TOKENS="${VLLM_DS4_PROFILE_RUN_MAX_TOKENS:-512}"
export VLLM_WORKSPACE_PREALLOC_BYTES="${VLLM_WORKSPACE_PREALLOC_BYTES:-$DSV4_WORKSPACE_PREALLOC_BYTES}"
export VLLM_DS4_COHORT_ADMISSION="${VLLM_DS4_COHORT_ADMISSION:-1}"
export VLLM_DS4_COHORT_ADMISSION_MIN_PROMPTS="${VLLM_DS4_COHORT_ADMISSION_MIN_PROMPTS:-2}"
export VLLM_DS4_COHORT_PAUSE_DURING_ADMISSION="${VLLM_DS4_COHORT_PAUSE_DURING_ADMISSION:-1}"
export VLLM_DS4_FINAL_ONLY_NONSTREAMING="${VLLM_DS4_FINAL_ONLY_NONSTREAMING:-1}"
export VLLM_DS4_SCHED_MAX_NEW_REQS_PER_STEP="${VLLM_DS4_SCHED_MAX_NEW_REQS_PER_STEP:-${DSV4_SCHED_MAX_NEW_REQS_PER_STEP:-32}}"
export VLLM_DS4_FUSED_EXECUTE_SAMPLE="${VLLM_DS4_FUSED_EXECUTE_SAMPLE:-${DSV4_FUSED_EXECUTE_SAMPLE:-1}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_TRITON_MLA_SPARSE="${VLLM_TRITON_MLA_SPARSE:-1}"
export VLLM_DS4_SM12X_MQA_ROWWISE="${VLLM_DS4_SM12X_MQA_ROWWISE:-1}"
export VLLM_DS4_SM12X_MQA_ROWWISE_MAX_ROWS="${VLLM_DS4_SM12X_MQA_ROWWISE_MAX_ROWS:-4}"
export VLLM_DS4_SM12X_MQA_ROWWISE_MIN_TOKENS="${VLLM_DS4_SM12X_MQA_ROWWISE_MIN_TOKENS:-0}"
export VLLM_DS4_SM12X_PAGED_MQA_TOPK_CHUNK_SIZE="${VLLM_DS4_SM12X_PAGED_MQA_TOPK_CHUNK_SIZE:-8192}"
export VLLM_DS4_SM12X_MQA_TOPK_TRITON="${VLLM_DS4_SM12X_MQA_TOPK_TRITON:-1}"
export VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT="${VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT:-1}"
export VLLM_DS4_SM12X_MQA_TOPK_CHUNK_SIZE="${VLLM_DS4_SM12X_MQA_TOPK_CHUNK_SIZE:-8192}"
export VLLM_DS4_SM12X_MQA_TOPK_MAX_LOGITS_BYTES="${VLLM_DS4_SM12X_MQA_TOPK_MAX_LOGITS_BYTES:-536870912}"
export VLLM_DS4_ALLOW_SM12X_MQA_TOPK_TORCH_FALLBACK="${VLLM_DS4_ALLOW_SM12X_MQA_TOPK_TORCH_FALLBACK:-0}"
export VLLM_DS4_DSV4_K_GATHER_BACKEND="${VLLM_DS4_DSV4_K_GATHER_BACKEND:-cutedsl}"
export VLLM_DS4_DSV4_ALLOW_TRITON_GATHER_DEBUG="${VLLM_DS4_DSV4_ALLOW_TRITON_GATHER_DEBUG:-0}"
export VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL_MAX_ROWS="${VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL_MAX_ROWS:--1}"
export VLLM_MQ_MAX_CHUNKS="${VLLM_MQ_MAX_CHUNKS:-64}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
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
export VLLM_DS4_PP_ONLY_GLOBAL_BACKEND="${VLLM_DS4_PP_ONLY_GLOBAL_BACKEND:-nccl}"
export VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR="${VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR:-1}"
export VLLM_DS4_PP_PYNCCL_TENSOR_DICT="${VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-0}"
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
if [[ "${DS4_PATCH_FLASHINFER_SM12X_FUSED_MOE_JIT:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  "$RUNTIME_PYTHON" "$SCRIPT_DIR/ds4_patch_flashinfer_sm12x_fused_moe_jit.py"
fi
ds4_require_200g_fabric
ds4_run_nccl_preflight "$NNODES"
ds4_run_dsv4_native_preflight
ds4_run_native_blackwell_preflight
ds4_run_triton_jit_preflight

KV_OFFLOAD_ARGS=()
case "$DSV4_KV_OFFLOADING_SIZE" in
  ""|0|0.0|off|OFF|none|NONE|false|FALSE)
    export VLLM_USE_SIMPLE_KV_OFFLOAD=0
    unset VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT
    unset VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT
    unset VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK
    ;;
  *)
    export VLLM_USE_SIMPLE_KV_OFFLOAD="${VLLM_USE_SIMPLE_KV_OFFLOAD:-1}"
    export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT:-$HOME/ds4_hma_store/dsv4_flash_pp8/simple_cpu_offload}"
    export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT:-1}"
    export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK="${VLLM_SIMPLE_KV_OFFLOAD_PERSIST_RANK:-$(hostname)-dsv4-pp8-r${NODE_RANK}}"
    mkdir -p "$VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT"
    KV_OFFLOAD_ARGS=(--kv-offloading-size "$DSV4_KV_OFFLOADING_SIZE" --kv-offloading-backend native)
    ;;
esac

if [[ -n "${DSV4_FLASH_PP_LAYER_PARTITION:-}" ]]; then
  "$RUNTIME_PYTHON" - "$NNODES" "$DSV4_FLASH_PP_LAYER_PARTITION" <<'PY'
import sys
stages = int(sys.argv[1])
raw = sys.argv[2]
parts = [int(item) for item in raw.split(",") if item.strip()]
if len(parts) != stages:
    raise SystemExit(f"DSV4_FLASH_PP_LAYER_PARTITION has {len(parts)} stages but NNODES={stages}: {raw}")
if sum(parts) != 43:
    raise SystemExit(f"DSV4_FLASH_PP_LAYER_PARTITION must sum to 43 DSV4 decoder layers, got {sum(parts)}: {raw}")
if any(part <= 0 for part in parts):
    raise SystemExit(f"DSV4_FLASH_PP_LAYER_PARTITION stages must all be positive: {raw}")
PY
  export VLLM_PP_LAYER_PARTITION="$DSV4_FLASH_PP_LAYER_PARTITION"
else
  unset VLLM_PP_LAYER_PARTITION
fi

echo "DSV4 PP${NNODES} profile=$DS4_DSV4_PIPELINE_RAM_PROFILE max_model_len=$DSV4_MAX_MODEL_LEN max_num_seqs=$DSV4_MAX_NUM_SEQS max_num_batched_tokens=$DSV4_MAX_NUM_BATCHED_TOKENS kv_cache_memory_bytes=$DSV4_KV_CACHE_MEMORY_BYTES kv_offloading_size=$DSV4_KV_OFFLOADING_SIZE gpu_memory_utilization=$DSV4_GPU_MEMORY_UTILIZATION workspace_prealloc_bytes=$VLLM_WORKSPACE_PREALLOC_BYTES mq_max_chunks=$VLLM_MQ_MAX_CHUNKS pp_layer_partition=${DSV4_FLASH_PP_LAYER_PARTITION:-auto}" >&2

KV_CACHE_MEMORY_ARGS=()
case "$DSV4_KV_CACHE_MEMORY_BYTES" in
  ""|0|auto|AUTO|none|NONE)
    ;;
  *)
    KV_CACHE_MEMORY_ARGS=(--kv-cache-memory-bytes "$DSV4_KV_CACHE_MEMORY_BYTES")
    ;;
esac

LOGGING_ITERATION_ARGS=()
if [[ "${DSV4_ENABLE_LOGGING_ITERATION_DETAILS:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  LOGGING_ITERATION_ARGS=(--enable-logging-iteration-details)
fi

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
  --max-model-len "$DSV4_MAX_MODEL_LEN"
  --max-num-seqs "$DSV4_MAX_NUM_SEQS"
  --max-num-batched-tokens "$DSV4_MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$DSV4_GPU_MEMORY_UTILIZATION"
  "${KV_CACHE_MEMORY_ARGS[@]}"
  "${FLASHINFER_AUTOTUNE_ARGS[@]}"
  --block-size 256
  --kv-cache-dtype fp8
  --enable-prefix-caching
  "${ASYNC_SCHEDULING_ARGS[@]}"
  "${KV_OFFLOAD_ARGS[@]}"
  --kv-cache-metrics
  "${LOGGING_ITERATION_ARGS[@]}"
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
