# DS4 Dual 8x Spark Pipelines

This deployment shape keeps both services resident on the same eight Sparks:

```text
qwen27-bf16-pp8:      Qwen3.6-27B dense BF16, text only, 8-way layer PP
                     LMCacheConnectorV1 native HMA, per-rank local SSD cache

deepseek-v4-flash-pp8: DeepSeek V4 Flash mixed FP4/FP8 as-is, text only,
                       8-way layer PP, SimpleCPUOffload persistent store
```

Do not launch these as two Ray jobs if the goal is co-residency on the same
GPUs. Ray placement groups reserve GPU resources for the first job and will not
place the second full-width job on the same devices. Use the vLLM multi-node
multiprocessing path with distinct master ports for the two services. CUDA will
schedule kernels from both processes; request-level fairness/admission belongs
in the DS4 gateway, not inside vLLM.

## Code changes required by this recipe

The Qwen3.6 text model no longer allocates the input embedding on every pipeline
rank. Only PP rank 0 owns `embed_tokens`; the last rank also owns it if a tied
embedding checkpoint requires that.

The native LMCache adapter no longer collapses HMA block IDs to group 0. It
preserves `tuple[list[int], ...]` block IDs for each request, auto-selects the
first local paged-attention group for LMCache slot mapping, and exposes the same
group id to `request_finished_all_groups()`. For Qwen3.6 HMA this selects the
FullAttentionSpec group instead of the GDN/Mamba state group.

vLLM already projects `KVCacheConfig` onto each PP worker's local layers before
KV allocation. With the adapter fix above, each Spark stores only the external
KV/state data for the layer slice it owns. The default recipe uses per-rank
home-owned local cache roots; use any special mount only by explicit override.

## Qwen27 BF16 8-way PP

Run one command on every Spark. Rank 0 starts the API server; ranks 1-7 run
headless workers.

```bash
export HEAD_ADDR=<spark0-private-ip-or-hostname>
export NODE_RANK=<0-7>
export API_PORT=8101
export MASTER_PORT=29527
export QWEN27_BF16_MODEL=/home/$USER/models/hf/Qwen/Qwen3.6-27B
export LMCACHE_ROOT=/home/$USER/ds4_lmcache/qwen27_pp8/spark${NODE_RANK}

/home/$USER/src/vllm/tools/ds4_launch_qwen27_pp8.sh
```

The launcher sets:

```text
--tensor-parallel-size 1
--pipeline-parallel-size 8
--nnodes 8
--dtype bfloat16
--language-model-only
--no-disable-hybrid-kv-cache-manager
--mamba-cache-mode align
VLLM_PP_LAYER_PARTITION=9,9,9,8,8,8,8,5
LMCache chunk_size=784
LMCache local_disk=file://$LMCACHE_ROOT
LMCacheConnectorV1 native adapter with lmcache_kv_cache_group_id=auto
```

The Qwen layer partition is intentionally not even. It keeps two full-attention
layers on each rank while moving decoder layers off the last rank to offset the
LM head:

```text
rank layers  count
0    0-8       9
1    9-17      9
2    18-26     9
3    27-34     8
4    35-42     8
5    43-50     8
6    51-58     8
7    59-63     5 + norm + lm_head
```

Override only if benchmark traces show a different slowest stage:

```bash
export QWEN27_PP_LAYER_PARTITION=9,9,9,8,8,8,8,5
```

The default queue-facing caps are intentionally conservative for the first
dual-resident rollout:

```text
QWEN27_MAX_NUM_SEQS=8
QWEN27_MAX_NUM_BATCHED_TOKENS=8192
QWEN27_GPU_MEMORY_UTILIZATION=0.24
QWEN27_KV_CACHE_MEMORY_BYTES=8589934592
LMCACHE_MAX_LOCAL_CPU_SIZE=2.0
DS4_ENABLE_FLASHINFER_AUTOTUNE=0
DS4_FLASHINFER_JIT_MAX_JOBS=1
QWEN27_ASYNC_SCHEDULING=1
PYTHONHASHSEED=0
```

Raise those caps only after both resident services are healthy together. Do not
start with `LMCACHE_MAX_LOCAL_CPU_SIZE=64.0`; a Spark gate run with that value
reached the LMCache FullAttentionSpec/hybrid-state initialization path and then
drove host `MemAvailable` below 1 GiB before the API became healthy. A later
NVFP4 PP2 smoke with `LMCACHE_MAX_LOCAL_CPU_SIZE=16.0` and
`QWEN27_GPU_MEMORY_UTILIZATION=0.55` also drove spark0 down to about 2.5 GiB
available during FlashInfer FP4 autotune. Retesting at 8 GiB local CPU and 0.50
GPU utilization still let a later autotune pass drive a rank near zero available
memory. The default host cache is therefore capped at 2 GiB, Qwen defaults to an
explicit 8 GiB per-rank KV cache, NVFP4 defaults GPU utilization to 0.24 as a
fallback for manual `QWEN27_KV_CACHE_MEMORY_BYTES=auto` runs, and
production/validation launchers fail closed with
`--no-enable-flashinfer-autotune`. FlashInfer runtime CUTLASS JIT can still spawn
heavy `cicc` compiles even with autotune disabled, so the DS4 launchers default
`MAX_JOBS` to `DS4_FLASHINFER_JIT_MAX_JOBS=1`. Raise it only in a dedicated
warmup/tuning job on idle nodes. FlashInfer autotune itself is only allowed from
an explicit tuning wrapper such as
`tools/ds4_launch_dsv4_flash_tp2_flashinfer_autotune.sh`; setting
`DS4_ENABLE_FLASHINFER_AUTOTUNE=1` directly in a production/validation launcher
is an error. The same failure path reproduced with async disabled, so async was
not the isolated trigger and the Qwen launcher enables it by default. Set
`QWEN27_ASYNC_SCHEDULING=0` only as a rollback or bisection switch.

## DeepSeek V4 Flash 8-way PP

Run one command on every Spark. Use a different master port from Qwen.

```bash
export HEAD_ADDR=<spark0-private-ip-or-hostname>
export NODE_RANK=<0-7>
export API_PORT=8102
export MASTER_PORT=29544
export DSV4_FLASH_MODEL=/home/$USER/models/hf/deepseek-ai/DeepSeek-V4-Flash
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT=/home/$USER/ds4_hma_store/dsv4_flash_pp8/simple_cpu_offload/spark${NODE_RANK}

/home/$USER/src/vllm/tools/ds4_launch_dsv4_flash_pp8.sh
```

The launcher preserves the existing DSV4 Flash recipe:

```text
DSV4_MAX_NUM_SEQS=8
DSV4_MAX_NUM_BATCHED_TOKENS=16384
DSV4_GPU_MEMORY_UTILIZATION=0.82
DSV4_KV_CACHE_MEMORY_BYTES=12884901888
DS4_ENABLE_FLASHINFER_AUTOTUNE=0
DS4_FLASHINFER_JIT_MAX_JOBS=1
DSV4_DISABLE_MTP=1 for first memory bringup, then unset after health is proven
--tensor-parallel-size 1
--pipeline-parallel-size 8
--nnodes 8
--block-size 256
--kv-cache-dtype fp8
native MXFP4/FP8 only: FlashInfer CUTLASS or explicitly validated TRTLLM
no Marlin, no DeepGEMM MXFP4 on SM12x unless explicitly opted in
CUDA graph compilation enabled
```

The DSV4 script leaves `VLLM_PP_LAYER_PARTITION` unset by default so vLLM uses
its normal non-even split for the model's layer count. Set
`DSV4_FLASH_PP_LAYER_PARTITION` only after profiling stage time. If the DSV4
stage needs more GPU memory, raise `DSV4_GPU_MEMORY_UTILIZATION` together with a
matching reduction in Qwen admission or KV budget. Do not use TP2 health as the
memory target for production residency; it is a native-kernel reproduction lane.
On Spark0/Spark1 TP2 reached the correct native backend with autotune disabled
and MTP disabled, but runtime FlashInfer CUTLASS JIT fanout drove available host
memory below the safety floor before `/health` came up.

## Admission control

Two independent vLLM services do not have a request-level shared scheduler. They
can be resident together with the multiprocessing launch above, but they will not
coordinate prompt admission or batch sizing across services. The DS4 gateway
should own that policy.

A minimal policy that matches this deployment:

```text
qwen27 queue hot, dsv4 queue cold:    admit Qwen up to its full batched cap
dsv4 queue hot, qwen27 queue cold:    admit DSV4 up to its full batched cap
both queues hot:                      split admission by priority or weight
interactive request present:          reserve low-latency slots before WM batch work
```

Do not solve this by fragmenting Sparks back into static model ownership. The
point of the dual PP layout is to keep layer residency and SSD cache ownership
stable while moving admission decisions to the gateway.

## Health and acceptance checks

Qwen:

```bash
curl -fsS http://127.0.0.1:8101/health
curl -fsS http://127.0.0.1:8101/v1/models
```

Expected Qwen log lines:

```text
VLLM_PP_LAYER_PARTITION=9,9,9,8,8,8,8,5
LMCache HMA slot mapping will use KV cache group <FullAttentionSpec group>
No LMCacheConnectorV1 does not support HMA error
No failed to convert the KV cache specs to one unified type error
Repeated long-prefix request logs an LMCache hit after trim_memory reset
```

DSV4:

```bash
curl -fsS http://127.0.0.1:8102/health
curl -fsS http://127.0.0.1:8102/v1/models
```

Expected DSV4 log lines:

```text
VLLM_USE_SIMPLE_KV_OFFLOAD=1
DS4 persistent SimpleCPUOffload worker store enabled
DS4 persistent SimpleCPUOffload scheduler hit: <nonzero> tokens on replay
```

Per-rank SSD ownership check:

```bash
find /home/$USER/ds4_lmcache/qwen27_pp8 -maxdepth 3 -type f | wc -l
find /home/$USER/ds4_hma_store/dsv4_flash_pp8/simple_cpu_offload -maxdepth 4 -type f | wc -l
```

Each Spark should grow its own local cache tree. A shared path with all ranks
writing to the same directory defeats the operational benefit.
