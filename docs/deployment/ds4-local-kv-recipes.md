# DS4 Local KV Recipes

This page records the local DS4 KV-cache recipes used by the Spark runtime
fork. Keep benchmark claims tied to the source commit, hardware, launch flags,
and validation logs.

## DeepSeek V4 Flash

The current DS4 production lane uses the host-local vLLM fork rather than the
older Docker copy-patch runtime:

```text
repository: https://github.com/experiencenow-ai/vllm
baseline: d240cdbcf3de175be57c108fd9cbfce04009ec29
runtime: /home/spark*/ds4-vllm-local editable install
source:  /home/spark*/src/vllm-b55c3b6-docker-lineage
```

Use one grouped vLLM service across the Spark tensor-parallel lane. The live
requalification shape used spark4 as head and spark7 as worker while spark5 was
unavailable; the normal lane is spark4 plus spark5.

Critical launch settings:

```bash
export VLLM_USE_SIMPLE_KV_OFFLOAD=1
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT=/var/tmp/ds4_hma_store/dsv4/simple_cpu_offload
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT=1

vllm serve /home/spark4/models/hf/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --max-model-len 262144 \
  --block-size 256 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --kv-offloading-size 8 \
  --kv-offloading-backend native \
  --kv-cache-metrics \
  --enable-logging-iteration-details \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --no-disable-hybrid-kv-cache-manager \
  --enforce-eager
```

Do not disable the hybrid KV cache manager for DSV4. Keep
`VLLM_USE_SIMPLE_KV_OFFLOAD=1`; otherwise the runtime can select the generic
offloading connector and miss the persistent SimpleCPUOffload store hooks.

Keep `--block-size 256`. DSV4 native offload uses the model-specific group hash
size internally while the scheduler block size remains 256.

The 2026-05-28 replay benchmark used a 6733-token shared prefix and observed:

```text
cold elapsed: 31.621346s
warm elapsed: 3.455483s
speedup: 9.151064x
DS4 persistent SimpleCPUOffload scheduler hit: 6144 tokens
warm replay computed context: 589 tokens
external prefix cache hit rate: 45.6%
```

Validation checklist:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  --data-binary @/tmp/ds4_chat_probe.json
curl -fsS -X POST http://127.0.0.1:8000/v1/trim_memory \
  -H 'content-type: application/json' \
  --data '{"reset_prefix_cache":true,"malloc_trim":true}'
```

The model response should include `deepseek-v4-flash` and the configured
`max_model_len`. The route list should include `POST /v1/trim_memory`.

## Qwen27 LMCache HMA

Qwen3.5/Qwen3.6-style models are hybrid GDN models. They do not have one
uniform attention-only KV layout, so do not launch Qwen27 LMCache with
`--disable-hybrid-kv-cache-manager`.

The required stack is:

```text
vLLM: LMCacheConnectorV1 implements SupportsHMA and forwards the adapter-selected KV group
vLLM: Mamba/GDN scheduler and prefix-cache fixes for external computed tokens
LMCache: companion hybrid-state restore branch selecting FullAttentionSpec groups
```

Install the LMCache companion branch into the same runtime environment used by
the vLLM server:

```bash
git clone https://github.com/LMCache/LMCache.git /home/$USER/src/LMCache-qwen-hma
cd /home/$USER/src/LMCache-qwen-hma
git fetch origin pull/3284/head:qwen-hybrid-state-cache
git checkout qwen-hybrid-state-cache
/home/$USER/ds4-vllm-local/bin/python -m pip install -e .
```

When installing on Spark runtimes without system Python development headers in
the default include path, set the CUDA and Python include paths before running
`pip install`:

```bash
export CUDA_HOME=/usr/local/cuda
export CPATH=/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include:/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include/python3.12:${CPATH:-}
```

Launch Qwen27 with HMA enabled:

```bash
export LMCACHE_CONFIG_FILE=/tmp/lmcache_qwen27.yaml
export LMCACHE_ROOT=/home/$USER/ds4_lmcache/qwen27
mkdir -p "$LMCACHE_ROOT"

cat > "$LMCACHE_CONFIG_FILE" <<YAML
chunk_size: 784
local_cpu: true
max_local_cpu_size: 64.0
local_disk: file://$LMCACHE_ROOT
max_local_disk_size: 1024.0
YAML

export PATH=/home/$USER/ds4-vllm-local/bin:$PATH
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_USE_V1=1 \
/home/$USER/ds4-vllm-local/bin/python -m vllm.entrypoints.cli.main serve \
  /home/$USER/models/hf/Qwen/Qwen3.6-27B-FP8 \
  --served-model-name qwen27 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-batched-tokens 7840 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --mamba-cache-mode align \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

For Qwen3.6-27B-FP8 on spark7, vLLM sets the HMA attention page size to 784
tokens. Keep the LMCache `chunk_size` and chunked-prefill budget aligned to
that page size. The smoke used `chunk_size: 784` and
`--max-num-batched-tokens 7840`; the earlier `chunk_size: 256` launch started
successfully but LMCache rejected external hits because matching hybrid-state
pages were not available at the same token boundary.

Acceptance requires all of the following:

```text
Qwen27 starts with HMA plus LMCache.
No "LMCacheConnectorV1 does not support HMA" error.
No "failed to convert the KV cache specs to one unified type" error.
Logs show LMCache selecting a FullAttentionSpec group.
Logs show hybrid state groups detected or restored.
Repeated long-prefix request has materially lower TTFT.
After resetting vLLM prefix cache, a repeated request logs "Loaded hybrid state"
and "Retrieved ... out of ... required tokens".
```

The spark7 smoke on 2026-05-28 used a 16206-token prompt. With the aligned
recipe, the first request stored hybrid state at 7840 and 15680 tokens. After
`POST /v1/trim_memory` reset the local vLLM prefix cache, the next request
logged:

```text
LMCache hit tokens: 15680, need to load: 15680
Loaded hybrid state ... at 15680 token(s)
Retrieved 15680 out of 15680 required tokens
External prefix cache hit rate: 93.7%
```

## Spark Rollout

Keep the vLLM source tree and runtime branch identical on every node that may
host the service:

```bash
cd /home/$USER/src/vllm-b55c3b6-docker-lineage
git fetch experiencenow
git checkout main
/home/$USER/ds4-vllm-local/bin/python -m py_compile \
  vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/core/single_type_kv_cache_manager.py \
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
```

For DSV4 nodes, no LMCache install is required. Verify the runtime still uses
SimpleCPUOffload and MTP by checking the launch command for:

```text
VLLM_USE_SIMPLE_KV_OFFLOAD=1
--kv-offloading-backend native
--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
--no-disable-hybrid-kv-cache-manager
```

For Qwen nodes, install the companion LMCache branch into the exact Python
environment used to start vLLM, then use the aligned Qwen recipe above. The
`/home/spark*/ds4-vllm-local/bin/vllm` wrapper may point at a different
interpreter on some Spark images; prefer:

```bash
export PATH=/home/$USER/ds4-vllm-local/bin:$PATH
/home/$USER/ds4-vllm-local/bin/python -m vllm.entrypoints.cli.main serve ...
```
