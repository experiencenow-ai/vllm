# DS4 Qwen27 LMCache MP Runtime

This fork is the DS4 vLLM runtime for both long-context model families:

- DSV4 uses the fork's native `SimpleCPUOffloadConnector`/DS4 persistent
  offload path. Do not route DSV4 through LMCache unless LMCache grows a
  DSV4/HMA-specific state contract.
- Qwen27 uses the built-in `LMCacheMPConnector` with an external `lmcache
  server`. This is the preferred Qwen external KV path because Qwen KV is
  normal dense KV, not DSV4 compressed MLA/HMA state.

The goal is one source-built `experiencenow-ai/vllm` runtime, not a runtime swap
between models. DS4 selects the model-specific connector at launch.

## Required Package Set

Install the KV connector requirements from this fork:

```bash
python -m pip install -r requirements/kv_connectors.txt
```

On Spark aarch64, build the LMCache wheel first and install that wheel into the
controlled vLLM environment:

```bash
tools/ds4_build_lmcache_wheel.sh \
  /home/spark7/standard-runtimes/vllm-main-gdn-nixl/venv/bin/python \
  /tmp/ds4_lmcache_wheels

/home/spark7/standard-runtimes/vllm-main-gdn-nixl/venv/bin/python \
  -m pip install --no-deps /tmp/ds4_lmcache_wheels/lmcache-0.4.5-*.whl
```

The wheel-first flow avoids surprising dependency churn in an already-running
vLLM environment.

LMCache's MP docs describe the same service split: vLLM connects with
`LMCacheMPConnector`, while `lmcache server` owns the external cache process.
Use the LMCache [MP quickstart](https://docs.lmcache.ai/mp/quickstart.html) and
[configuration reference](https://docs.lmcache.ai/mp/configuration.html) for
the upstream option surface, but keep the DS4 profile values pinned unless a new
Spark acceptance run proves a change.

## Qwen27 Launch Shape

Start the LMCache server before vLLM:

```bash
lmcache server \
  --host 127.0.0.1 \
  --port 5555 \
  --http-port 18080 \
  --l1-size-gb 16 \
  --eviction-policy LRU \
  --chunk-size 256 \
  --l1-use-lazy \
  --l1-init-size-gb 2 \
  --max-workers 4 \
  --hash-algorithm blake3 \
  --l2-adapter '{"type":"nixl_store","backend":"POSIX","backend_params":{"file_path":"/mnt/nvme/ds4_lmcache/qwen27/l2","use_direct_io":"false"},"pool_size":64}'
```

Then launch Qwen27 with the MP connector:

```bash
vllm serve /home/spark7/models/hf/Qwen/Qwen3.6-27B-FP8 \
  --host 127.0.0.1 \
  --port 18110 \
  --served-model-name Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.50 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --reasoning-parser qwen3 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'
```

`LMCacheMPConnector` currently raises if vLLM hands it more than one HMA block
group. For the Qwen27 DS4 profile that should be a single full-attention group;
if that is not true for a future Qwen build, fail the launch and fix the profile
or connector rather than silently disabling the best runtime mode.

## Request Contract

Client requests should use the same DS4 high-level cache contract regardless of
the model:

- `prefix_text`/`shared_prefix` for text prefixes the gateway can warm or reuse;
- `kv_cache_ref` for already-ingested external KV packages;
- normal OpenAI-compatible generation payloads for the model call.

The request must not contain raw KV tensors. For request-submitted cache data,
the gateway first sends the blob to the node-local or shared cache service,
receives a small `kv_cache_ref`, and then sends the generation request to the
same GPU node. vLLM still verifies prefix hashes and tensor layout before using
any block.

## Acceptance Gate

The Qwen LMCache profile is accepted only after:

1. `lmcache server` is running and its management port responds.
2. vLLM starts with `LMCacheMPConnector` and the configured LMCache host/port.
3. A cold long-prefix request succeeds.
4. A repeated request with the same prefix shows LMCache hits and lower prefill
   or TTFT than the cold request.
5. Restarting vLLM while keeping the LMCache server/store alive preserves the
   reusable prefix cache.

DSV4 has a separate acceptance gate because its useful long-context state is
compressed/HMA-specific. Passing this Qwen gate does not prove DSV4 persistence.
