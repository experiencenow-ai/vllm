# DS4 Persistent Simple CPU Offload

This is an experimental DS4 path for making vLLM's
`SimpleCPUOffloadConnector` survive process restarts without pretending that a
generic KV tensor cache understands DeepSeek V4's HMA, MLA, sliding-window, and
compressed-state constraints.

## Current Patch

The current implementation is deliberately narrow:

- the scheduler records CPU block hashes after a completed CPU offload store;
- the worker persists the matching CPU KV block payloads after the GPU-to-CPU
  transfer completes;
- restart restores scheduler metadata and advertises hits with one aligned HMA
  guard block withheld;
- when `VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_URL` is set, the scheduler asks the
  cache service only about hashes from the current request instead of loading
  the lifetime cache index at startup;
- worker tensor payloads are restored lazily, only when a scheduled load names
  the exact CPU block IDs and hashes to materialize.

The worker must not restore every persisted tensor block during startup. That is
not scalable for a durable cache.

## External Cache API Target

The durable cache should be managed by an external node-local service. vLLM
should be a client, not the owner of eviction, disk layout, compaction, or
corruption policy.

The API should stay a small control plane. Large KV tensors should move through
node-local files, shared memory, or another explicit data handle, not through
JSON request bodies.

Recommended endpoints:

- `POST /v1/kv/ingest`
  - input: cache package blob or package handle, model key, tokenizer/model
    revision, tensor layout, rank metadata, manifest, and checksums
  - output: committed `cache_ref` plus indexed block hashes
- `POST /v1/kv/lookup`
  - input: model key, rank key, optional `cache_ref`, ordered block hashes
  - output: ordered hit/miss metadata, cache generation, and optional leases
- `POST /v1/kv/materialize`
  - input: rank key and `(cpu_block_id, block_hash, cache_ref)` pairs selected
    by the scheduler
  - output: local payload handles for the worker to copy into CPU KV slots
- `POST /v1/kv/store/prepare`
  - input: rank key and `(cpu_block_id, block_hash)` pairs after a completed
    GPU-to-CPU offload
  - output: staging handles owned by the cache service
- `POST /v1/kv/store/commit`
  - input: staging handles, hashes, byte counts, checksums, and tensor names
  - output: committed cache records
- `POST /v1/kv/release`
  - input: leases or cache generation handles that are no longer pinned
- `GET /v1/kv/stats`
  - output: bytes, block counts, hit/miss counters, corrupt records, evictions

The initial client supports:

- `VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_URL`
- `VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_TOKEN`
- `VLLM_SIMPLE_KV_OFFLOAD_PERSIST_API_TIMEOUT`

When the API URL is set, vLLM uses:

- `POST /v1/kv/lookup` before advertising a CPU prefix hit, including
  `cache_ref` when the request provides one;
- `POST /v1/kv/materialize` before CPU-to-GPU load, including per-block
  `cache_ref` values;
- `POST /v1/kv/store/prepare` and `POST /v1/kv/store/commit` after a completed
  GPU-to-CPU offload;
- `POST /v1/kv/scheduler/commit` after the scheduler observes that all workers
  committed the store event.

Per-request cache refs are passed through existing vLLM request plumbing:

```json
{
  "extra_args": {
    "kv_transfer_params": {
      "cache_ref": "cachepkg_..."
    }
  }
}
```

Failure policy should remain fail-closed for this DS4 path. If the service says
a block exists but the worker cannot materialize the exact hash into the exact
CPU slot that the scheduler selected, the request should visibly fail instead
of silently recomputing or loading wrong state.

## vLLM Boundary

The vLLM side should eventually reduce to three responsibilities:

1. Hash lookup before accepting an external prefix hit.
2. Materialize only the selected `(cpu_block_id, block_hash)` pairs before a
   CPU-to-GPU load.
3. Store only completed offload blocks and report completion after the cache
   service commits them.

The cache service should own TTLs, leases, disk manifests, checksums, compaction,
cross-process coordination, and admission/eviction policy.

## Request-Submitted Cache Data

Submitting cached data with a generation request is possible only as a two-step
protocol:

1. The gateway routes the request to one GPU node and pushes the bundled cache
   package to that node's cache service with `/v1/kv/ingest`.
2. The cache service validates and indexes the package, returning `cache_ref`.
3. The gateway sends the normal vLLM generation request to the same node with
   `kv_transfer_params.cache_ref`.
4. vLLM derives the prompt block hashes, calls `/v1/kv/lookup`, and only uses
   blocks the service can materialize for the exact model, rank, dtype, tensor
   layout, and hash.

The generation request should not carry raw KV tensors directly. KV payloads are
model- and rank-specific and can be enormous. The safe request-level shape is a
small cache reference or generation tag that the cache service has already
validated. vLLM should still verify by hash before using any block.

## Qwen27

Qwen27 should not use this DSV4 persistent SimpleCPUOffload path as its primary
external KV implementation. Qwen3.6-27B is hybrid GDN/full-attention in this
runtime, so the DS4 Qwen lane uses native `LMCacheConnectorV1` with HMA enabled
and `lmcache_kv_cache_group_id=auto`. Do not route Qwen27 through
`LMCacheMPConnector`. See [DS4 Qwen27 LMCache Runtime](ds4_qwen_lmcache_mp.md)
and [DS4 Dual 8x Spark Pipelines](../deployment/ds4-dual-8x-pipelines.md).

The shared DS4 API can still expose one high-level contract to clients:
`prefix_text` or `shared_prefix` for text prefixes, and `kv_cache_ref` for
already-ingested cache packages. The model-specific vLLM launch decides whether
that maps to DSV4 native offload or Qwen native LMCache HMA.
