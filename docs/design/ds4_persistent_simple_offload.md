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

- `POST /v1/kv/lookup`
  - input: model key, rank key, ordered block hashes
  - output: ordered hit/miss metadata, cache generation, and optional leases
- `POST /v1/kv/materialize`
  - input: rank key and `(cpu_block_id, block_hash)` pairs selected by the
    scheduler
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

- `POST /v1/kv/lookup` before advertising a CPU prefix hit;
- `POST /v1/kv/materialize` before CPU-to-GPU load;
- `POST /v1/kv/store/prepare` and `POST /v1/kv/store/commit` after a completed
  GPU-to-CPU offload;
- `POST /v1/kv/scheduler/commit` after the scheduler observes that all workers
  committed the store event.

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

1. The caller uploads or registers the cache package with the external cache
   service, receiving a committed cache generation or cache reference.
2. The caller sends the normal vLLM generation request. vLLM derives the prompt
   block hashes, calls `/v1/kv/lookup`, and only uses blocks the service can
   materialize for the exact model, rank, dtype, tensor layout, and hash.

The generation request should not carry raw KV tensors directly. KV payloads are
model- and rank-specific and can be enormous. The safe request-level shape is a
small cache reference or generation tag that the cache service has already
validated. vLLM should still verify by hash before using any block.

## Qwen27

This API boundary is not DSV4-only. It can work for Qwen27 if Qwen27 is running
through `SimpleCPUOffloadConnector` with matching block hashes, tensor names,
dtype, tensor parallel rank, and tokenizer/model revision.

It does not make dense Qwen27 KV small. Qwen27 does not get DSV4's compressed
context-state advantage from this connector; it only gets durable/reusable CPU
offload blocks. For Qwen27 today, node-sticky APC or normal prefix warming may
still be the better first-line cache unless CPU offload is already enabled and
the external service can keep the relevant blocks near the serving node.
