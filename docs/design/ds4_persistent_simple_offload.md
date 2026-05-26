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
