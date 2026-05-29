# DS4 Qwen27 LMCache Runtime

This document is kept as a tombstone for the older LMCache MP plan.

Qwen3.6-27B is a hybrid GDN/full-attention model in this runtime. Do not use
`LMCacheMPConnector` for the DS4 Qwen27 lane. The MP connector cannot safely
handle multiple HMA block groups for this model.

Use the native `LMCacheConnectorV1` HMA path instead:

```text
kv_connector: LMCacheConnectorV1
kv_role: kv_both
kv_connector_extra_config.use_native: true
kv_connector_extra_config.lmcache_kv_cache_group_id: auto
--no-disable-hybrid-kv-cache-manager
--mamba-cache-mode align
```

The current eight-Spark resident recipe is documented in:

```text
docs/deployment/ds4-dual-8x-pipelines.md
```

The native adapter preserves HMA block-id groups and uses the selected
FullAttentionSpec block table for LMCache slot mapping. vLLM's PP worker
projection then limits external KV/state ownership to each Spark's local layer
slice.
