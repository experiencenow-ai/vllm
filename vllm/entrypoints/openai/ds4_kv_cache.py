# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


def lift_ds4_kv_cache_request(data: Any) -> Any:
    """Map DS4 cache directives onto vLLM's per-request KV transfer field."""
    if not isinstance(data, dict):
        return data
    ds4_kv_cache = data.get("ds4_kv_cache")
    extra_body = data.get("extra_body")
    if ds4_kv_cache is None and isinstance(extra_body, dict):
        ds4_kv_cache = extra_body.get("ds4_kv_cache")
    if ds4_kv_cache is None:
        return data
    if not isinstance(ds4_kv_cache, dict):
        raise ValueError("ds4_kv_cache must be an object")
    kv_transfer_params = data.get("kv_transfer_params")
    if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
        raise ValueError("kv_transfer_params must be an object")
    lifted = dict(kv_transfer_params or {})
    existing = lifted.get("ds4_kv_cache")
    if existing is not None and existing != ds4_kv_cache:
        raise ValueError("kv_transfer_params.ds4_kv_cache conflicts with ds4_kv_cache")
    lifted["ds4_kv_cache"] = dict(ds4_kv_cache)
    lifted["ds4_require_kv_transfer"] = True
    cache_ref = _ds4_cache_ref(ds4_kv_cache)
    if cache_ref is not None:
        lifted.setdefault("cache_ref", cache_ref)
        lifted.setdefault("ds4_cache_ref", cache_ref)
        lifted.setdefault("simple_kv_cache_ref", cache_ref)
    out = dict(data)
    out["kv_transfer_params"] = lifted
    return out


def requires_ds4_kv_transfer(params: Any) -> bool:
    return isinstance(params, dict) and bool(
        params.get("ds4_require_kv_transfer") or params.get("ds4_kv_cache") is not None
    )


def _ds4_cache_ref(plan: dict[str, Any]) -> str | None:
    for key in ("cache_id", "prefix_hash"):
        value = plan.get(key)
        if isinstance(value, str) and value:
            return value
    load = plan.get("load")
    if isinstance(load, dict):
        for key in ("cache_key", "kv_key"):
            value = load.get(key)
            if isinstance(value, str) and value:
                return value
    return None
