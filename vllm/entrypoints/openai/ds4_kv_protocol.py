# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


def extract_ds4_plan(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    value = data.get("ds4_kv_cache")
    if isinstance(value, dict):
        return value
    extra = data.get("extra_body")
    if isinstance(extra, dict) and isinstance(extra.get("ds4_kv_cache"), dict):
        return extra["ds4_kv_cache"]
    params = data.get("kv_transfer_params")
    if isinstance(params, dict) and isinstance(params.get("ds4_kv_cache"), dict):
        return params["ds4_kv_cache"]
    return None


def cache_ref_from_plan(plan: dict[str, Any]) -> str | None:
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


def kv_transfer_params_from_plan(plan: dict[str, Any], existing: Any = None) -> dict[str, Any]:
    params = dict(existing) if isinstance(existing, dict) else {}
    params["ds4_kv_cache"] = dict(plan)
    params["ds4_require_kv_transfer"] = True
    cache_ref = cache_ref_from_plan(plan)
    if cache_ref:
        params.setdefault("cache_ref", cache_ref)
        params.setdefault("ds4_cache_ref", cache_ref)
        params.setdefault("simple_kv_cache_ref", cache_ref)
    fingerprint = plan.get("engine_fingerprint_hash")
    if isinstance(fingerprint, str) and fingerprint:
        params.setdefault("ds4_engine_fingerprint_hash", fingerprint)
    ticket = params.get("prefetch_ticket") or plan.get("prefetch_ticket")
    if isinstance(ticket, str) and ticket:
        params.setdefault("prefetch_ticket", ticket)
        params.setdefault("ds4_prefetch_ticket", ticket)
    return params
