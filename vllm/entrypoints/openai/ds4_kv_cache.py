# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.entrypoints.openai.ds4_kv_protocol import (
    cache_ref_from_plan,
    extract_ds4_plan,
    kv_transfer_params_from_plan,
)


def lift_ds4_kv_cache_request(data: Any) -> Any:
    """Map DS4 cache directives onto vLLM's per-request KV transfer field."""
    if not isinstance(data, dict):
        return data
    ds4_kv_cache = extract_ds4_plan(data)
    if ds4_kv_cache is None:
        return data
    if not isinstance(ds4_kv_cache, dict):
        raise ValueError("ds4_kv_cache must be an object")
    kv_transfer_params = data.get("kv_transfer_params")
    if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
        raise ValueError("kv_transfer_params must be an object")
    existing = kv_transfer_params.get("ds4_kv_cache") if isinstance(kv_transfer_params, dict) else None
    if existing is not None and existing != ds4_kv_cache:
        raise ValueError("kv_transfer_params.ds4_kv_cache conflicts with ds4_kv_cache")
    out = dict(data)
    out["kv_transfer_params"] = kv_transfer_params_from_plan(ds4_kv_cache, kv_transfer_params)
    return out


def requires_ds4_kv_transfer(params: Any) -> bool:
    return isinstance(params, dict) and bool(
        params.get("ds4_require_kv_transfer") or params.get("ds4_kv_cache") is not None
    )


_ds4_cache_ref = cache_ref_from_plan
