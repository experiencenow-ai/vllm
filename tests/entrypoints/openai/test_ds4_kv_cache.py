# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.ds4_kv_cache import requires_ds4_kv_transfer


def _plan() -> dict:
    return {
        "format": "ds4-kv-cache-plan-v1",
        "backend": "simple_cpu_offload",
        "cache_id": "prefix-a",
        "load": {
            "mode": "require",
            "transport": "local_store",
            "cache_key": "prefix-a",
            "sha256": "sha256:" + ("a" * 64),
        },
        "store": {"mode": "skip", "transport": "none"},
        "miss_policy": "fail",
        "route_affinity": "required",
        "model_fingerprint": {},
        "operation": "load",
        "batch_key_hash": "sha256:" + ("b" * 64),
    }


def test_completion_lifts_ds4_kv_cache_extra_body_to_kv_transfer_params():
    req = CompletionRequest(
        model="test-model",
        prompt="hello",
        extra_body={"ds4_kv_cache": _plan()},
    )

    assert req.kv_transfer_params is not None
    assert req.kv_transfer_params["ds4_kv_cache"] == _plan()
    assert req.kv_transfer_params["cache_ref"] == "prefix-a"
    assert requires_ds4_kv_transfer(req.kv_transfer_params)


def test_chat_lifts_ds4_kv_cache_extra_body_to_kv_transfer_params():
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"ds4_kv_cache": _plan()},
    )

    assert req.kv_transfer_params is not None
    assert req.kv_transfer_params["ds4_kv_cache"] == _plan()
    assert req.kv_transfer_params["ds4_cache_ref"] == "prefix-a"
    assert requires_ds4_kv_transfer(req.kv_transfer_params)
