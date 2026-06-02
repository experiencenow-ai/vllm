# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cache.api_router import attach_router


class FakeEngineClient:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_external: bool = False,
    ) -> bool:
        self.calls.append((reset_running_requests, reset_external))
        return True


def _client() -> tuple[TestClient, FakeEngineClient]:
    app = FastAPI()
    fake = FakeEngineClient()
    app.state.engine_client = fake
    attach_router(app)
    return TestClient(app), fake


def test_cache_router_hidden_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    monkeypatch.delenv("VLLM_DS4_ENABLE_CACHE_ADMIN", raising=False)
    client, fake = _client()

    response = client.post("/reset_prefix_cache")

    assert response.status_code == 404
    assert fake.calls == []


def test_ds4_cache_admin_exposes_local_prefix_reset(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    monkeypatch.setenv("VLLM_DS4_ENABLE_CACHE_ADMIN", "1")
    client, fake = _client()

    response = client.post("/reset_prefix_cache?reset_external=false")

    assert response.status_code == 200
    assert fake.calls == [(False, False)]


def test_ds4_cache_admin_can_forward_external_reset(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    monkeypatch.setenv("VLLM_DS4_ENABLE_CACHE_ADMIN", "1")
    client, fake = _client()

    response = client.post(
        "/reset_prefix_cache",
        params={
            "reset_running_requests": "true",
            "reset_external": "true",
        },
    )

    assert response.status_code == 200
    assert fake.calls == [(True, True)]
