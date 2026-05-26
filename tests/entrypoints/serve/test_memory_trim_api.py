# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.memory.api_router import attach_router


class FakeEngineClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def trim_memory(
        self,
        *,
        mode: str = "abort",
        reset_external: bool = True,
        release_offload_memory: bool = True,
        malloc_trim: bool = True,
        resume: bool = True,
    ) -> dict[str, object]:
        call = {
            "mode": mode,
            "reset_external": reset_external,
            "release_offload_memory": release_offload_memory,
            "malloc_trim": malloc_trim,
            "resume": resume,
        }
        self.calls.append(call)
        return {"prefix_cache_reset": True, "call": call}


def _client() -> tuple[TestClient, FakeEngineClient]:
    app = FastAPI()
    fake = FakeEngineClient()
    app.state.engine_client = fake
    attach_router(app)
    return TestClient(app), fake


def test_trim_memory_route_calls_engine_with_defaults() -> None:
    client, fake = _client()

    response = client.post("/v1/trim_memory")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert fake.calls == [
        {
            "mode": "abort",
            "reset_external": True,
            "release_offload_memory": True,
            "malloc_trim": True,
            "resume": True,
        }
    ]


def test_trim_memory_route_accepts_overrides() -> None:
    client, fake = _client()

    response = client.post(
        "/v1/trim_memory",
        params={
            "mode": "wait",
            "reset_external": "false",
            "release_offload_memory": "false",
            "malloc_trim": "false",
            "resume": "false",
        },
    )

    assert response.status_code == 200
    assert fake.calls == [
        {
            "mode": "wait",
            "reset_external": False,
            "release_offload_memory": False,
            "malloc_trim": False,
            "resume": False,
        }
    ]


def test_trim_memory_rejects_releasing_offload_without_external_reset() -> None:
    client, fake = _client()

    response = client.post(
        "/v1/trim_memory",
        params={
            "reset_external": "false",
            "release_offload_memory": "true",
        },
    )

    assert response.status_code == 400
    assert fake.calls == []
