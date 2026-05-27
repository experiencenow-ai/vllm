# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve import trim_memory_api
from vllm.entrypoints.serve.trim_memory_api import attach_router


class FakeEngine:

    def __init__(self, fail_external: bool = False):
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.fail_external = fail_external

    async def pause_generation(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("pause_generation", args, kwargs))

    async def reset_prefix_cache(self, *args: object, **kwargs: object) -> bool:
        self.calls.append(("reset_prefix_cache", args, kwargs))
        if self.fail_external and args == (True, True):
            raise RuntimeError("external reset not supported")
        return True

    async def resume_generation(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("resume_generation", args, kwargs))


def build_client(engine: FakeEngine) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app)


def test_trim_memory_pauses_resets_trims_and_resumes(monkeypatch):
    engine = FakeEngine()
    client = build_client(engine)
    monkeypatch.setattr(trim_memory_api, "_malloc_trim", lambda warnings: "ok:1")

    response = client.post("/v1/trim_memory?mode=wait")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["actions"]["mode"] == "wait"
    assert body["actions"]["paused"] is True
    assert body["actions"]["reset_external"] == "ok"
    assert body["actions"]["malloc_trim"] == "ok:1"
    assert body["actions"]["resumed"] is True
    assert engine.calls == [
        ("pause_generation", (), {"mode": "wait", "clear_cache": True}),
        ("reset_prefix_cache", (True, True), {}),
        ("resume_generation", (), {}),
    ]


def test_trim_memory_falls_back_to_local_reset(monkeypatch):
    engine = FakeEngine(fail_external=True)
    client = build_client(engine)
    monkeypatch.setattr(trim_memory_api, "_malloc_trim", lambda warnings: "ok:1")

    response = client.post("/v1/trim_memory")

    assert response.status_code == 200
    body = response.json()
    assert body["actions"]["reset_external"] == "unsupported_external_local_reset_ok"
    assert body["warnings"]
    assert engine.calls[1:] == [
        ("reset_prefix_cache", (True, True), {}),
        ("reset_prefix_cache", (True, False), {}),
        ("resume_generation", (), {}),
    ]


def test_trim_memory_rejects_unknown_mode():
    response = build_client(FakeEngine()).post("/v1/trim_memory?mode=drain")

    assert response.status_code == 400
    assert response.json()["status"] == "error"
