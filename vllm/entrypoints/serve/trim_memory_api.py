# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import ctypes
import gc
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/trim_memory")
async def trim_memory(
    raw_request: Request,
    mode: str = Query(default="abort"),
    reset_external: bool = Query(default=True),
    release_offload_memory: bool = Query(default=True),
    malloc_trim: bool = Query(default=True),
    resume: bool = Query(default=True),
) -> JSONResponse:
    if mode not in {"abort", "wait"}:
        return _error(HTTPStatus.BAD_REQUEST, "mode must be abort or wait")
    warnings: list[str] = []
    actions: dict[str, Any] = {
        "mode": mode,
        "paused": False,
        "resumed": False,
        "reset_external": "skipped",
        "release_offload_memory": "skipped",
        "malloc_trim": "skipped",
    }
    engine = engine_client(raw_request)
    try:
        await engine.pause_generation(mode=mode, clear_cache=True)
        actions["paused"] = True
        if reset_external:
            actions["reset_external"] = await _try_reset_external(engine, warnings)
        if release_offload_memory:
            actions["release_offload_memory"] = "unsupported"
            warnings.append(
                "release_offload_memory requested, but this vLLM build exposes "
                "no SimpleCPUOffload release hook"
            )
        if malloc_trim:
            actions["malloc_trim"] = _malloc_trim(warnings)
        if resume:
            await engine.resume_generation()
            actions["resumed"] = True
        return JSONResponse(
            content={"status": "ok", "actions": actions, "warnings": warnings},
            status_code=HTTPStatus.OK.value,
        )
    except Exception as err:
        logger.exception("Failed to trim vLLM memory")
        return _error(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            f"failed to trim memory: {err}",
            warnings=warnings,
            actions=actions,
        )


async def _try_reset_external(engine: EngineClient, warnings: list[str]) -> str:
    try:
        await engine.reset_prefix_cache(True, True)
        return "ok"
    except Exception as err:
        warnings.append(f"external prefix-cache reset unsupported or failed: {err}")
    try:
        await engine.reset_prefix_cache(True, False)
        return "unsupported_external_local_reset_ok"
    except Exception as err:
        warnings.append(f"local prefix-cache reset after external failure failed: {err}")
        return "failed"


def _malloc_trim(warnings: list[str]) -> str:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        rc = int(libc.malloc_trim(0))
        return f"ok:{rc}"
    except Exception as err:
        warnings.append(f"malloc_trim failed or unavailable: {err}")
        return "failed"


def _error(
    status: HTTPStatus,
    message: str,
    *,
    warnings: list[str] | None = None,
    actions: dict[str, Any] | None = None,
) -> JSONResponse:
    content: dict[str, Any] = {"status": "error", "error": message}
    if warnings is not None:
        content["warnings"] = warnings
    if actions is not None:
        content["actions"] = actions
    return JSONResponse(content=content, status_code=status.value)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
