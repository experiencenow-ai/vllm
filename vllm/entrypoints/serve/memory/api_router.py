# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
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
    mode: Literal["abort", "wait"] = Query(default="abort"),
    reset_external: bool = Query(default=True),
    release_offload_memory: bool = Query(default=True),
    malloc_trim: bool = Query(default=True),
    resume: bool = Query(default=True),
):
    """Minimize process memory without unloading model weights."""
    if release_offload_memory and not reset_external:
        raise HTTPException(
            status_code=400,
            detail="release_offload_memory=true requires reset_external=true",
        )
    logger.info(
        "Trimming memory: mode=%s reset_external=%s "
        "release_offload_memory=%s malloc_trim=%s resume=%s",
        mode,
        reset_external,
        release_offload_memory,
        malloc_trim,
        resume,
    )
    result = await engine_client(raw_request).trim_memory(
        mode=mode,
        reset_external=reset_external,
        release_offload_memory=release_offload_memory,
        malloc_trim=malloc_trim,
        resume=resume,
    )
    return JSONResponse(content={"status": "ok", "result": result})


def attach_router(app: FastAPI):
    app.include_router(router)
