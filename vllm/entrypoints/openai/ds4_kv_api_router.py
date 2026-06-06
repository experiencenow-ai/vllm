# SPDX-License-Identifier: Apache-2.0
"""DS4 KV-cache prefetch endpoint.

This route is explicit about its residency guarantees. The current
implementation can use vLLM's normal completion path to prewarm or
compute/store a prefix under a DS4 cache ref. It does not claim an adoptable
GPU ticket unless a future block-manager path returns one.
"""

from __future__ import annotations

from http import HTTPStatus
import os
import threading
import time
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.ds4_kv_protocol import (
    cache_ref_from_plan,
    extract_ds4_plan,
    kv_transfer_params_from_plan,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()
_PREFETCH_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.environ.get("VLLM_DS4_KV_PREFETCH_MAX_CONCURRENT", "4") or "4")))

DS4_KV_PREFETCH_FORMAT = "ds4-vllm-kv-prefetch-v2"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


@router.post("/ds4/kv/prefetch")
async def ds4_kv_prefetch(raw_request: Request):
    if not _env_bool("VLLM_DS4_KV_PREFETCH_API", False):
        return JSONResponse(status_code=HTTPStatus.NOT_FOUND.value, content={"error": "DS4 KV prefetch API is disabled"})
    auth_error = _authorize_prefetch_request(raw_request)
    if auth_error is not None:
        return auth_error
    if not _PREFETCH_SEMAPHORE.acquire(blocking=False):
        return JSONResponse(status_code=HTTPStatus.TOO_MANY_REQUESTS.value, content={"error": "too many DS4 KV prefetch requests"})
    try:
        try:
            body = await raw_request.json()
        except Exception as exc:
            return JSONResponse(status_code=HTTPStatus.BAD_REQUEST.value, content={"error": f"invalid JSON: {exc}"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=HTTPStatus.BAD_REQUEST.value, content={"error": "request body must be an object"})
        result = await _run_ds4_kv_prefetch_body(raw_request, body)
        status = HTTPStatus.OK.value if str(result.get("status") or "") not in {"failed", "error"} else HTTPStatus.INTERNAL_SERVER_ERROR.value
        if result.get("http_status") is not None:
            status = int(result["http_status"])
        return JSONResponse(status_code=status, content=result)
    finally:
        _PREFETCH_SEMAPHORE.release()


async def _run_ds4_kv_prefetch_body(raw_request: Request, body: dict[str, Any]) -> dict[str, Any]:
    handler = getattr(raw_request.app.state, "openai_serving_completion", None)
    if handler is None:
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "http_status": HTTPStatus.NOT_IMPLEMENTED.value, "error": "completion serving is not available"}
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "http_status": HTTPStatus.BAD_REQUEST.value, "error": "prompt is required"}
    max_prompt_bytes = _env_int("VLLM_DS4_KV_PREFETCH_MAX_PROMPT_BYTES", 256 * 1024 * 1024)
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_prompt_bytes:
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "http_status": HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value, "error": "prompt exceeds VLLM_DS4_KV_PREFETCH_MAX_PROMPT_BYTES", "prompt_bytes": prompt_bytes, "limit": max_prompt_bytes}
    model = body.get("model")
    if not isinstance(model, str) or not model:
        model = _default_served_model(raw_request)
    plan = extract_ds4_plan(body)
    if not isinstance(plan, dict):
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "http_status": HTTPStatus.BAD_REQUEST.value, "error": "ds4_kv_cache plan is required"}
    requested_mode = str(body.get("mode") or "cpu").lower()
    if requested_mode not in {"cpu", "completion", "gpu", "adoptable"}:
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "http_status": HTTPStatus.BAD_REQUEST.value, "error": f"unsupported prefetch mode: {requested_mode}"}
    started_at = time.time()
    max_tokens = max(0, int(body.get("max_tokens") or 0))
    generation_tokens = max(1, max_tokens)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": generation_tokens,
        "temperature": float(body.get("temperature") or 0.0),
        "stream": False,
        "ds4_kv_cache": plan,
        "extra_body": {"ds4_kv_cache": plan},
        "kv_transfer_params": kv_transfer_params_from_plan(plan, body.get("kv_transfer_params")),
    }
    if body.get("request_id") is not None:
        payload["request_id"] = str(body["request_id"])
    try:
        request = CompletionRequest.model_validate(payload)
        result = await handler.create_completion(request, raw_request)
    except Exception as exc:
        logger.warning("DS4 KV prefetch failed before completion path", exc_info=True)
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "error": str(exc)}
    if isinstance(result, ErrorResponse):
        return {"format": DS4_KV_PREFETCH_FORMAT, "status": "failed", "http_status": result.error.code, "error": result.model_dump()}
    cache_ref = cache_ref_from_plan(plan)
    elapsed_ms = (time.time() - started_at) * 1000.0
    return {
        "format": DS4_KV_PREFETCH_FORMAT,
        "status": "completion_prewarmed",
        "residency": "completion_prewarmed",
        "adoptable": False,
        "prefetch_ticket": None,
        "model": model,
        "cache_ref": cache_ref,
        "prefix_chars": len(prompt),
        "prompt_bytes": prompt_bytes,
        "used_completion_path": True,
        "forced_decode_tokens": generation_tokens,
        "requested_mode": requested_mode,
        "elapsed_ms": elapsed_ms,
    }


@router.post("/ds4/kv/prefetch_many")
async def ds4_kv_prefetch_many(raw_request: Request):
    if not _env_bool("VLLM_DS4_KV_PREFETCH_API", False):
        return JSONResponse(status_code=HTTPStatus.NOT_FOUND.value, content={"error": "DS4 KV prefetch API is disabled"})
    auth_error = _authorize_prefetch_request(raw_request)
    if auth_error is not None:
        return auth_error
    if not _PREFETCH_SEMAPHORE.acquire(blocking=False):
        return JSONResponse(status_code=HTTPStatus.TOO_MANY_REQUESTS.value, content={"error": "too many DS4 KV prefetch requests"})
    try:
        try:
            body = await raw_request.json()
        except Exception as exc:
            return JSONResponse(status_code=HTTPStatus.BAD_REQUEST.value, content={"error": f"invalid JSON: {exc}"})
        if not isinstance(body, dict) or not isinstance(body.get("items"), list):
            return JSONResponse(status_code=HTTPStatus.BAD_REQUEST.value, content={"error": "request body must include items[]"})
        max_items = max(1, _env_int("VLLM_DS4_KV_PREFETCH_MANY_MAX_ITEMS", 32))
        items = body["items"][:max_items]
        results: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                results.append({"index": index, "status": "failed", "error": "item must be an object"})
                continue
            merged = dict(body)
            merged.pop("items", None)
            merged.update(item)
            result = await _run_ds4_kv_prefetch_body(raw_request, merged)
            result.setdefault("index", index)
            results.append(result)
        return JSONResponse(content={"format": "ds4-vllm-kv-prefetch-many-v1", "status": "completed", "count": len(results), "results": results})
    finally:
        _PREFETCH_SEMAPHORE.release()


def _authorize_prefetch_request(raw_request: Request) -> JSONResponse | None:
    expected_token = os.environ.get("VLLM_DS4_KV_PREFETCH_TOKEN")
    require_token = _env_bool("VLLM_DS4_KV_PREFETCH_REQUIRE_TOKEN", True)
    if require_token and not expected_token:
        return JSONResponse(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value, content={"error": "VLLM_DS4_KV_PREFETCH_TOKEN is required when DS4 KV prefetch API is enabled"})
    if expected_token:
        supplied = raw_request.headers.get("x-ds4-kv-prefetch-token") or raw_request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if supplied != expected_token:
            return JSONResponse(status_code=HTTPStatus.UNAUTHORIZED.value, content={"error": "invalid DS4 KV prefetch token"})
    return None


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def _default_served_model(raw_request: Request) -> str:
    args = getattr(raw_request.app.state, "args", None)
    served = getattr(args, "served_model_name", None)
    if isinstance(served, list) and served:
        return str(served[0])
    if isinstance(served, str) and served:
        return served
    model = getattr(args, "model", None)
    return str(model or "model")
