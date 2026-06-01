# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x fallback implementations for DeepGEMM-only interfaces."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning_once("Ignoring invalid integer %s=%r", name, value)
        return default


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


_SM120_MQA_LOGITS_MAX_SCORE_BYTES = 64 * 1024 * 1024
_SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES = 512 * 1024 * 1024
_SM120_MQA_TOPK_CHUNK_SIZE = 8192
_SM120_PAGED_MQA_TOPK_CHUNK_SIZE = 8192


def _top_k_per_row_prefill_op():
    try:
        from vllm import _custom_ops as _custom_ops  # noqa: F401

        return torch.ops._C.top_k_per_row_prefill
    except (AttributeError, ImportError, RuntimeError):
        return None



@triton.jit
def _ds4_gather_values_i32_indices_kernel(
    values_ptr,
    indices_ptr,
    out_values_ptr,
    num_rows: tl.constexpr,
    source_cols: tl.constexpr,
    topk_cols: tl.constexpr,
    values_stride_row: tl.constexpr,
    values_stride_col: tl.constexpr,
    indices_stride_row: tl.constexpr,
    indices_stride_col: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = (row < num_rows) & (offs < topk_cols)
    idx = tl.load(
        indices_ptr + row * indices_stride_row + offs * indices_stride_col,
        mask=mask,
        other=0,
    )
    valid_idx = (idx >= 0) & (idx < source_cols)
    vals = tl.load(
        values_ptr + row * values_stride_row + idx * values_stride_col,
        mask=mask & valid_idx,
        other=float("-inf"),
    )
    tl.store(
        out_values_ptr + row * out_stride_row + offs * out_stride_col,
        vals,
        mask=mask,
    )


@triton.jit
def _ds4_gather_values_and_indices_i32_kernel(
    candidate_values_ptr,
    candidate_indices_ptr,
    selected_ptr,
    out_values_ptr,
    out_indices_ptr,
    num_rows: tl.constexpr,
    candidate_cols: tl.constexpr,
    topk_cols: tl.constexpr,
    candidate_values_stride_row: tl.constexpr,
    candidate_values_stride_col: tl.constexpr,
    candidate_indices_stride_row: tl.constexpr,
    candidate_indices_stride_col: tl.constexpr,
    selected_stride_row: tl.constexpr,
    selected_stride_col: tl.constexpr,
    out_values_stride_row: tl.constexpr,
    out_values_stride_col: tl.constexpr,
    out_indices_stride_row: tl.constexpr,
    out_indices_stride_col: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = (row < num_rows) & (offs < topk_cols)
    selected = tl.load(
        selected_ptr + row * selected_stride_row + offs * selected_stride_col,
        mask=mask,
        other=0,
    )
    valid_selected = (selected >= 0) & (selected < candidate_cols)
    vals = tl.load(
        candidate_values_ptr
        + row * candidate_values_stride_row
        + selected * candidate_values_stride_col,
        mask=mask & valid_selected,
        other=float("-inf"),
    )
    idx = tl.load(
        candidate_indices_ptr
        + row * candidate_indices_stride_row
        + selected * candidate_indices_stride_col,
        mask=mask & valid_selected,
        other=-1,
    )
    tl.store(
        out_values_ptr + row * out_values_stride_row + offs * out_values_stride_col,
        vals,
        mask=mask,
    )
    tl.store(
        out_indices_ptr + row * out_indices_stride_row + offs * out_indices_stride_col,
        idx,
        mask=mask,
    )


def _ceil_power_of_two_at_least_one(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def _cuda_topk_per_row_enabled() -> bool:
    return _env_flag("VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT", True)


def _topk_per_row_cuda(
    values: torch.Tensor,
    topk: int,
    valid_cols: int,
    out_indices: torch.Tensor,
) -> bool:
    if not _cuda_topk_per_row_enabled():
        return False
    op = _top_k_per_row_prefill_op()
    if op is None:
        if _env_flag("VLLM_DS4_STRICT_NATIVE_FP4", False):
            raise RuntimeError(
                "DS4 requested CUDA top-k selection but _C.top_k_per_row_prefill "
                "is not available. Rebuild the vLLM CUDA extension or disable "
                "VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT only for diagnostics."
            )
        return False
    num_rows = values.shape[0]
    row_starts = torch.zeros(num_rows, device=values.device, dtype=torch.int32)
    row_ends = torch.empty(num_rows, device=values.device, dtype=torch.int32)
    row_ends.fill_(valid_cols)
    op(
        values,
        row_starts,
        row_ends,
        out_indices,
        num_rows,
        values.stride(0),
        values.stride(1),
        topk,
    )
    return True


def _gather_values_i32_indices(
    values: torch.Tensor,
    indices: torch.Tensor,
    out_values: torch.Tensor,
) -> None:
    num_rows, topk_cols = indices.shape
    if num_rows == 0 or topk_cols == 0:
        return
    block_k = min(4096, _ceil_power_of_two_at_least_one(topk_cols))
    _ds4_gather_values_i32_indices_kernel[(num_rows,)](
        values,
        indices,
        out_values,
        num_rows,
        values.shape[1],
        topk_cols,
        values.stride(0),
        values.stride(1),
        indices.stride(0),
        indices.stride(1),
        out_values.stride(0),
        out_values.stride(1),
        BLOCK_K=block_k,
        num_warps=8,
    )


def _gather_values_and_indices_i32(
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    selected: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    num_rows, topk_cols = selected.shape
    if num_rows == 0 or topk_cols == 0:
        return
    block_k = min(4096, _ceil_power_of_two_at_least_one(topk_cols))
    _ds4_gather_values_and_indices_i32_kernel[(num_rows,)](
        candidate_values,
        candidate_indices,
        selected,
        out_values,
        out_indices,
        num_rows,
        candidate_values.shape[1],
        topk_cols,
        candidate_values.stride(0),
        candidate_values.stride(1),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        selected.stride(0),
        selected.stride(1),
        out_values.stride(0),
        out_values.stride(1),
        out_indices.stride(0),
        out_indices.stride(1),
        BLOCK_K=block_k,
        num_warps=8,
    )


def _fp8_mqa_logits_head_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
) -> int:
    # The SM120 torch path is used on long prefill paths where materializing
    # [head_chunk, M, N] scores can otherwise allocate multiple GiB. Keep the
    # transient score tensor bounded, while still using larger head chunks for
    # short prompts where they are faster.
    score_elems_per_head = max(1, seq_len * seq_len_kv)
    max_heads = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_head * 4)
    return max(1, min(8, num_heads, max_heads))


def _fp8_mqa_logits_k_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    head_chunk_size: int,
) -> int:
    score_elems_per_key = max(1, seq_len * head_chunk_size)
    max_keys = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_key * 4)
    return max(1, min(seq_len_kv, max_keys))


def _fp8_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA logits torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q_values.device, dtype=torch.float32
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)

    for head_start in range(0, num_heads, head_chunk_size):
        head_end = min(head_start + head_chunk_size, num_heads)
        q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
        q_chunk = q_chunk.transpose(0, 1).contiguous()
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        k_chunk_size = _fp8_mqa_logits_k_chunk_size(
            seq_len, seq_len_kv, head_end - head_start
        )
        for k_start in range(0, seq_len_kv, k_chunk_size):
            k_end = min(k_start + k_chunk_size, seq_len_kv)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            logits[:, k_start:k_end].add_(
                scores[0] if scores.shape[0] == 1 else scores.sum(dim=0)
            )

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def _fp8_mqa_logits_topk_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA top-k torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    if out is None:
        out = torch.empty(
            (seq_len, topk_tokens), device=q_values.device, dtype=torch.int32
        )
    else:
        assert out.shape == (seq_len, topk_tokens)
        assert out.dtype == torch.int32
    out.fill_(-1)

    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)
    k_chunk_size = _fp8_mqa_logits_k_chunk_size(seq_len, seq_len_kv, head_chunk_size)
    max_chunk_topk = min(topk_tokens, k_chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )
    selected_i32 = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int32,
    )

    for k_start in range(0, seq_len_kv, k_chunk_size):
        k_end = min(k_start + k_chunk_size, seq_len_kv)
        chunk_logits = torch.zeros(
            (seq_len, k_end - k_start),
            device=q_values.device,
            dtype=torch.float32,
        )
        for head_start in range(0, num_heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, num_heads)
            q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
            q_chunk = q_chunk.transpose(0, 1).contiguous()
            head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            chunk_logits.add_(scores[0] if scores.shape[0] == 1 else scores.sum(dim=0))

        offsets = torch.arange(k_start, k_end, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        chunk_logits.masked_fill_(~valid, float("-inf"))

        chunk_topk = min(topk_tokens, k_end - k_start)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return out


def _fp8_mqa_logits_topk_triton(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    q_values, q_scale = q
    if not _env_flag("VLLM_DS4_SM12X_MQA_TOPK_TRITON", True):
        logger.warning_once(
            "DS4 SM12x dense-MQA top-k Triton path disabled by "
            "VLLM_DS4_SM12X_MQA_TOPK_TRITON=0")
        return False
    k_values, _ = kv
    if not (q_scale is None and q_values.dim() == 3 and k_values.dim() == 2):
        return False

    return _fp8_mqa_logits_topk_triton_chunked(
        q_values,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
    )


def _fp8_mqa_logits_topk_triton_chunked(
    q_values: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    k_values, k_scales = kv
    seq_len, _num_heads, _ = q_values.shape
    seq_len_kv = k_values.shape[0]
    topk_tokens = out.shape[1]
    out.fill_(-1)
    if seq_len == 0 or seq_len_kv == 0 or topk_tokens == 0:
        return True

    max_logits_bytes = _env_int(
        "VLLM_DS4_SM12X_MQA_TOPK_MAX_LOGITS_BYTES",
        _SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES,
    )
    env_chunk_size = _env_int(
        "VLLM_DS4_SM12X_MQA_TOPK_CHUNK_SIZE",
        _SM120_MQA_TOPK_CHUNK_SIZE,
    )
    max_chunk_by_bytes = max(1, max_logits_bytes // (seq_len * torch.float32.itemsize))
    chunk_size = max(1, min(seq_len_kv, env_chunk_size, max_chunk_by_bytes))
    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    next_best_values = torch.empty_like(best_values)
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )
    selected_i32 = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int32,
    )
    chunk_logits_buf = torch.empty(
        (seq_len, chunk_size),
        device=q_values.device,
        dtype=torch.float32,
    )

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_mqa_logits_triton,
    )

    for k_start in range(0, seq_len_kv, chunk_size):
        k_end = min(k_start + chunk_size, seq_len_kv)
        rel_ks = torch.clamp(cu_seqlen_ks - k_start, min=0, max=k_end - k_start)
        rel_ke = torch.clamp(cu_seqlen_ke - k_start, min=0, max=k_end - k_start)
        logits = fp8_mqa_logits_triton(
            q_values,
            (k_values[k_start:k_end], k_scales[k_start:k_end]),
            weights,
            rel_ks,
            rel_ke,
            logits_out=chunk_logits_buf[:, : k_end - k_start],
        )
        chunk_topk = min(topk_tokens, logits.shape[1])
        if chunk_topk == 0:
            continue
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        if _topk_per_row_cuda(logits, chunk_topk, logits.shape[1], chunk_indices_out):
            _gather_values_i32_indices(logits, chunk_indices_out, chunk_values)
        else:
            chunk_indices = chunk_indices_buf[:, :chunk_topk]
            torch.topk(logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
            chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        if _topk_per_row_cuda(
            candidate_values_view,
            topk_tokens,
            candidate_cols,
            selected_i32[:, :topk_tokens],
        ):
            _gather_values_and_indices_i32(
                candidate_values_view,
                candidate_indices_view,
                selected_i32[:, :topk_tokens],
                next_best_values,
                out,
            )
        else:
            torch.topk(
                candidate_values_view,
                topk_tokens,
                dim=1,
                out=(next_best_values, selected),
            )
            torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)
    return True


def fp8_fp4_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 MQA top-k indices without materializing full logits."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q[1] is None
    ):
        return False
    if _fp8_mqa_logits_topk_triton(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True
    if not _env_flag("VLLM_DS4_ALLOW_SM12X_MQA_TOPK_TORCH_FALLBACK", False):
        raise RuntimeError(
            "DS4 SM12x dense-MQA top-k requires the bounded Triton path. "
            "The Torch fallback is forbidden by default because it is slow "
            "and has produced CUDA illegal-address failures on GB10. "
            "Set VLLM_DS4_SM12X_MQA_TOPK_TRITON=1 or explicitly opt in with "
            "VLLM_DS4_ALLOW_SM12X_MQA_TOPK_TORCH_FALLBACK=1 for diagnostics."
        )
    _fp8_mqa_logits_topk_torch(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices.shape[1],
        out=topk_indices,
    )
    return True


def _fp8_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if clean_logits and q_scale is None and q_values.dim() == 3 and kv[0].dim() == 2:
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_mqa_logits_triton,
        )

        return fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    return _fp8_mqa_logits_torch(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def _fp8_paged_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 paged MQA torch path only supports FP8 Q")

    batch_size, next_n, num_heads, head_dim = q_values.shape
    head_dim_with_scale = kv_cache.shape[-1]
    assert head_dim_with_scale > head_dim
    assert weights.shape == (batch_size * next_n, num_heads)
    assert context_lens.shape == (batch_size, next_n)

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        _view_packed_fp8_paged_mqa_kv_cache,
    )

    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_kv, _, _ = kv_values.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )

    q_f32 = q_values.float()
    score_bytes = _SM120_MQA_LOGITS_MAX_SCORE_BYTES
    max_tokens_per_chunk = max(1, score_bytes // max(1, num_heads * 4))
    token_offsets_cache: dict[int, torch.Tensor] = {}

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            if context_len <= 0:
                continue

            q_row = q_f32[batch_idx, next_idx]
            row_weights = weights[row]
            for token_start in range(0, context_len, max_tokens_per_chunk):
                token_end = min(context_len, token_start + max_tokens_per_chunk)
                chunk_len = token_end - token_start
                token_offsets = token_offsets_cache.get(chunk_len)
                if token_offsets is None or token_offsets.device != q_values.device:
                    token_offsets = torch.arange(
                        chunk_len, device=q_values.device, dtype=torch.long
                    )
                    token_offsets_cache[chunk_len] = token_offsets
                token_ids = token_start + token_offsets
                logical_blocks = token_ids // block_kv
                token_in_block = token_ids - logical_blocks * block_kv
                physical_blocks = block_tables[batch_idx, logical_blocks]
                kv_chunk = kv_values[physical_blocks, token_in_block, 0].float()
                scale_chunk = kv_scales[physical_blocks, token_in_block, 0].squeeze(-1)
                kv_chunk.mul_(scale_chunk[:, None])
                scores = torch.matmul(q_row, kv_chunk.T)
                scores.relu_()
                scores.mul_(row_weights[:, None])
                logits[row, token_start:token_end] = scores.sum(dim=0)

    return logits


def _fp8_paged_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if (
        q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_paged_mqa_logits_triton,
        )

        return fp8_paged_mqa_logits_triton(
            q_values, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    logger.warning_once(
        "SM12x paged-MQA falling back to the torch reference path "
        "(q_scale=%s, q.dim=%s, kv_cache.dtype=%s, kv_cache.shape[-1]=%s, "
        "q_values.shape[-1]=%s). This path is intended for correctness checks "
        "and is not graph-compatible; expect a large per-step latency.",
        "set" if q_scale is not None else "None",
        q_values.dim(),
        kv_cache.dtype,
        kv_cache.shape[-1] if kv_cache.dim() else None,
        q_values.shape[-1],
    )
    return _fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_fp4_paged_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 paged MQA top-k indices without full logits."""
    q_values, q_scale = q
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        return False

    num_rows = q_values.shape[0] * q_values.shape[1]
    topk_tokens = topk_indices.shape[1]
    assert topk_indices.shape == (num_rows, topk_tokens)
    assert topk_indices.dtype == torch.int32
    topk_indices.fill_(-1)
    if num_rows == 0 or topk_tokens == 0 or max_model_len == 0:
        return True

    best_values = torch.full(
        (num_rows, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_size = max(1, _env_int(
        "VLLM_DS4_SM12X_PAGED_MQA_TOPK_CHUNK_SIZE",
        _SM120_PAGED_MQA_TOPK_CHUNK_SIZE,
    ))
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (num_rows, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )
    selected_i32 = torch.empty(
        (num_rows, topk_tokens),
        device=q_values.device,
        dtype=torch.int32,
    )
    chunk_logits_buf = torch.empty(
        (num_rows, chunk_size),
        device=q_values.device,
        dtype=torch.float32,
    )

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_paged_mqa_logits_triton,
    )

    for token_start in range(0, max_model_len, chunk_size):
        token_count = min(chunk_size, max_model_len - token_start)
        chunk_logits = fp8_paged_mqa_logits_triton(
            q_values,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
            logits_out=chunk_logits_buf[:, :token_count],
        )
        chunk_topk = min(topk_tokens, token_count)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        if _topk_per_row_cuda(
            chunk_logits,
            chunk_topk,
            token_count,
            chunk_indices_out,
        ):
            _gather_values_i32_indices(chunk_logits, chunk_indices_out, chunk_values)
        else:
            chunk_indices = chunk_indices_buf[:, :chunk_topk]
            torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
            chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(token_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(topk_indices)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        if _topk_per_row_cuda(
            candidate_values_view,
            topk_tokens,
            candidate_cols,
            selected_i32[:, :topk_tokens],
        ):
            _gather_values_and_indices_i32(
                candidate_values_view,
                candidate_indices_view,
                selected_i32[:, :topk_tokens],
                next_best_values,
                topk_indices,
            )
        else:
            torch.topk(
                candidate_values_view,
                topk_tokens,
                dim=1,
                out=(next_best_values, selected),
            )
            torch.gather(candidate_indices_view, 1, selected, out=topk_indices)
        best_values, next_best_values = next_best_values, best_values
        topk_indices.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def _tf32_hc_prenorm_gemm_torch(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """Portable SM12x HyperConnection prenorm GEMM fallback.

    DeepGEMM's split ABI only requires that downstream consumers recover the
    full result by summing over the split dimension. Keep the implementation
    simple by writing the full product to split zero and clearing the rest.
    """
    del num_split
    product = x.float() @ fn.float().T
    norm = x.float().square().sum(dim=-1)

    if out.dim() == 3:
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(product)
        sqrsum[0].copy_(norm)
    else:
        out.copy_(product)
        sqrsum.copy_(norm)
    return out


def _tf32_hc_prenorm_gemm_sm12x(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    if out.dim() == 3 and sqrsum.dim() == 2:
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            tf32_hc_prenorm_gemm_triton,
        )

        tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split)
        return out

    return _tf32_hc_prenorm_gemm_torch(x, fn, out, sqrsum, num_split)
