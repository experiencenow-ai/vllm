# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    dequantize_global_slots_k_cache,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    is_triton_sparse_mla_enabled,
    triton_sparse_mla_matmul_decode_enabled,
    triton_sparse_mla_query_chunk_size,
    triton_sparse_mla_splitkv_decode_enabled,
    triton_sparse_mla_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead,
    accumulate_indexed_sparse_mla_attention_chunk,
    build_combined_sparse_mla_decode_valid_mask,
    choose_sparse_mla_splitkv_splits,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    matmul_sparse_mla_attention_with_sink,
    sparse_mla_decode_head_block_size,
    splitkv_sparse_mla_attention_with_sink,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.nvidia.ops.attention import (
        DeepseekV4MLAAttention,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


logger = init_logger(__name__)


@torch.compiler.disable
def _ds4_sparse_mla_tensor_stats(tensor: torch.Tensor) -> str:
    if tensor.numel() == 0:
        return "numel=0"
    sample = tensor.detach()
    if sample.dtype == torch.bool:
        sample = sample.to(torch.int32)
    if not sample.is_floating_point():
        sample = sample.to(torch.float32)
    else:
        sample = sample.float()
    finite = torch.isfinite(sample)
    finite_count = int(finite.sum().item())
    if finite_count == 0:
        return f"numel={tensor.numel()} finite=0"
    valid = sample[finite]
    return (
        f"numel={tensor.numel()} finite={finite_count} "
        f"min={float(valid.min().item()):.6g} "
        f"max={float(valid.max().item()):.6g} "
        f"mean={float(valid.mean().item()):.6g} "
        f"absmax={float(valid.abs().max().item()):.6g}"
    )


@torch.compiler.disable
def _ds4_validate_indexed_sparse_mla_inputs(
    *,
    layer_prefix: str,
    stage: str,
    indices: torch.Tensor,
    lens: torch.Tensor,
    max_index: int,
) -> None:
    if not (envs.VLLM_DS4_DSV4_SPARSE_MLA_VALIDATE or envs.VLLM_DS4_DSV4_SPARSE_MLA_TRACE):
        return
    if indices.dim() == 3 and indices.shape[1] == 1:
        indices = indices[:, 0, :]
    elif indices.dim() != 2:
        raise RuntimeError(
            f"DS4 sparse MLA {stage} expected 2D indices or a singleton "
            f"head axis for {layer_prefix}, "
            f"got shape={tuple(indices.shape)}"
        )
    if lens.dim() != 1 or lens.shape[0] != indices.shape[0]:
        raise RuntimeError(
            f"DS4 sparse MLA {stage} lens mismatch for {layer_prefix}: "
            f"indices_shape={tuple(indices.shape)} lens_shape={tuple(lens.shape)}"
        )
    width = indices.shape[1]
    if lens.numel() > 0:
        lens_min = int(lens.min().item())
        lens_max = int(lens.max().item())
    else:
        lens_min = 0
        lens_max = 0
    if lens_min < 0 or lens_max > width:
        raise RuntimeError(
            f"DS4 sparse MLA {stage} invalid lens for {layer_prefix}: "
            f"lens_min={lens_min} lens_max={lens_max} width={width}"
        )
    if indices.numel() > 0:
        offsets = torch.arange(width, device=indices.device).view(1, width)
        valid_mask = offsets < lens.view(-1, 1)
        valid_indices = indices[valid_mask]
    else:
        valid_indices = indices.reshape(-1)
    if valid_indices.numel() > 0:
        idx_min = int(valid_indices.min().item())
        idx_max = int(valid_indices.max().item())
        if idx_min < 0 or idx_max >= max_index:
            bad_count = int(((valid_indices < 0) | (valid_indices >= max_index)).sum().item())
            raise RuntimeError(
                f"DS4 sparse MLA {stage} out-of-range indices for {layer_prefix}: "
                f"bad_count={bad_count} idx_min={idx_min} idx_max={idx_max} "
                f"max_index={max_index} width={width} lens_min={lens_min} "
                f"lens_max={lens_max}"
            )
    else:
        idx_min = 0
        idx_max = -1
    if envs.VLLM_DS4_DSV4_SPARSE_MLA_TRACE:
        logger.info(
            "DS4 sparse MLA %s %s rows=%d width=%d lens_min=%d lens_max=%d "
            "idx_min=%d idx_max=%d max_index=%d",
            layer_prefix,
            stage,
            indices.shape[0],
            width,
            lens_min,
            lens_max,
            idx_min,
            idx_max,
            max_index,
        )


@torch.compiler.disable
def _ds4_check_sparse_mla_output(
    *,
    layer_prefix: str,
    stage: str,
    output: torch.Tensor,
    num_heads: int,
) -> None:
    if not (envs.VLLM_DS4_DSV4_SPARSE_MLA_VALIDATE or envs.VLLM_DS4_DSV4_SPARSE_MLA_TRACE):
        return
    active = output[:, :num_heads]
    if active.numel() > 0 and not bool(torch.isfinite(active).all().item()):
        raise RuntimeError(
            f"DS4 sparse MLA {stage} produced non-finite output for {layer_prefix}: "
            f"{_ds4_sparse_mla_tensor_stats(active)}"
        )
    if envs.VLLM_DS4_DSV4_SPARSE_MLA_TRACE:
        logger.info(
            "DS4 sparse MLA %s %s output %s",
            layer_prefix,
            stage,
            _ds4_sparse_mla_tensor_stats(active),
        )


@torch.compiler.disable
def _ds4_reference_check_sparse_mla_prefill(
    *,
    layer_prefix: str,
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
    output: torch.Tensor,
    num_heads: int,
) -> None:
    if not envs.VLLM_DS4_DSV4_SPARSE_MLA_REF_CHECK:
        return
    num_tokens = min(
        q.shape[0],
        output.shape[0],
        max(0, envs.VLLM_DS4_DSV4_SPARSE_MLA_REF_MAX_TOKENS),
    )
    if num_tokens == 0:
        return
    ref = torch.zeros_like(output[:num_tokens])
    q_active = q[:num_tokens, :num_heads].float()
    sink = attn_sink[:num_heads].float()
    for token_idx in range(num_tokens):
        lens = int(combined_lens[token_idx].item())
        if lens <= 0:
            continue
        slot_ids = combined_indices[token_idx, :lens]
        slot_ids = slot_ids[(slot_ids >= 0) & (slot_ids < kv_flat.shape[0])]
        if slot_ids.numel() == 0:
            continue
        kv = kv_flat.index_select(0, slot_ids.long()).float()
        scores = torch.matmul(q_active[token_idx], kv.t()) * scale
        scores_with_sink = torch.cat((scores, sink.view(num_heads, 1)), dim=1)
        probs = torch.softmax(scores_with_sink, dim=1)[:, : kv.shape[0]]
        ref[token_idx, :num_heads] = torch.matmul(probs, kv).to(output.dtype)
    diff = (output[:num_tokens, :num_heads].float() - ref[:, :num_heads].float()).abs()
    max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    if max_diff > envs.VLLM_DS4_DSV4_SPARSE_MLA_REF_ATOL:
        raise RuntimeError(
            f"DS4 sparse MLA prefill reference mismatch for {layer_prefix}: "
            f"tokens={num_tokens} max_diff={max_diff:.6g} "
            f"atol={envs.VLLM_DS4_DSV4_SPARSE_MLA_REF_ATOL:.6g} "
            f"actual={_ds4_sparse_mla_tensor_stats(output[:num_tokens, :num_heads])} "
            f"ref={_ds4_sparse_mla_tensor_stats(ref[:, :num_heads])}"
        )
    if envs.VLLM_DS4_DSV4_SPARSE_MLA_TRACE:
        logger.info(
            "DS4 sparse MLA %s prefill reference ok tokens=%d max_diff=%.6g",
            layer_prefix,
            num_tokens,
            max_diff,
        )


class DeepseekV4SparseMLAAttentionImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    """Abstract parent for DeepseekV4 sparse MLA impls.

    V4 sparse MLA is driven by the layer (``DeepseekV4MLAAttention.forward``)
    rather than the v1 framework, so ``forward_mqa`` is overridden with a
    classmethod that takes the layer as its first argument. This Liskov-broken
    override is intentional: the grandparent's instance-method ``forward_mqa``
    is never called on V4 layers.
    """

    backend_cls: ClassVar[type[AttentionBackend]]

    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the V4 layer's
    # dummy-run path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class DeepseekV4FlashMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_name() -> str:
        return "V4_FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SparseMLAAttentionImpl"]:
        return DeepseekV4FlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512 (overrides the
        # V3.2 default of 576 from FlashMLASparseBackend).
        return [512]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)


class DeepseekV4FlashMLASparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """FlashMLA sparse MLA implementation for DeepSeek V4's custom MLA layer."""

    backend_cls = DeepseekV4FlashMLASparseBackend

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = layer.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (layer.max_model_len + layer.compress_ratio - 1)
                // layer.compress_ratio
            )
            M = N + layer.window_size + layer.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((cls.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so layer.kv_cache may be empty after profiling cleanup.
        self_kv_cache = layer.kv_cache if not swa_only else None
        swa_kv_cache = layer.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            cls._forward_prefill(
                layer=layer,
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            cls._forward_decode(
                layer=layer,
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _forward_sparse_mla_swa_decode_triton(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        _ds4_validate_indexed_sparse_mla_inputs(
            layer_prefix=layer.prefix,
            stage="swa_decode",
            indices=swa_indices,
            lens=swa_lens,
            max_index=swa_k_cache.shape[0] * swa_metadata.block_size,
        )
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if not mtp_decode:
            fp8ds_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=layer.num_heads,
            )
            if output.shape[1] > layer.num_heads:
                output[:, layer.num_heads :].zero_()
            _ds4_check_sparse_mla_output(
                layer_prefix=layer.prefix,
                stage="swa_decode_paged",
                output=output,
                num_heads=layer.num_heads,
            )
            return

        (
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads, q.shape[-1]), torch.float32),
        )
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_k_cache,
            slot_ids=swa_indices,
            lens=swa_lens,
            block_size=swa_metadata.block_size,
            scale=layer.scale,
            max_score=swa_max_score,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=head_block_size,
        )
        finish_sparse_mla_attention_with_sink(
            swa_max_score,
            swa_denom,
            swa_acc,
            layer.attn_sink,
            output=output,
        )
        if output.shape[1] > layer.num_heads:
            output[:, layer.num_heads :].zero_()
        _ds4_check_sparse_mla_output(
            layer_prefix=layer.prefix,
            stage="swa_decode_accum",
            output=output,
            num_heads=layer.num_heads,
        )

    @classmethod
    def _forward_sparse_mla_compressed_decode_triton(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata,
        output: torch.Tensor,
    ) -> None:
        if layer.compress_ratio not in (4, 128):
            raise NotImplementedError(
                "Triton sparse MLA compressed decode currently supports "
                f"compress_ratio=4 or 128, got {layer.compress_ratio}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        compressed_block_size = attn_metadata.block_size // layer.compress_ratio
        compressed_topk = topk_indices.shape[-1]
        topk_chunk_size = min(
            compressed_topk,
            triton_sparse_mla_topk_chunk_size(),
        )
        compressed_slot_ids = topk_indices[:, 0, :]
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        _ds4_validate_indexed_sparse_mla_inputs(
            layer_prefix=layer.prefix,
            stage="compressed_decode_topk",
            indices=compressed_slot_ids,
            lens=topk_lens,
            max_index=compressed_k_cache.shape[0] * compressed_block_size,
        )
        _ds4_validate_indexed_sparse_mla_inputs(
            layer_prefix=layer.prefix,
            stage="compressed_decode_swa",
            indices=swa_indices,
            lens=swa_lens,
            max_index=swa_k_cache.shape[0] * swa_metadata.block_size,
        )
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if (
            compressed_topk <= topk_chunk_size
            and triton_sparse_mla_matmul_decode_enabled()
        ):
            total_candidates = compressed_topk + max_swa_len
            num_splitkv_splits = 1
            if triton_sparse_mla_splitkv_decode_enabled():
                num_splitkv_splits = choose_sparse_mla_splitkv_splits(
                    num_tokens=num_decode_tokens,
                    num_heads=layer.num_heads,
                    num_candidates=total_candidates,
                    sm_count=num_compute_units(
                        torch.accelerator.current_device_index()
                    ),
                )
            use_splitkv = num_splitkv_splits > 1
            workspace_specs = [
                (
                    (num_decode_tokens, total_candidates, q.shape[-1]),
                    torch.bfloat16,
                ),
                ((num_decode_tokens, total_candidates), torch.bool),
            ]
            if use_splitkv:
                workspace_specs.append(
                    (
                        (
                            num_decode_tokens,
                            layer.num_heads,
                            num_splitkv_splits,
                            q.shape[-1] + 1,
                        ),
                        torch.float32,
                    )
                )
            else:
                workspace_specs.append(
                    (
                        (num_decode_tokens, layer.num_heads, total_candidates),
                        torch.bfloat16,
                    )
                )
            (
                combined_kv,
                valid_tokens,
                score_or_mid_buffer,
            ) = current_workspace_manager().get_simultaneous(*workspace_specs)
            if mtp_decode:
                dequantize_global_slots_k_cache(
                    combined_kv[:, :compressed_topk],
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                )
                dequantize_global_slots_k_cache(
                    combined_kv[:, compressed_topk:],
                    swa_k_cache,
                    swa_indices,
                    swa_metadata.block_size,
                )
            else:
                dequantize_combined_sparse_mla_decode_kv(
                    combined_kv,
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                    swa_k_cache,
                    swa_metadata.seq_lens[:num_decodes],
                    swa_lens,
                    swa_metadata.block_table[:num_decodes],
                    swa_metadata.block_size,
                )

            build_combined_sparse_mla_decode_valid_mask(
                valid_tokens,
                compressed_slot_ids,
                topk_lens,
                swa_lens,
            )
            if use_splitkv:
                splitkv_sparse_mla_attention_with_sink(
                    q=q,
                    kv=combined_kv,
                    valid_tokens=valid_tokens,
                    scale=layer.scale,
                    attn_sink=layer.attn_sink,
                    output=output,
                    mid=score_or_mid_buffer,
                    num_splits=num_splitkv_splits,
                    num_heads=layer.num_heads,
                )
                if output.shape[1] > layer.num_heads:
                    output[:, layer.num_heads :].zero_()
                _ds4_check_sparse_mla_output(
                    layer_prefix=layer.prefix,
                    stage="compressed_decode_splitkv",
                    output=output,
                    num_heads=layer.num_heads,
                )
                return

            use_dot_finish = num_decode_tokens <= 16
            matmul_sparse_mla_attention_with_sink(
                q=q,
                kv=combined_kv,
                valid_tokens=valid_tokens,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                num_heads=layer.num_heads,
                score_buffer=score_or_mid_buffer,
                value_block_size=512 if use_dot_finish else 256,
                candidate_block_size=128 if use_dot_finish else None,
            )
            _ds4_check_sparse_mla_output(
                layer_prefix=layer.prefix,
                stage="compressed_decode_matmul",
                output=output,
                num_heads=layer.num_heads,
            )
            return

        if not mtp_decode and compressed_topk <= topk_chunk_size:
            fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                compressed_k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids,
                topk_lens=topk_lens,
                compressed_block_size=compressed_block_size,
                swa_k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                swa_block_size=swa_metadata.block_size,
                num_compressed_candidates=compressed_topk,
                num_swa_candidates=max_swa_len,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=layer.num_heads,
            )
            if output.shape[1] > layer.num_heads:
                output[:, layer.num_heads :].zero_()
            _ds4_check_sparse_mla_output(
                layer_prefix=layer.prefix,
                stage="compressed_decode_global_paged",
                output=output,
                num_heads=layer.num_heads,
            )
            return

        (
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads, q.shape[-1]), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads), torch.float32),
            ((num_decode_tokens, layer.num_heads, q.shape[-1]), torch.float32),
        )
        comp_max_score.fill_(float("-inf"))
        comp_denom.zero_()
        comp_acc.zero_()
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()

        for chunk_start in range(0, compressed_topk, topk_chunk_size):
            chunk_end = min(chunk_start + topk_chunk_size, compressed_topk)
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids[:, chunk_start:chunk_end],
                lens=topk_lens,
                block_size=compressed_block_size,
                candidate_offset=chunk_start,
                scale=layer.scale,
                max_score=comp_max_score,
                denom=comp_denom,
                acc=comp_acc,
                head_block_size=head_block_size,
            )
        if mtp_decode:
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=swa_k_cache,
                slot_ids=swa_indices,
                lens=swa_lens,
                block_size=swa_metadata.block_size,
                scale=layer.scale,
                max_score=swa_max_score,
                denom=swa_denom,
                acc=swa_acc,
                head_block_size=head_block_size,
            )
        else:
            accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=layer.scale,
                max_score=swa_max_score,
                denom=swa_denom,
                acc=swa_acc,
                head_block_size=head_block_size,
            )
        finish_two_sparse_mla_attention_states_with_sink(
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
            layer.attn_sink,
            output=output,
        )
        if output.shape[1] > layer.num_heads:
            output[:, layer.num_heads :].zero_()
        _ds4_check_sparse_mla_output(
            layer_prefix=layer.prefix,
            stage="compressed_decode_accum",
            output=output,
            num_heads=layer.num_heads,
        )

    @classmethod
    def _forward_sparse_mla_prefill_triton(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        kv_flat = kv.reshape(-1, q.shape[-1])
        _ds4_validate_indexed_sparse_mla_inputs(
            layer_prefix=layer.prefix,
            stage="prefill",
            indices=combined_indices,
            lens=combined_lens,
            max_index=kv_flat.shape[0],
        )
        topk_chunk_size = min(
            combined_indices.shape[-1],
            triton_sparse_mla_topk_chunk_size(),
        )
        query_chunk_size = min(
            q.shape[0],
            triton_sparse_mla_query_chunk_size(),
        )
        (
            max_score_buffer,
            denom_buffer,
            output_buffer,
        ) = current_workspace_manager().get_simultaneous(
            ((query_chunk_size, layer.num_heads), torch.float32),
            ((query_chunk_size, layer.num_heads), torch.float32),
            ((query_chunk_size, layer.num_heads, q.shape[-1]), torch.float32),
        )

        for token_start in range(0, q.shape[0], query_chunk_size):
            token_end = min(token_start + query_chunk_size, q.shape[0])
            q_chunk = q[token_start:token_end]
            indices_chunk_full = combined_indices[token_start:token_end]
            lens_chunk = combined_lens[token_start:token_end]
            num_tokens = token_end - token_start
            max_score = max_score_buffer[:num_tokens]
            denom = denom_buffer[:num_tokens]
            subset_acc = output_buffer[:num_tokens]
            max_score.fill_(float("-inf"))
            denom.zero_()
            subset_acc.zero_()

            for index_start in range(0, combined_indices.shape[-1], topk_chunk_size):
                index_end = min(
                    index_start + topk_chunk_size,
                    combined_indices.shape[-1],
                )
                accumulate_indexed_sparse_mla_attention_chunk(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full[:, index_start:index_end],
                    lens=lens_chunk,
                    candidate_offset=index_start,
                    scale=layer.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                )

            finish_sparse_mla_attention_with_sink(
                max_score,
                denom,
                subset_acc,
                layer.attn_sink,
                output=output[token_start:token_end],
            )
            if output.shape[1] > layer.num_heads:
                output[token_start:token_end, layer.num_heads :].zero_()
            _ds4_check_sparse_mla_output(
                layer_prefix=layer.prefix,
                stage="prefill",
                output=output[token_start:token_end],
                num_heads=layer.num_heads,
            )
        _ds4_reference_check_sparse_mla_prefill(
            layer_prefix=layer.prefix,
            q=q,
            kv_flat=kv_flat,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            attn_sink=layer.attn_sink,
            scale=layer.scale,
            output=output,
            num_heads=layer.num_heads,
        )

    @classmethod
    def _forward_decode(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // layer.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert layer.topk_indices_buffer is not None
                local_topk_indices = layer.topk_indices_buffer[:num_decode_tokens]
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                    global_topk_indices=local_topk_indices,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to layer.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        compressed_k_cache = kv_cache
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        if is_triton_sparse_mla_enabled(q.device):
            if swa_only:
                cls._forward_sparse_mla_swa_decode_triton(
                    layer=layer,
                    q=q,
                    swa_k_cache=layer.swa_cache_layer.kv_cache,
                    swa_metadata=swa_metadata,
                    output=output,
                )
                return
            if layer.compress_ratio in (4, 128):
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                cls._forward_sparse_mla_compressed_decode_triton(
                    layer=layer,
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=layer.swa_cache_layer.kv_cache,
                    topk_indices=topk_indices,
                    topk_lens=topk_lens,
                    swa_metadata=swa_metadata,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                return

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if layer.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif layer.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif layer.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={layer.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={layer.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=layer.scale,
            attn_sink=layer.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    @classmethod
    def _forward_prefill(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (layer.max_model_len + layer.compress_ratio - 1) // layer.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert layer.topk_indices_buffer is not None
            topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + layer.window_size + layer.max_num_batched_tokens
        chunk_size_const = cls.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // layer.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // layer.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                layer.window_size,
                layer.compress_ratio,
                top_k,
                M,
                N,
            )
            if is_triton_sparse_mla_enabled(q.device):
                cls._forward_sparse_mla_prefill_triton(
                    layer=layer,
                    q=q[query_start:query_end],
                    kv=kv[:chunk_size],
                    combined_indices=combined_indices,
                    combined_lens=combined_lens,
                    output=output[query_start:query_end],
                )
                continue
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=layer.scale,
                attn_sink=layer.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )
