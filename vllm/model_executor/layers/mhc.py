# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

# this import will also register the custom ops
import vllm.model_executor.kernels.mhc as mhc_kernels
import vllm.envs as envs
from vllm.model_executor.custom_op import CustomOp


def _maybe_check_hc_head_ref(
    got: torch.Tensor,
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> None:
    if not envs.VLLM_DS4_DSV4_HC_HEAD_REF_CHECK:
        return
    if torch.compiler.is_compiling():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    num_tokens, hc_mult, hidden_size = residual.shape
    max_tokens = envs.VLLM_DS4_DSV4_HC_HEAD_REF_MAX_TOKENS
    if max_tokens <= 0 or num_tokens > max_tokens:
        return
    with torch.no_grad():
        x = residual.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(x, hc_fn.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(
            (sqrsum / float(hc_mult * hidden_size)) + rms_norm_eps)
        pre_mix = (
            torch.sigmoid(mixes * hc_scale[0] + hc_base.view(1, hc_mult))
            + hc_eps
        )
        ref = torch.sum(
            pre_mix.unsqueeze(-1) * residual.to(torch.float32), dim=1
        ).to(got.dtype)
        diff = (got.detach().to(torch.float32) - ref.to(torch.float32)).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        finite = bool(
            torch.isfinite(got).all().item()
            and torch.isfinite(ref).all().item()
        )
    if (
        not finite
        or max_abs > envs.VLLM_DS4_DSV4_HC_HEAD_REF_ATOL
        or mean_abs > envs.VLLM_DS4_DSV4_HC_HEAD_REF_MEAN_ATOL
    ):
        raise RuntimeError(
            "DS4 DSV4 hc_head reference check failed: "
            f"tokens={num_tokens} hc_mult={hc_mult} hidden_size={hidden_size} "
            f"finite={finite} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
            f"max_allowed={envs.VLLM_DS4_DSV4_HC_HEAD_REF_ATOL:.6g} "
            f"mean_allowed={envs.VLLM_DS4_DSV4_HC_HEAD_REF_MEAN_ATOL:.6g}"
        )


# --8<-- [start:mhc_pre]
@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Reenable aiter after we are at the aiter
        # version that has this bugfix
        # https://github.com/ROCm/aiter/commit/b639cb63bcac4672dce33a731fad042a65cb3649
        # It has accuracy problem at large number of tokens.
        # hidden_size = residual.shape[-1]
        # if hidden_size % 256 == 0:
        #     return torch.ops.vllm.mhc_pre_aiter(
        #         residual,
        #         fn,
        #         hc_scale,
        #         hc_base,
        #         rms_eps,
        #         hc_pre_eps,
        #         hc_sinkhorn_eps,
        #         hc_post_mult_value,
        #         sinkhorn_repeat,
        #     )
        # else:
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of mhc_pre is not available")


# --8<-- [start:mhc_post]
@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: Reenable aiter after we are at the aiter
        # version that has this bugfix
        # https://github.com/ROCm/aiter/commit/b639cb63bcac4672dce33a731fad042a65cb3649
        # It has accuracy problem at large number of tokens.
        # hidden_size = residual.shape[-1]
        # if hidden_size % 256 == 0:
        #     return torch.ops.vllm.mhc_post_aiter(
        #         x,
        #         residual,
        #         post_layer_mix,
        #         comb_res_mix,
        #     )
        # else:
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of mhc_post is not available")


# --8<-- [start:hc_head]
@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        backend = envs.VLLM_DS4_DSV4_HC_HEAD_BACKEND
        if backend in ("", "tilelang"):
            torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )
        elif backend == "triton-debug":
            torch.ops.vllm.hc_head_triton(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )
        else:
            raise RuntimeError(
                "Unsupported VLLM_DS4_DSV4_HC_HEAD_BACKEND="
                f"{backend!r}. Use 'tilelang' for production or "
                "'triton-debug' for a correctness isolation run."
            )
        _maybe_check_hc_head_ref(
            out,
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        torch.ops.vllm.hc_head_triton(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            rms_norm_eps,
            hc_eps,
            hc_mult,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of hc_head is not available")


# --8<-- [start:mhc_fused_post_pre]
@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_hip(self, *args, **kwargs):
        raise NotImplementedError(
            "Hip implementation of mhc_fused_post_pre is not available"
        )

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available"
        )
