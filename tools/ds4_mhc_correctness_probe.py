#!/usr/bin/env python3
"""Compare DS4 mHC kernels against torch references on CUDA."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Metric:
    name: str
    max_abs: float
    mean_abs: float
    ref_absmax: float
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "ref_absmax": self.ref_absmax,
            "finite": self.finite,
        }


def _parse_tokens(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [1]


def _metric(name: str, got, ref) -> Metric:
    import torch

    got_f = got.detach().to(torch.float32)
    ref_f = ref.detach().to(torch.float32)
    diff = (got_f - ref_f).abs()
    finite = bool(torch.isfinite(got_f).all().item() and torch.isfinite(ref_f).all().item())
    return Metric(
        name=name,
        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
        ref_absmax=float(ref_f.abs().max().item()) if ref_f.numel() else 0.0,
        finite=finite,
    )


def _hc_head_torch(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps: float,
    hc_eps: float,
):
    import torch

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * hc_scale[0] + hc_base.view(1, hc_mult)) + hc_eps
    out = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1)
    return out.to(torch.bfloat16).view(*residual.shape[:-2], hidden_size)


def _rms_norm_torch(x, weight, eps: float):
    import torch

    x_f = x.to(torch.float32)
    w_f = weight.to(torch.float32)
    scale = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f * scale * w_f).to(torch.bfloat16)


def _run_case(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    seed: int,
    include_triton_head: bool,
) -> list[Metric]:
    import torch
    import vllm.model_executor.layers.mhc  # noqa: F401
    import vllm.model_executor.kernels.mhc as mhc_kernels

    torch.cuda.set_device(0)
    torch.manual_seed(seed + num_tokens)
    device = torch.device("cuda:0")
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = (2 * hc_mult) + hc_mult2
    hc_dim = hc_mult * hidden_size
    rms_eps = 1.0e-6
    hc_eps = 1.0e-6
    sinkhorn_eps = 1.0e-6
    hc_post_alpha = 2.0
    sinkhorn_repeat = 2

    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device
    )
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc_mult3, hc_dim, dtype=torch.float32, device=device)
    head_fn = torch.randn(hc_mult, hc_dim, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device).abs() + 0.25
    head_scale = torch.randn(1, dtype=torch.float32, device=device).abs() + 0.25
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1
    head_base = torch.randn(hc_mult, dtype=torch.float32, device=device) * 0.1
    norm_weight = (
        torch.randn(hidden_size, dtype=torch.bfloat16, device=device).abs() + 0.25
    )
    norm_eps = 1.0e-6

    post_mix, comb_mix, layer_input = torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        None,
        0.0,
    )
    ref_post, ref_comb, ref_input = mhc_kernels.mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    post_mix_norm, comb_mix_norm, layer_input_norm = torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        norm_weight,
        norm_eps,
    )
    ref_input_norm = _rms_norm_torch(ref_input, norm_weight, norm_eps)

    post_out = torch.ops.vllm.mhc_post_tilelang(x, residual, post_mix, comb_mix)
    ref_post_out = mhc_kernels.mhc_post_torch(x, residual, post_mix, comb_mix)

    fused_res, fused_post, fused_comb, fused_input = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        1,
        None,
        0.0,
    )
    (
        fused_res_norm,
        fused_post_norm,
        fused_comb_norm,
        fused_input_norm,
    ) = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        1,
        norm_weight,
        norm_eps,
    )
    ref_fused_res = mhc_kernels.mhc_post_torch(x, residual, post_mix, comb_mix)
    ref_fused_post, ref_fused_comb, ref_fused_input = mhc_kernels.mhc_pre_torch(
        ref_fused_res,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    ref_fused_input_norm = _rms_norm_torch(ref_fused_input, norm_weight, norm_eps)

    head_tilelang = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    torch.ops.vllm.hc_head_fused_kernel_tilelang(
        residual,
        head_fn,
        head_scale,
        head_base,
        head_tilelang,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    ref_head = _hc_head_torch(
        residual,
        head_fn,
        head_scale,
        head_base,
        rms_eps,
        hc_eps,
    )
    head_triton = None
    if include_triton_head:
        head_triton = torch.empty_like(head_tilelang)
        torch.ops.vllm.hc_head_triton(
            residual,
            head_fn,
            head_scale,
            head_base,
            head_triton,
            hidden_size,
            rms_eps,
            hc_eps,
            hc_mult,
        )
    torch.cuda.synchronize()

    prefix = f"tokens={num_tokens}"
    metrics = [
        _metric(f"{prefix}:mhc_pre:post_mix", post_mix, ref_post),
        _metric(f"{prefix}:mhc_pre:comb_mix", comb_mix, ref_comb),
        _metric(f"{prefix}:mhc_pre:layer_input", layer_input, ref_input),
        _metric(f"{prefix}:mhc_pre_norm:post_mix", post_mix_norm, ref_post),
        _metric(f"{prefix}:mhc_pre_norm:comb_mix", comb_mix_norm, ref_comb),
        _metric(
            f"{prefix}:mhc_pre_norm:layer_input",
            layer_input_norm,
            ref_input_norm,
        ),
        _metric(f"{prefix}:mhc_post", post_out, ref_post_out),
        _metric(f"{prefix}:mhc_fused:residual", fused_res, ref_fused_res),
        _metric(f"{prefix}:mhc_fused:post_mix", fused_post, ref_fused_post),
        _metric(f"{prefix}:mhc_fused:comb_mix", fused_comb, ref_fused_comb),
        _metric(f"{prefix}:mhc_fused:layer_input", fused_input, ref_fused_input),
        _metric(f"{prefix}:mhc_fused_norm:residual", fused_res_norm, ref_fused_res),
        _metric(f"{prefix}:mhc_fused_norm:post_mix", fused_post_norm, ref_fused_post),
        _metric(f"{prefix}:mhc_fused_norm:comb_mix", fused_comb_norm, ref_fused_comb),
        _metric(
            f"{prefix}:mhc_fused_norm:layer_input",
            fused_input_norm,
            ref_fused_input_norm,
        ),
        _metric(f"{prefix}:hc_head_tilelang", head_tilelang, ref_head),
    ]
    if head_triton is not None:
        metrics.append(_metric(f"{prefix}:hc_head_triton", head_triton, ref_head))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default=os.getenv("DS4_MHC_CORRECTNESS_TOKENS", "1,4"))
    parser.add_argument("--hidden-size", type=int, default=int(os.getenv("DS4_MHC_CORRECTNESS_HIDDEN_SIZE", "4096")))
    parser.add_argument("--hc-mult", type=int, default=int(os.getenv("DS4_MHC_CORRECTNESS_HC_MULT", "4")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("DS4_MHC_CORRECTNESS_SEED", "1234")))
    parser.add_argument("--max-abs", type=float, default=float(os.getenv("DS4_MHC_CORRECTNESS_MAX_ABS", "1.0")))
    parser.add_argument("--mean-abs", type=float, default=float(os.getenv("DS4_MHC_CORRECTNESS_MEAN_ABS", "0.05")))
    parser.add_argument(
        "--include-triton-head",
        action="store_true",
        default=os.getenv("DS4_MHC_CORRECTNESS_INCLUDE_TRITON_HEAD", "0").strip().lower()
        in ("1", "true", "yes", "on"),
        help="Also test the debug Triton hc_head backend. This requires Python.h.",
    )
    args = parser.parse_args()

    all_metrics: list[Metric] = []
    for num_tokens in _parse_tokens(args.tokens):
        all_metrics.extend(
            _run_case(
                num_tokens=num_tokens,
                hidden_size=args.hidden_size,
                hc_mult=args.hc_mult,
                seed=args.seed,
                include_triton_head=args.include_triton_head,
            )
        )

    failed = [
        metric
        for metric in all_metrics
        if (
            not metric.finite
            or metric.max_abs > args.max_abs
            or metric.mean_abs > args.mean_abs
        )
    ]
    payload = {
        "status": "fail" if failed else "pass",
        "hidden_size": args.hidden_size,
        "hc_mult": args.hc_mult,
        "thresholds": {"max_abs": args.max_abs, "mean_abs": args.mean_abs},
        "metrics": [metric.as_dict() for metric in all_metrics],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
