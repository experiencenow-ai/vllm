#!/usr/bin/env python3
"""Static audit for DSV4 FlashInfer CUTLASS MXFP4 W13 layout policy."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
mxfp4 = (root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "DSV4 CUTLASS MXFP4 W13 layout env is registered",
        "VLLM_DS4_DSV4_CUTLASS_MXFP4_W13_LAYOUT" in envs
        and '"swapped"' in envs,
    ),
    (
        "CUTLASS converter accepts only explicit gate-up or swapped layouts",
        'w13_layout not in ("gate-up", "swapped")' in mxfp4
        and "raise RuntimeError" in mxfp4,
    ),
    (
        "gate-up layout preserves deinterleaved DSV4 reference order",
        'if w13_layout == "gate-up":' in mxfp4
        and "w13_weight_cutlass = deinterleaved_w13_w" in mxfp4
        and "w13_bias_cutlass = deinterleaved_w13_b.to(torch.bfloat16)" in mxfp4
        and "w13_scale_cutlass = deinterleaved_w13_s" in mxfp4,
    ),
    (
        "legacy swapped layout remains explicit-only for comparison",
        "w13_weight_cutlass = torch.cat([w3_w, w1_w], dim=1)" in mxfp4
        and "w13_bias_cutlass = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)"
        in mxfp4
        and "w13_scale_cutlass = torch.cat([s3, s1], dim=1)" in mxfp4,
    ),
    (
        "DSV4 production launcher defaults to gate-up",
        'VLLM_DS4_DSV4_CUTLASS_MXFP4_W13_LAYOUT:-gate-up' in launcher,
    ),
    (
        "relaunch build validates DSV4 MXFP4 layout audit",
        "tools/ds4_dsv4_mxfp4_layout_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
