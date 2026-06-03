#!/usr/bin/env python3
"""Static checks for DS4 DSV4 PP hyper-connection boundary modes."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
model = (root / "vllm/models/deepseek_v4/nvidia/model.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "flush HC boundary env is registered",
        "VLLM_DS4_DSV4_PP_FLUSH_HC_BOUNDARY" in envs
        and "cross-rank fused MHC state carry" in envs,
    ),
    (
        "flush mode receives canonical multi-stream hidden tensors",
        "if envs.VLLM_DS4_DSV4_PP_FLUSH_HC_BOUNDARY:" in model
        and "(batch_size, self.hc_mult, self.config.hidden_size)" in model,
    ),
    (
        "flush mode reopens stage with no inherited HC state",
        "flush_hc_boundary = envs.VLLM_DS4_DSV4_PP_FLUSH_HC_BOUNDARY" in model
        and "if flush_hc_boundary:\n                residual, post_mix, res_mix = None, None, None"
        in model,
    ),
    (
        "flush mode closes HC state before non-last PP boundary",
        "and (get_pp_group().is_last_rank or flush_hc_boundary)" in model
        and 'return IntermediateTensors({"hidden_states": hidden_states})' in model,
    ),
    (
        "default fused HC boundary mode remains available",
        "DeepSeek V4 PP boundary is missing hyper-connection" in model
        and '"residual": residual' in model
        and '"post_mix": post_mix' in model
        and '"res_mix": res_mix' in model,
    ),
    (
        "launcher exposes and logs HC boundary mode",
        'export VLLM_DS4_DSV4_PP_FLUSH_HC_BOUNDARY="${VLLM_DS4_DSV4_PP_FLUSH_HC_BOUNDARY:-0}"'
        in launcher
        and "pp_hc_boundary_flush=$VLLM_DS4_DSV4_PP_FLUSH_HC_BOUNDARY" in launcher,
    ),
    (
        "relaunch build validates HC boundary audit",
        "tools/ds4_dsv4_pp_hc_boundary_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
