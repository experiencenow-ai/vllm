#!/usr/bin/env python3
"""Static checks for DS4 DSV4 PP weight-load auditing."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
model = (root / "vllm/models/deepseek_v4/nvidia/model.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "DSV4 weight audit env is registered",
        "VLLM_DS4_DSV4_WEIGHT_AUDIT: bool = False" in envs
        and "\"VLLM_DS4_DSV4_WEIGHT_AUDIT\": lambda:" in envs,
    ),
    (
        "expert loads are only counted after success",
        "loaded_expert_param = None" in model
        and "if loaded_expert_param is not None:" in model
        and "loaded_params.add(loaded_expert_param)" in model,
    ),
    (
        "owned edge weights are required",
        "model.embed_tokens.weight" in model
        and "lm_head.weight" in model
        and "model.norm.weight" in model
        and "model.hc_head_fn" in model
        and "model.hc_head_base" in model
        and "model.hc_head_scale" in model,
    ),
    (
        "owned local layer coverage is checked",
        "expected_layers = set(range(self.model.start_layer, self.model.end_layer))"
        in model
        and "missing_layers = sorted(expected_layers - loaded_layers)" in model,
    ),
    (
        "weight audit fails closed",
        "DS4 DSV4 weight audit failed" in model
        and "raise RuntimeError" in model,
    ),
    (
        "launcher enables and logs weight audit",
        'export VLLM_DS4_DSV4_WEIGHT_AUDIT="${VLLM_DS4_DSV4_WEIGHT_AUDIT:-1}"'
        in launcher
        and "weight_audit=$VLLM_DS4_DSV4_WEIGHT_AUDIT" in launcher,
    ),
    (
        "relaunch build validates weight audit",
        "tools/ds4_dsv4_weight_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
