#!/usr/bin/env python3
"""Static checks for DS4 DSV4 layer backend isolation."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
model = (root / "vllm/models/deepseek_v4/nvidia/model.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "DSV4 layer backend env is registered",
        'VLLM_DS4_DSV4_LAYER_BACKEND: str = "cuda"' in envs
        and '"VLLM_DS4_DSV4_LAYER_BACKEND": lambda:' in envs,
    ),
    (
        "CUDA remains the production default",
        'os.environ.get(\n        "VLLM_DS4_DSV4_LAYER_BACKEND", "cuda"\n    )'
        in envs
        and 'export VLLM_DS4_DSV4_LAYER_BACKEND="${VLLM_DS4_DSV4_LAYER_BACKEND:-cuda}"'
        in launcher,
    ),
    (
        "native-debug is explicit diagnostic only",
        'elif backend == "native-debug":' in model
        and "return self._forward_native(" in model,
    ),
    (
        "unknown DSV4 layer backend fails closed",
        "Unsupported VLLM_DS4_DSV4_LAYER_BACKEND" in model,
    ),
    (
        "native-debug uses hidden-only PP handoff",
        "native_layer_debug = envs.VLLM_DS4_DSV4_LAYER_BACKEND == \"native-debug\""
        in model
        and "close_hc_boundary = flush_hc_boundary or native_layer_debug" in model,
    ),
    (
        "native-debug skips extra stage hc_post",
        "and not native_layer_debug" in model
        and "if close_hc_boundary:" in model,
    ),
    (
        "launcher logs layer backend",
        "dsv4_layer_backend=$VLLM_DS4_DSV4_LAYER_BACKEND" in launcher,
    ),
    (
        "relaunch build validates layer backend audit",
        "tools/ds4_dsv4_layer_backend_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
