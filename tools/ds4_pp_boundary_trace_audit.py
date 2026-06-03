#!/usr/bin/env python3
"""Static checks for DS4 PP tensor boundary diagnostics."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
parallel = (root / "vllm/distributed/parallel_state.py").read_text()
worker = (root / "vllm/v1/worker/gpu_worker.py").read_text()
dsv4_pp8 = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "boundary trace envs are registered",
        "VLLM_DS4_PP_BOUNDARY_TRACE" in envs
        and "VLLM_DS4_PP_BOUNDARY_TRACE_EVERY" in envs
        and "VLLM_DS4_PP_BOUNDARY_TRACE_MAX_ELEMS" in envs
        and "VLLM_DS4_PP_BOUNDARY_TRACE_SYNC" in envs,
    ),
    (
        "boundary trace helper logs tensor stats",
        "def ds4_log_pp_tensor_dict_boundary(" in parallel
        and "DS4 PP boundary trace:" in parallel
        and "checksum=" in parallel
        and "finite=" in parallel
        and "prefix=" in parallel,
    ),
    (
        "send path traces actual tensor payloads",
        "ds4_trace_tensors" in parallel
        and 'direction="send"' in parallel
        and "ds4_log_pp_tensor_dict_boundary(" in parallel,
    ),
    (
        "sync receive path traces after postprocess",
        "for fn in postprocess:" in parallel
        and 'direction="recv"' in parallel,
    ),
    (
        "async receive path traces after wait_for_comm postprocess",
        "from vllm.distributed.parallel_state import (" in worker
        and "ds4_log_pp_tensor_dict_boundary" in worker
        and 'direction="recv"' in worker,
    ),
    (
        "DSV4 launcher exposes trace knobs and logs them",
        'export VLLM_DS4_PP_BOUNDARY_TRACE="${VLLM_DS4_PP_BOUNDARY_TRACE:-0}"'
        in dsv4_pp8
        and "pp_boundary_trace=$VLLM_DS4_PP_BOUNDARY_TRACE" in dsv4_pp8,
    ),
    (
        "relaunch build validates boundary trace audit",
        "tools/ds4_pp_boundary_trace_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
