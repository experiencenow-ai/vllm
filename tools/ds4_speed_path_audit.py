#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def check(name: str, ok: bool) -> None:
    print(("PASS" if ok else "FAIL") + ": " + name)
    if not ok:
        raise SystemExit(1)


pp8 = read("tools/ds4_launch_dsv4_flash_pp8.sh")
pp4 = read("tools/ds4_launch_dsv4_flash_pp4_tp2_ep.sh")
relaunch = read("tools/ds4_relaunch_spark_service.py")
envs = read("vllm/envs.py")
ps = read("vllm/distributed/parallel_state.py")

check("PP8 keeps PyNCCL tensor-dict disabled by default after live regression", 'VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-0' in pp8)
check("PP8 keeps device communicator disabled by default after live regression", 'VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR:-1' in pp8)
check("PyNCCL tensor-dict env still exists as diagnostic speed path", "VLLM_DS4_PP_PYNCCL_TENSOR_DICT" in envs)
check("parallel_state has DS4 PP PyNCCL tensor-dict gate", "_can_use_ds4_pynccl_tensor_dict" in ps or "_can_use_ds4_pp_pynccl_tensor_dict" in ps)
check("PP4xTP2 launcher exists", 'TP_SIZE="${TP_SIZE:-2}"' in pp4 and 'PP_SIZE="${PP_SIZE:-4}"' in pp4)
check("PP4xTP2 launcher enforces 8x world geometry", 'requires NNODES=8 PP_SIZE=4 TP_SIZE=2' in pp4)
check("PP4xTP2 launcher enables expert parallel", "--enable-expert-parallel" in pp4)
check("PP4xTP2 launcher has 43-layer partition", 'DSV4_FLASH_PP_LAYER_PARTITION="7,11,14,11"' in pp4)
check("PP4xTP2 launcher preserves bounded prefill waves", "VLLM_DS4_SCHED_MAX_NEW_PREFILL_TOKENS_PER_STEP" in pp4)
check("PP4xTP2 launcher preserves final-only nonstreaming output", "VLLM_DS4_FINAL_ONLY_NONSTREAMING" in pp4)
check("PP4xTP2 launcher preserves fused execute/sample default", "VLLM_DS4_FUSED_EXECUTE_SAMPLE" in pp4)
check("PP4xTP2 launcher serves existing DS4 DSV4 API model by default", 'DSV4_SERVED_MODEL_NAME:-deepseek-v4-flash-pp8' in pp4)
check("PP4xTP2 launcher does not set PP-only global backend", 'export VLLM_DS4_PP_ONLY_GLOBAL_BACKEND=' not in pp4)
check("PP4xTP2 launcher fails early if PP-only global backend leaks in", "PP4xTP2xEP refuses VLLM_DS4_PP_ONLY_GLOBAL_BACKEND" in pp4)
check("relaunch supports PP4xTP2xEP service", "dsv4-pp4-tp2-ep" in relaunch)
check("relaunch validates PP4xTP2 launcher and speed audit", "ds4_launch_dsv4_flash_pp4_tp2_ep.sh" in relaunch and "ds4_speed_path_audit.py" in relaunch)
