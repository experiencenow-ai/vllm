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
guard = read("tools/ds4_200g_guard.sh")
preflight = read("tools/ds4_nccl_preflight.py")
relaunch = read("tools/ds4_relaunch_spark_service.py")
stop = read("tools/ds4_stop_spark_processes.py")
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
check("PP4xTP2 launcher admits throughput-profile request waves", 'DSV4_SCHED_MAX_NEW_REQS_PER_STEP:-$DSV4_MAX_NUM_SEQS' in pp4)
check("PP4xTP2 launcher admits throughput-profile prefill token waves", 'DSV4_SCHED_MAX_NEW_PREFILL_TOKENS_PER_STEP:-$DSV4_MAX_NUM_BATCHED_TOKENS' in pp4)
check("PP4xTP2 launcher preserves final-only nonstreaming output", "VLLM_DS4_FINAL_ONLY_NONSTREAMING" in pp4)
check("PP4xTP2 launcher preserves fused execute/sample default", "VLLM_DS4_FUSED_EXECUTE_SAMPLE" in pp4)
check("PP4xTP2 launcher serves existing DS4 DSV4 API model by default", 'DSV4_SERVED_MODEL_NAME:-deepseek-v4-flash-pp8' in pp4)
check("PP4xTP2 launcher does not set PP-only global backend", 'export VLLM_DS4_PP_ONLY_GLOBAL_BACKEND=' not in pp4)
check("PP4xTP2 launcher fails early if PP-only global backend leaks in", "PP4xTP2xEP refuses VLLM_DS4_PP_ONLY_GLOBAL_BACKEND" in pp4)
check("PP4xTP2 launcher uses gloo for global control", 'VLLM_DS4_DISTRIBUTED_BACKEND="${VLLM_DS4_DISTRIBUTED_BACKEND:-gloo}"' in pp4)
check("PP4xTP2 launcher uses pairwise TP/EP NCCL preflight", 'DS4_NCCL_PREFLIGHT_MODE="${DS4_NCCL_PREFLIGHT_MODE:-tp_pair_nccl}"' in pp4)
check("PP4xTP2 launcher declares adjacent TP/EP NCCL groups", 'DS4_NCCL_PREFLIGHT_GROUPS="${DS4_NCCL_PREFLIGHT_GROUPS:-0,1;2,3;4,5;6,7}"' in pp4)
check("PP4xTP2 launcher derives pair-local NCCL interface from TP partner route", "ds4_select_tp_pair_nccl_ifname" in pp4 and "partner_loopback" in pp4 and "DS4_200G_NCCL_IFNAME" in pp4)
check("200G guard separates route validation IFNAME from NCCL IFNAME", "DS4_200G_NCCL_IFNAME" in guard and "nccl_ifnames_csv" in guard and "NCCL interface" in guard)
check("200G guard pins IB HCA from NCCL IFNAME, not the full route list", 'ds4_200g_check_or_export NCCL_IB_HCA "$nccl_hcas_csv"' in guard)
check("NCCL preflight prints pair-local NCCL interface", '"DS4_200G_NCCL_IFNAME"' in preflight)
check("PP4xTP2 launcher CPU-stages non-adjacent PP tensor dicts", "VLLM_DS4_PP_CPU_STAGED_TENSOR_DICT" in pp4)
check("PP4xTP2 launcher refuses PyNCCL PP P2P on the routed ring", "refuses VLLM_DS4_PP_PYNCCL_TENSOR_DICT=1" in pp4)
check("parallel_state has DS4 PP CPU-staged tensor-dict path", "TensorMetadataCpuStaged" in ps and "_should_cpu_stage_ds4_pp_tensor_dict" in ps)
check("relaunch supports PP4xTP2xEP service", "dsv4-pp4-tp2-ep" in relaunch)
check("relaunch validates PP4xTP2 launcher and speed audit", "ds4_launch_dsv4_flash_pp4_tp2_ep.sh" in relaunch and "ds4_speed_path_audit.py" in relaunch)
check("stop script catches PP4xTP2 launcher", "pp4_tp2_ep" in stop)
check("stop script catches stale NCCL preflight processes", "ds4_nccl_preflight" in stop)
check("relaunch fails early when head startup process exits", "startup-fail-fast-s" in relaunch and "head_service_process_alive" in relaunch)
check("relaunch process probe avoids pgrep self-match", "[d]s4_nccl_preflight.py" in relaunch and "[v]llm.entrypoints" in relaunch)
