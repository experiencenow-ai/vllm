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
rail_tcp = read("tools/ds4_rail_tcp_preflight.py")
relaunch = read("tools/ds4_relaunch_spark_service.py")
stop = read("tools/ds4_stop_spark_processes.py")
envs = read("vllm/envs.py")
ps = read("vllm/distributed/parallel_state.py")

check("PP8 enables PyNCCL tensor-dict by default for performance runs", 'VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-1' in pp8)
check("PP8 stripes PyNCCL tensor-dict transfers by default", 'VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES:-8' in pp8)
check("PP8 keeps the PP device communicator enabled by default", 'VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR:-0' in pp8)
check("PyNCCL tensor-dict env exists as the required PP fast path", "VLLM_DS4_PP_PYNCCL_TENSOR_DICT" in envs)
check("PyNCCL tensor-dict striping env exists", "VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES" in envs)
check("parallel_state has DS4 PP PyNCCL tensor-dict gate", "_can_use_ds4_pynccl_tensor_dict" in ps or "_can_use_ds4_pp_pynccl_tensor_dict" in ps)
check("parallel_state stripes DS4 PP PyNCCL tensor-dict payloads", "_ds4_pynccl_p2p_chunks" in ps)
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
check("PP4xTP2 launcher uses NCCL for global control", 'VLLM_DS4_DISTRIBUTED_BACKEND="${VLLM_DS4_DISTRIBUTED_BACKEND:-nccl}"' in pp4)
check("PP4xTP2 launcher refuses gloo/CPU global control", "gloo/CPU control paths hide PP transport regressions" in pp4)
check("PP4xTP2 launcher preflights required TP groups and PP P2P links", 'DS4_NCCL_PREFLIGHT_MODE="${DS4_NCCL_PREFLIGHT_MODE:-p2p_nccl}"' in pp4 and "DS4_NCCL_PREFLIGHT_GROUPS" in pp4)
check("PP4xTP2 launcher declares TP/EP pair collectives", "0,1;2,3;4,5;6,7" in pp4)
check("PP4xTP2 launcher declares directional PP boundary P2P pairs", "0-2;1-3;2-4;3-5;4-6;5-7" in pp4)
check("PP4xTP2 launcher uses unidirectional PP P2P preflight", 'DS4_NCCL_PREFLIGHT_P2P_DIRECTION="${DS4_NCCL_PREFLIGHT_P2P_DIRECTION:-unidirectional}"' in pp4)
check("PP4xTP2 launcher refuses pair-local NCCL view", "refuses DS4_200G_PAIR_LOCAL_NCCL=1" in pp4)
check("200G guard separates route validation IFNAME from NCCL IFNAME", "DS4_200G_NCCL_IFNAME" in guard and "nccl_ifnames_csv" in guard and "NCCL interface" in guard)
check("200G guard pins IB HCA from NCCL IFNAME, not the full route list", 'ds4_200g_check_or_export NCCL_IB_HCA "$nccl_hcas_csv"' in guard)
check("NCCL preflight prints NCCL interface override", '"DS4_200G_NCCL_IFNAME"' in preflight)
check("NCCL preflight has P2P bandwidth mode", "def _run_p2p_nccl_preflight(" in preflight and "DS4_NCCL_PREFLIGHT_P2P_PAIRS" in preflight)
check("NCCL preflight stripes P2P probes to match PP tensor transport", "DS4_NCCL_PREFLIGHT_P2P_STRIPES" in preflight and "_split_p2p_tensor" in preflight)
check("NCCL preflight separates directional PP P2P from TP pair collectives", "DS4_NCCL_PREFLIGHT_P2P_DIRECTION" in preflight and "pairwise NCCL group probes begin" in preflight)
check("rail TCP preflight script exists", "ds4_transfer.fast_copy data-plane shape" in rail_tcp)
check("rail TCP preflight discovers route rails like fast_copy", "ip\", \"route\", \"show\"" in rail_tcp and "destination_ip=dst_ip" in rail_tcp)
check("rail TCP preflight binds explicit client source rail IPs", "nc -N -s {rail.source_ip}" in rail_tcp)
check("rail TCP preflight uses many unencrypted streams per edge", "DS4_RAIL_TCP_PREFLIGHT_STREAMS" in rail_tcp and "threading.Thread" in rail_tcp)
check("200G guard can run rail TCP preflight before NCCL", "ds4_run_rail_tcp_preflight" in guard and "ds4_rail_tcp_preflight.py" in guard)
check("DSV4 PP8 enables rail TCP fabric preflight", "DS4_RAIL_TCP_PREFLIGHT_ACTIVE" in pp8 and "ds4_run_rail_tcp_preflight" in pp8)
check("DSV4 PP4xTP2 enables rail TCP physical fabric preflight", "DS4_RAIL_TCP_PREFLIGHT_ACTIVE" in pp4 and "DS4_RAIL_TCP_PREFLIGHT_PAIRS" in pp4)
check("PP4xTP2 launcher refuses CPU-staged PP tensor dict fallback", "refuses VLLM_DS4_PP_CPU_STAGED_TENSOR_DICT=1" in pp4)
check("PP4xTP2 launcher requires PyNCCL PP tensor dict fast path", "requires VLLM_DS4_PP_PYNCCL_TENSOR_DICT=1" in pp4)
check("PP4xTP2 launcher stripes PyNCCL tensor-dict transfers by default", 'VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES:-8' in pp4)
check("parallel_state has DS4 PP CPU-staged tensor-dict path", "TensorMetadataCpuStaged" in ps and "_should_cpu_stage_ds4_pp_tensor_dict" in ps)
check("relaunch supports PP4xTP2xEP service", "dsv4-pp4-tp2-ep" in relaunch)
check("relaunch validates PP4xTP2 launcher and speed audit", "ds4_launch_dsv4_flash_pp4_tp2_ep.sh" in relaunch and "ds4_speed_path_audit.py" in relaunch)
check("stop script catches PP4xTP2 launcher", "pp4_tp2_ep" in stop)
check("stop script catches stale NCCL preflight processes", "ds4_nccl_preflight" in stop)
check("relaunch fails early when head startup process exits", "startup-fail-fast-s" in relaunch and "head_service_process_alive" in relaunch)
check("relaunch process probe avoids pgrep self-match", "[d]s4_nccl_preflight.py" in relaunch and "[v]llm.entrypoints" in relaunch)
