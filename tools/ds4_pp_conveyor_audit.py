#!/usr/bin/env python3
"""Static audit for DS4 PP conveyor transport defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def check(name: str, ok: bool) -> int:
    print(("PASS" if ok else "FAIL") + f": {name}")
    return 0 if ok else 1


def main() -> int:
    envs = read("vllm/envs.py")
    worker = read("vllm/v1/worker/gpu_worker.py")
    parallel = read("vllm/distributed/parallel_state.py")
    dsv4 = read("tools/ds4_launch_dsv4_flash_pp8.sh")
    qwen_fast = read("tools/ds4_launch_qwen27_nvfp4_pp8.sh")
    qwen_bf16 = read("tools/ds4_launch_qwen27_pp8.sh")
    failures = 0
    failures += check(
        "env exposes PP conveyor overlap controls",
        "VLLM_DS4_PP_OVERLAP_SEND" in envs
        and "VLLM_DS4_PP_SEND_BUFFER_SLOTS" in envs
        and "VLLM_DS4_PP_SEND_BUFFER_MAX_BYTES" in envs,
    )
    failures += check(
        "env exposes PP direct CUDA tensor-dict controls",
        "VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT" in envs
        and "VLLM_DS4_PP_DIRECT_CUDA_MIN_BYTES" in envs,
    )
    failures += check(
        "env exposes PP Gantt trace controls",
        "VLLM_DS4_PP_GANTT_TRACE" in envs
        and "VLLM_DS4_PP_GANTT_TRACE_EVERY" in envs,
    )
    failures += check(
        "worker owns a CUDA PP send buffer ring",
        "class _Ds4PpSendBufferSlot" in worker
        and "_pp_send_buffer_slots" in worker
        and "_get_ds4_pp_send_buffer_slot" in worker,
    )
    failures += check(
        "worker copies PP outputs into owned send buffers",
        "_buffer_pp_output_for_send" in worker
        and "send_tensor.copy_(tensor, non_blocking=True)" in worker
        and "send_slot.handles = send_handles" in worker,
    )
    failures += check(
        "worker avoids execute-start send barrier in conveyor mode",
        "if envs.VLLM_DS4_PP_OVERLAP_SEND and forward_pass" in worker
        and "self._drain_completed_pp_send_work()" in worker,
    )
    failures += check(
        "worker blocks only when a reusable send buffer slot is still busy",
        "send_buffer_wait" in worker
        and "_wait_pp_send_handles(slot.handles)" in worker,
    )
    failures += check(
        "worker emits PP Gantt events",
        "DS4 PP GANTT" in worker
        and "recv_post" in worker
        and "forward_done_intermediate" in worker
        and "send_enqueue_buffered" in worker,
    )
    failures += check(
        "parallel_state keeps send waits stream-ordered in conveyor mode",
        "synchronize_on_wait=not envs.VLLM_DS4_PP_OVERLAP_SEND" in parallel,
    )
    failures += check(
        "parallel_state accepts direct CUDA tensor-dict alias",
        "VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT" in parallel
        and "VLLM_DS4_PP_PYNCCL_TENSOR_DICT" in parallel,
    )
    for script_name, script in (
        ("DSV4 PP", dsv4),
        ("Qwen NVFP4 PP", qwen_fast),
        ("Qwen BF16 PP", qwen_bf16),
    ):
        failures += check(
            f"{script_name} launcher enables PP conveyor defaults",
            'VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT="${VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT:-1}"'
            in script
            and 'VLLM_DS4_PP_OVERLAP_SEND="${VLLM_DS4_PP_OVERLAP_SEND:-1}"'
            in script
            and 'VLLM_DS4_PP_SEND_BUFFER_SLOTS="${VLLM_DS4_PP_SEND_BUFFER_SLOTS:-4}"'
            in script,
        )
    failures += check(
        "DSV4 PP launcher defaults to 8K mHC TileLang slabs",
        'VLLM_DS4_MHC_TILELANG_MAX_TOKENS="${VLLM_DS4_MHC_TILELANG_MAX_TOKENS:-8192}"'
        in dsv4,
    )
    failures += check(
        "DSV4 throughput profiles use conveyor-shaped request waves",
        'DSV4_SCHED_MAX_NEW_REQS_PER_STEP:=64' in dsv4
        and "sched_max_new_reqs=$VLLM_DS4_SCHED_MAX_NEW_REQS_PER_STEP" in dsv4,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
