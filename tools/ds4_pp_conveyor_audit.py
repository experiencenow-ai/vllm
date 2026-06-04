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
    runner = read("vllm/v1/worker/gpu_model_runner.py")
    parallel = read("vllm/distributed/parallel_state.py")
    tcp_channel = read("vllm/distributed/ds4_tcp_tensor_channel.py")
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
        "env exposes PP CUDA tensor-dict transport controls",
        "VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT" in envs
        and "VLLM_DS4_PP_TORCH_PG_TENSOR_DICT" in envs
        and "VLLM_DS4_PP_DIRECT_CUDA_MIN_BYTES" in envs,
    )
    failures += check(
        "env exposes symmetric PyNCCL P2P credit control",
        "VLLM_DS4_PP_PYNCCL_P2P_CREDIT" in envs,
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
        "def _ds4_pp_overlap_send_enabled" in worker
        and "if self._ds4_pp_overlap_send_enabled() and forward_pass" in worker
        and "self._drain_completed_pp_send_work()" in worker,
    )
    failures += check(
        "worker disables PP send overlap for CPU-staged transport",
        "envs.VLLM_DS4_PP_CPU_STAGED_TENSOR_DICT" in worker
        and "return False" in worker
        and "not self._ds4_pp_overlap_send_enabled()" in worker,
    )
    failures += check(
        "worker drains CPU-staged PP sends immediately",
        "if envs.VLLM_DS4_PP_CPU_STAGED_TENSOR_DICT and send_handles:" in worker
        and "send_wait_cpu_staged" in worker
        and "send_handles = []" in worker,
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
        "model runner lazily allocates PP receive intermediate buffers",
        "def sync_and_gather_intermediate_tensors" in runner
        and "if self.intermediate_tensors is None:" in runner
        and "self.model.make_empty_intermediate_tensors" in runner
        and "batch_size=self.max_num_tokens" in runner,
    )
    failures += check(
        "parallel_state keeps send waits stream-ordered in conveyor mode",
        "synchronize_on_wait=not envs.VLLM_DS4_PP_OVERLAP_SEND" in parallel,
    )
    failures += check(
        "parallel_state posts reverse credit for unidirectional PyNCCL P2P",
        "VLLM_DS4_PP_PYNCCL_P2P_CREDIT" in parallel
        and "credit_op" in parallel,
    )
    failures += check(
        "parallel_state keeps torch ProcessGroup CUDA tensor-dict diagnostic path",
        "VLLM_DS4_PP_TORCH_PG_TENSOR_DICT" in parallel
        and "_ds4_pp_torch_pair_group(dst)" in parallel
        and "_ds4_pp_torch_pair_group(src)" in parallel,
    )
    failures += check(
        "parallel_state has rail TCP PP payload channel",
        "build_ds4_pp_tcp_tensor_channel" in parallel
        and "_can_use_ds4_tcp_tensor_dict" in parallel
        and "_enqueue_ds4_tcp_send" in parallel
        and "_enqueue_ds4_tcp_recv" in parallel,
    )
    failures += check(
        "rail TCP PP channel uses per-peer directional sequence counters",
        "_send_seq: dict[int, int]" in tcp_channel
        and "_recv_seq: dict[int, int]" in tcp_channel
        and "def _next_send_seq(self, dst: int)" in tcp_channel
        and "def _next_recv_seq(self, src: int)" in tcp_channel
        and "Sequence counters are per peer and per direction" in tcp_channel,
    )
    failures += check(
        "parallel_state can warm torch pair groups with process-wide IFNAMEs",
        "VLLM_DS4_PP_TORCH_GROUP_WARMUP=1" in parallel
        and "VLLM_DS4_PP_TORCH_PAIR_IFNAME_MODE" in parallel
        and "pair_ifname_mode == \"edge\"" in parallel
        and "else nullcontext()" in parallel
        and "self._warm_ds4_pp_torch_pair_group(" in parallel,
    )
    failures += check(
        "parallel_state keeps PyNCCL tensor-dict path explicit",
        "VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT" in parallel
        and "VLLM_DS4_PP_PYNCCL_TENSOR_DICT" in parallel,
    )
    failures += check(
        "parallel_state defaults PyNCCL pair communicators to process-wide ifnames",
        "_build_ds4_pp_pynccl_pair_communicators" in parallel
        and "_ds4_pp_pynccl_pair_ifname_mode" in parallel
        and 'pair_ifname_mode == "edge"' in parallel
        and 'process_ifname = os.environ.get("NCCL_SOCKET_IFNAME"' in parallel,
    )
    failures += check(
        "DSV4 PP launcher defaults to explicit CPU-staged PP transport",
        'DS4_PP_TRANSPORT="${DS4_PP_TRANSPORT:-cpu-staged}"' in dsv4
        and 'VLLM_DS4_PP_CPU_STAGED_TENSOR_DICT="${VLLM_DS4_PP_CPU_STAGED_TENSOR_DICT:-1}"'
        in dsv4
        and 'VLLM_DS4_PP_OVERLAP_SEND="${VLLM_DS4_PP_OVERLAP_SEND:-0}"'
        in dsv4
        and 'DS4_NCCL_PREFLIGHT_MODE="${DS4_NCCL_PREFLIGHT_MODE:-skip}"'
        in dsv4,
    )
    failures += check(
        "DSV4 PP launcher keeps PyNCCL pair conveyor as explicit diagnostic mode",
        "pynccl-pair|pynccl_pair" in dsv4
        and 'VLLM_DS4_PP_EDGE_RAIL="${VLLM_DS4_PP_EDGE_RAIL:-${DS4_PP_EDGE_RAIL:-route}}"' in dsv4
        and 'VLLM_DS4_PP_PYNCCL_TENSOR_DICT="${VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-1}"'
        in dsv4
        and 'VLLM_DS4_PP_PYNCCL_PAIR_COMMUNICATORS="${VLLM_DS4_PP_PYNCCL_PAIR_COMMUNICATORS:-1}"'
        in dsv4,
    )
    failures += check(
        "DSV4 PP launcher has first-class torch pair transport",
        "torch-pair|torch_pair" in dsv4
        and 'DS4_PP_TRANSPORT="torch-pair"' in dsv4
        and 'VLLM_DS4_PP_EDGE_RAIL="${VLLM_DS4_PP_EDGE_RAIL:-${DS4_PP_EDGE_RAIL:-route}}"' in dsv4
        and 'VLLM_DS4_PP_TORCH_PG_TENSOR_DICT="${VLLM_DS4_PP_TORCH_PG_TENSOR_DICT:-1}"'
        in dsv4
        and 'VLLM_DS4_PP_TORCH_PAIR_GROUPS="${VLLM_DS4_PP_TORCH_PAIR_GROUPS:-1}"'
        in dsv4
        and 'VLLM_DS4_PP_TORCH_GROUP_WARMUP="${VLLM_DS4_PP_TORCH_GROUP_WARMUP:-1}"'
        in dsv4,
    )
    failures += check(
        "DSV4 PP launcher has first-class rail TCP staged transport",
        "tcp-staged|tcp_staged|rail-tcp|rail_tcp" in dsv4
        and 'DS4_PP_TRANSPORT="tcp-staged"' in dsv4
        and 'VLLM_DS4_PP_TCP_TENSOR_DICT="${VLLM_DS4_PP_TCP_TENSOR_DICT:-1}"'
        in dsv4
        and 'DS4_NCCL_PREFLIGHT_MODE="${DS4_NCCL_PREFLIGHT_MODE:-skip}"'
        in dsv4,
    )
    for script_name, script in (
        ("Qwen NVFP4 PP", qwen_fast),
        ("Qwen BF16 PP", qwen_bf16),
    ):
        failures += check(
            f"{script_name} launcher enables PP conveyor defaults",
            'VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR="${VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR:-1}"'
            in script
            and 'VLLM_DS4_PP_TORCH_PG_TENSOR_DICT="${VLLM_DS4_PP_TORCH_PG_TENSOR_DICT:-0}"'
            in script
            and 'VLLM_DS4_PP_PYNCCL_TENSOR_DICT="${VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-1}"'
            in script
            and 'VLLM_DS4_PP_PYNCCL_PAIR_COMMUNICATORS="${VLLM_DS4_PP_PYNCCL_PAIR_COMMUNICATORS:-1}"'
            in script
            and 'VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT="${VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT:-0}"'
            in script
            and 'VLLM_DS4_PP_OVERLAP_SEND="${VLLM_DS4_PP_OVERLAP_SEND:-1}"'
            in script
            and 'VLLM_DS4_PP_SEND_BUFFER_SLOTS="${VLLM_DS4_PP_SEND_BUFFER_SLOTS:-4}"'
            in script
            and 'VLLM_DS4_PP_PYNCCL_P2P_CREDIT="${VLLM_DS4_PP_PYNCCL_P2P_CREDIT:-1}"'
            in script
            and "VLLM_DS4_SKIP_PYNCCL_WARMUP_ALLREDUCE" in script
            and 'VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT="${VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT:-0}"'
            in script,
        )
    failures += check(
        "DSV4 PP launcher keeps PP conveyor controls available",
        'VLLM_DS4_PP_OVERLAP_SEND="${VLLM_DS4_PP_OVERLAP_SEND:-1}"' in dsv4
        and 'VLLM_DS4_PP_SEND_BUFFER_SLOTS="${VLLM_DS4_PP_SEND_BUFFER_SLOTS:-4}"'
        in dsv4
        and 'VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT="${VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT:-0}"'
        in dsv4,
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
