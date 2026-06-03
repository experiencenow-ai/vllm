#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parallel = (ROOT / "vllm/distributed/parallel_state.py").read_text()
envs = (ROOT / "vllm/envs.py").read_text()
dsv4 = (ROOT / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()

checks: list[tuple[str, bool]] = [
    (
        "PP PyNCCL tensor-dict env exists",
        "VLLM_DS4_PP_PYNCCL_TENSOR_DICT" in envs,
    ),
    (
        "PP PyNCCL tensor-dict striping env exists",
        "VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES" in envs,
    ),
    (
        "PP tensor-dict fast path has CUDA event handle",
        "class _CudaEventHandle" in parallel and "record_event" in parallel,
    ),
    (
        "PP tensor-dict fast path enqueues PyNCCL P2P",
        "_enqueue_ds4_pynccl_p2p" in parallel and "batch_isend_irecv" in parallel,
    ),
    (
        "PP tensor-dict fast path stripes large CUDA tensors",
        "_ds4_pynccl_p2p_chunks" in parallel
        and "VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES" in parallel,
    ),
    (
        "requested fast path refuses disabled device communicator",
        "VLLM_DS4_PP_DISABLE_DEVICE_COMMUNICATOR is enabled" in parallel,
    ),
    (
        "requested fast path refuses missing device communicator",
        "has no device communicator" in parallel,
    ),
    (
        "DSV4 launcher exposes PyNCCL tensor-dict switch",
        "VLLM_DS4_PP_PYNCCL_TENSOR_DICT" in dsv4,
    ),
    (
        "DSV4 launcher defaults to PyNCCL tensor-dict path",
        'VLLM_DS4_PP_PYNCCL_TENSOR_DICT:-1' in dsv4,
    ),
    (
        "DSV4 launcher keeps torch ProcessGroup tensor-dict diagnostic opt-in",
        'VLLM_DS4_PP_TORCH_PG_TENSOR_DICT:-0' in dsv4,
    ),
    (
        "DSV4 launcher stripes tensor-dict transfers by default",
        'VLLM_DS4_PP_PYNCCL_TENSOR_DICT_STRIPES:-8' in dsv4,
    ),
]

failed = False
for name, ok in checks:
    print(("PASS" if ok else "FAIL") + ": " + name)
    failed |= not ok
if failed:
    raise SystemExit(1)
