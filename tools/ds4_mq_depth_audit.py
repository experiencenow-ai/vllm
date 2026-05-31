#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for DS4 pipeline message-queue depth.

PP8 can keep eight execute_model futures in flight. The multiprocessing
response queues must have a ring depth larger than that, otherwise fast workers
can block while returning outputs before the engine drains older futures.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"PASS: {message}")


def main() -> None:
    envs = (ROOT / "vllm/envs.py").read_text()
    parallel_state = (ROOT / "vllm/distributed/parallel_state.py").read_text()
    multiproc = (ROOT / "vllm/v1/executor/multiproc_executor.py").read_text()
    ray_v2 = (ROOT / "vllm/v1/executor/ray_executor_v2.py").read_text()
    dsv4_pp8 = (ROOT / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
    qwen_nvfp4 = (ROOT / "tools/ds4_launch_qwen27_nvfp4_pp8.sh").read_text()
    qwen_bf16 = (ROOT / "tools/ds4_launch_qwen27_pp8.sh").read_text()

    require("VLLM_MQ_MAX_CHUNKS" in envs, "VLLM_MQ_MAX_CHUNKS env exists")
    require(
        'os.getenv("VLLM_MQ_MAX_CHUNKS", "10")' in envs,
        "VLLM_MQ_MAX_CHUNKS has upstream-compatible default",
    )
    require(
        "envs.VLLM_MQ_MAX_CHUNKS" in parallel_state
        and "self.cpu_group,\n                1 << 22,\n                6," not in parallel_state,
        "parallel-state MQ broadcasters use env depth instead of hardcoded 6",
    )
    require(
        "max_chunks=envs.VLLM_MQ_MAX_CHUNKS" in multiproc,
        "multiproc scheduler/response queues use env depth",
    )
    require(
        "max_chunks=envs.VLLM_MQ_MAX_CHUNKS" in ray_v2,
        "ray v2 scheduler/response queues use env depth",
    )
    for name, text in (
        ("DSV4 PP8", dsv4_pp8),
        ("Qwen NVFP4 PP8", qwen_nvfp4),
        ("Qwen BF16 PP8", qwen_bf16),
    ):
        require(
            'export VLLM_MQ_MAX_CHUNKS="${VLLM_MQ_MAX_CHUNKS:-64}"' in text,
            f"{name} launcher defaults MQ depth to 64",
        )


if __name__ == "__main__":
    main()
