#!/usr/bin/env python3
"""Static audit for DS4 high-speed channel integration."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

checks = []

def require(path: str, needle: str, label: str) -> None:
    text = (ROOT / path).read_text()
    if needle not in text:
        raise SystemExit(f"FAIL: {label}\nmissing {needle!r} in {path}")
    checks.append(label)

require(
    "vllm/distributed/ds4_high_speed_channel.py",
    "class Ds4StripedNcclTensorChannel",
    "striped NCCL tensor channel exists",
)
require(
    "vllm/distributed/ds4_high_speed_channel.py",
    "PyNcclCommunicator(cpu_group, device)",
    "each stripe uses an independent PyNCCL communicator",
)
require(
    "vllm/distributed/ds4_high_speed_channel.py",
    "torch.cuda.Stream(device=device)",
    "striped channel can use independent CUDA streams",
)
require(
    "vllm/distributed/ds4_high_speed_channel.py",
    "VLLM_DS4_PP_PYNCCL_P2P_CREDIT",
    "striped channel can post a tiny reverse credit for PyNCCL P2P",
)
require(
    "vllm/distributed/parallel_state.py",
    "self.ds4_pp_striped_nccl_channel = build_ds4_pp_striped_nccl_channel",
    "PP group constructs striped channel",
)
require(
    "vllm/distributed/parallel_state.py",
    "striped_channel.send(tensor, peer)",
    "PP tensor send uses striped channel",
)
require(
    "vllm/distributed/parallel_state.py",
    "striped_channel.recv(tensor, peer)",
    "PP tensor recv uses striped channel",
)
require(
    "vllm/envs.py",
    "VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT",
    "striped channel env exists",
)
require(
    "vllm/envs.py",
    "VLLM_DS4_PP_PYNCCL_P2P_CREDIT",
    "PyNCCL P2P credit env exists",
)

for script in [
    "tools/ds4_launch_dsv4_flash_pp8.sh",
    "tools/ds4_launch_qwen27_nvfp4_pp8.sh",
    "tools/ds4_launch_qwen27_pp8.sh",
]:
    path = ROOT / script
    if path.exists():
        require(script, "VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT", f"{script} exposes striped channel")
        require(script, "VLLM_DS4_PP_STRIPED_NCCL_TENSOR_DICT:-0", f"{script} keeps independent striped channel opt-in")
        require(script, "VLLM_DS4_PP_PYNCCL_P2P_CREDIT", f"{script} enables PyNCCL P2P credit")

for label in checks:
    print(f"PASS: {label}")
