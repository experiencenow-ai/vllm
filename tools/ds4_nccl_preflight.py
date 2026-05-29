#!/usr/bin/env python3
"""Fail-fast NCCL fabric preflight for DS4 Spark launchers."""

from __future__ import annotations

import datetime as _dt
import os
import sys

import torch
import torch.distributed as dist


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _print_env() -> None:
    names = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "DS4_200G_IFNAME",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
        "VLLM_HOST_IP",
        "NCCL_NET",
        "NCCL_IB_HCA",
        "NCCL_IB_DISABLE",
    ]
    for name in names:
        print(f"{name}={_env(name, '<unset>')}", file=sys.stderr)


def main() -> int:
    rank = int(_env("RANK"))
    world_size = int(_env("WORLD_SIZE"))
    master_addr = _env("MASTER_ADDR")
    master_port = _env("MASTER_PORT")
    timeout_s = int(_env("DS4_NCCL_PREFLIGHT_TIMEOUT", "90"))
    print(
        "DS4 NCCL preflight starting "
        f"rank={rank}/{world_size} endpoint={master_addr}:{master_port}",
        file=sys.stderr,
    )
    _print_env()
    if not torch.cuda.is_available():
        print("DS4 NCCL preflight failed: CUDA is not available", file=sys.stderr)
        return 65
    try:
        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=_dt.timedelta(seconds=timeout_s),
        )
        value = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        expected = float((world_size * (world_size + 1)) // 2)
        actual = float(value.item())
        if actual != expected:
            print(
                "DS4 NCCL preflight failed: "
                f"all_reduce sum {actual} != expected {expected}",
                file=sys.stderr,
            )
            return 66
        print(f"DS4 NCCL preflight passed on rank {rank}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"DS4 NCCL preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        _print_env()
        return 67
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
