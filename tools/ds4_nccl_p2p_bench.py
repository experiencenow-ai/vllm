#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Self-contained DS4 NCCL/PyNCCL point-to-point benchmark.

Run one process per Spark rank.  This intentionally does not start vLLM serve:
it isolates the PP boundary transport candidates used by DS4 pipelines.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _verbose() -> bool:
    return _bool_env("DS4_NCCL_P2P_BENCH_VERBOSE", False)


def _phase(rank: int, text: str) -> None:
    if _verbose():
        print(f"DS4 NCCL P2P bench rank={rank} {text}", file=sys.stderr, flush=True)


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in _env(name, default).split(",") if item.strip()]


def _int_csv(name: str, default: str) -> list[int]:
    return [int(item) for item in _csv(name, default)]


def _parse_pairs(world_size: int) -> list[tuple[int, int]]:
    raw = _env(
        "DS4_NCCL_P2P_BENCH_PAIRS",
        ";".join(f"{rank}-{rank + 1}" for rank in range(world_size - 1)),
    )
    pairs: list[tuple[int, int]] = []
    for item in raw.split(";"):
        text = item.strip()
        if not text:
            continue
        if "-" in text:
            left, right = text.split("-", 1)
        elif ":" in text:
            left, right = text.split(":", 1)
        else:
            left, right = text.split(",", 1)
        src = int(left.strip())
        dst = int(right.strip())
        if src == dst:
            raise ValueError(f"invalid self pair {item!r}")
        if src < 0 or src >= world_size or dst < 0 or dst >= world_size:
            raise ValueError(f"pair {item!r} outside WORLD_SIZE={world_size}")
        pairs.append((src, dst))
    if not pairs:
        raise ValueError("DS4_NCCL_P2P_BENCH_PAIRS produced no pairs")
    return pairs


def _dtype() -> torch.dtype:
    name = _env("DS4_NCCL_P2P_BENCH_DTYPE", "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"uint8", "u8"}:
        return torch.uint8
    raise ValueError(f"unsupported DS4_NCCL_P2P_BENCH_DTYPE={name!r}")


def _chunks(tensor: torch.Tensor, stripes: int) -> list[torch.Tensor]:
    stripes = min(max(1, stripes), tensor.numel())
    if stripes <= 1:
        return [tensor]
    flat = tensor.view(-1)
    base = flat.numel() // stripes
    rem = flat.numel() % stripes
    out: list[torch.Tensor] = []
    offset = 0
    for index in range(stripes):
        length = base + (1 if index < rem else 0)
        if length > 0:
            out.append(flat.narrow(0, offset, length))
        offset += length
    return out


def _torch_exchange(
    *,
    rank: int,
    src: int,
    dst: int,
    send: torch.Tensor,
    recv: torch.Tensor,
    stripes: int,
    bidirectional: bool,
) -> None:
    ops: list[dist.P2POp] = []
    if bidirectional and rank == src:
        for send_chunk, recv_chunk in zip(_chunks(send, stripes), _chunks(recv, stripes)):
            ops.append(dist.P2POp(dist.isend, send_chunk, dst))
            ops.append(dist.P2POp(dist.irecv, recv_chunk, dst))
    elif bidirectional:
        for send_chunk, recv_chunk in zip(_chunks(send, stripes), _chunks(recv, stripes)):
            ops.append(dist.P2POp(dist.irecv, recv_chunk, src))
            ops.append(dist.P2POp(dist.isend, send_chunk, src))
    elif rank == src:
        for send_chunk in _chunks(send, stripes):
            ops.append(dist.P2POp(dist.isend, send_chunk, dst))
    else:
        for recv_chunk in _chunks(recv, stripes):
            ops.append(dist.P2POp(dist.irecv, recv_chunk, src))
    for req in dist.batch_isend_irecv(ops):
        req.wait()


def _pynccl_exchange(
    *,
    rank: int,
    src: int,
    send: torch.Tensor,
    recv: torch.Tensor,
    send_credit: torch.Tensor | None,
    recv_credit: torch.Tensor | None,
    stripes: int,
    bidirectional: bool,
    credit: bool,
    communicator,
) -> None:
    group_rank = 0 if rank == src else 1
    peer = 1 - group_rank
    communicator.group_start()
    if bidirectional or rank == src:
        for send_chunk in _chunks(send, stripes):
            communicator.send(send_chunk, peer)
    if bidirectional or rank != src:
        for recv_chunk in _chunks(recv, stripes):
            communicator.recv(recv_chunk, peer)
    if credit and not bidirectional:
        assert send_credit is not None
        assert recv_credit is not None
        if rank == src:
            communicator.recv(recv_credit, peer)
        else:
            communicator.send(send_credit, peer)
    communicator.group_end()


def _striped_exchange(
    *,
    rank: int,
    src: int,
    send: torch.Tensor,
    recv: torch.Tensor,
    bidirectional: bool,
    channel,
) -> None:
    group_rank = 0 if rank == src else 1
    peer = 1 - group_rank
    handles = []
    if bidirectional or rank == src:
        handles.append(channel.send(send, peer))
    if bidirectional or rank != src:
        handles.append(channel.recv(recv, peer))
    for handle in handles:
        handle.wait()


def _validate_recv(
    *,
    method: str,
    rank: int,
    src: int,
    dst: int,
    recv: torch.Tensor,
    bidirectional: bool,
) -> None:
    if not (rank == dst or bidirectional):
        return
    expected_rank = dst if rank == src else src
    expected = float(expected_rank + 1)
    head = float(recv[0].item())
    tail = float(recv[-1].item())
    if head != expected or tail != expected:
        raise RuntimeError(
            f"{method} pair={src}-{dst} rank={rank} recv {head}/{tail} "
            f"!= expected {expected}"
        )


def _measure(
    *,
    method: str,
    exchange: Callable[[], None],
    rank: int,
    src: int,
    dst: int,
    actual_bytes: int,
    warmup: int,
    iters: int,
    bidirectional: bool,
    recv: torch.Tensor,
) -> dict[str, object]:
    for _ in range(warmup):
        exchange()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        exchange()
    torch.cuda.synchronize()
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    direction_factor = 2 if bidirectional else 1
    gbps = ((actual_bytes * direction_factor * iters) / elapsed_s) / 1e9
    _validate_recv(
        method=method,
        rank=rank,
        src=src,
        dst=dst,
        recv=recv,
        bidirectional=bidirectional,
    )
    return {
        "method": method,
        "pair": f"{src}-{dst}",
        "rank": rank,
        "bytes": actual_bytes,
        "iters": iters,
        "warmup": warmup,
        "elapsed_s": round(elapsed_s, 6),
        "GBps": round(gbps, 6),
        "Gbit_s": round(gbps * 8.0, 6),
        "direction": "bidirectional" if bidirectional else "unidirectional",
    }


def _make_pair_cpu_group(src: int, dst: int):
    return dist.new_group(ranks=[src, dst], backend="gloo")


def _make_pynccl(cpu_group):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    return PyNcclCommunicator(cpu_group, torch.device("cuda:0"))


def _make_striped_channel(cpu_group, stripes: int):
    from vllm.distributed.ds4_high_speed_channel import Ds4StripedNcclTensorChannel

    return Ds4StripedNcclTensorChannel(
        cpu_group=cpu_group,
        device=torch.device("cuda:0"),
        stripe_count=stripes,
        min_striped_bytes=0,
        use_independent_streams=_bool_env(
            "DS4_NCCL_P2P_BENCH_STRIPED_STREAMS", True
        ),
    )


def _bench_pair(
    rank: int,
    src: int,
    dst: int,
    methods: list[str],
    byte_sizes: list[int],
    dtype: torch.dtype,
    control_group: ProcessGroup,
) -> None:
    active = rank in {src, dst}
    _phase(rank, f"pair={src}-{dst} create gloo pair group")
    cpu_group = _make_pair_cpu_group(src, dst)
    communicators: dict[str, object] = {}
    dist.barrier(group=control_group)
    try:
        if not active:
            return
        bidirectional = _env(
            "DS4_NCCL_P2P_BENCH_DIRECTION", "unidirectional"
        ).lower() in {"bidirectional", "bidir"}
        iters = max(1, int(_env("DS4_NCCL_P2P_BENCH_ITERS", "20")))
        warmup = max(0, int(_env("DS4_NCCL_P2P_BENCH_WARMUP", "5")))
        stripes = max(1, int(_env("DS4_NCCL_P2P_BENCH_STRIPES", "8")))
        credit = _bool_env("DS4_NCCL_P2P_BENCH_CREDIT", False)
        for byte_size in byte_sizes:
            numel = max(1, byte_size // torch.empty((), dtype=dtype).element_size())
            send = torch.full((numel,), rank + 1, dtype=dtype, device="cuda")
            recv = torch.empty_like(send)
            send_credit = torch.full((1,), rank + 1, dtype=torch.uint8, device="cuda")
            recv_credit = torch.empty_like(send_credit)
            actual_bytes = numel * send.element_size()
            for method in methods:
                recv.fill_(0)
                if method == "torch":
                    exchange = lambda: _torch_exchange(
                        rank=rank,
                        src=src,
                        dst=dst,
                        send=send,
                        recv=recv,
                        stripes=stripes,
                        bidirectional=bidirectional,
                    )
                elif method == "pynccl":
                    if "pynccl" not in communicators:
                        _phase(rank, f"pair={src}-{dst} create pynccl communicator")
                        communicators["pynccl"] = _make_pynccl(cpu_group)
                    exchange = lambda: _pynccl_exchange(
                        rank=rank,
                        src=src,
                        send=send,
                        recv=recv,
                        send_credit=send_credit,
                        recv_credit=recv_credit,
                        stripes=stripes,
                        bidirectional=bidirectional,
                        credit=credit,
                        communicator=communicators["pynccl"],
                    )
                elif method == "striped":
                    if "striped" not in communicators:
                        _phase(rank, f"pair={src}-{dst} create striped communicator")
                        communicators["striped"] = _make_striped_channel(
                            cpu_group,
                            max(1, int(_env("DS4_NCCL_P2P_BENCH_STRIPES", "8"))),
                        )
                    exchange = lambda: _striped_exchange(
                        rank=rank,
                        src=src,
                        send=send,
                        recv=recv,
                        bidirectional=bidirectional,
                        channel=communicators["striped"],
                    )
                else:
                    raise ValueError(f"unknown method {method!r}")
                _phase(rank, f"pair={src}-{dst} method={method} bytes={actual_bytes} begin")
                row = _measure(
                    method=method,
                    exchange=exchange,
                    rank=rank,
                    src=src,
                    dst=dst,
                    actual_bytes=actual_bytes,
                    warmup=warmup,
                    iters=iters,
                    bidirectional=bidirectional,
                    recv=recv,
                )
                row["dtype"] = str(dtype).replace("torch.", "")
                row["stripes"] = stripes
                row["credit"] = credit
                print(json.dumps(row, sort_keys=True), flush=True)
                _phase(rank, f"pair={src}-{dst} method={method} bytes={actual_bytes} done")
    finally:
        for value in communicators.values():
            destroy = getattr(value, "destroy", None)
            if destroy is not None:
                destroy()
        dist.barrier(group=control_group)


def main() -> int:
    rank = int(_env("RANK"))
    world_size = int(_env("WORLD_SIZE"))
    master_addr = _env("MASTER_ADDR")
    master_port = _env("MASTER_PORT")
    timeout_s = int(_env("DS4_NCCL_P2P_BENCH_TIMEOUT", "180"))
    torch.cuda.set_device(0)
    print(
        "DS4 NCCL P2P bench start "
        f"rank={rank}/{world_size} endpoint={master_addr}:{master_port} "
        f"time={dt.datetime.now(dt.UTC).isoformat()}",
        file=sys.stderr,
        flush=True,
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(seconds=timeout_s),
    )
    try:
        control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        value = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        methods = _csv("DS4_NCCL_P2P_BENCH_METHODS", "torch,pynccl,striped")
        byte_sizes = _int_csv(
            "DS4_NCCL_P2P_BENCH_BYTES_LIST",
            _env("DS4_NCCL_P2P_BENCH_BYTES", "1048576,16777216,67108864"),
        )
        dtype = _dtype()
        for src, dst in _parse_pairs(world_size):
            _bench_pair(rank, src, dst, methods, byte_sizes, dtype, control_group)
            dist.barrier(group=control_group)
        print(f"DS4 NCCL P2P bench passed rank={rank}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(
            f"DS4 NCCL P2P bench failed rank={rank}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 67
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
