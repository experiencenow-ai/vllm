#!/usr/bin/env python3
"""Fail-fast NCCL fabric preflight for DS4 Spark launchers."""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _nccl_barrier(group=None) -> None:
    # Each Spark rank owns local CUDA device 0.  Letting ProcessGroupNCCL guess
    # from the global rank can select non-existent devices and hang the guard.
    dist.barrier(group=group, device_ids=[0])


def _store_barrier(rank: int, world_size: int, label: str) -> None:
    store = dist.distributed_c10d._get_default_store()
    prefix = f"ds4_nccl_preflight/{label}"
    store.set(f"{prefix}/{rank}", b"1")
    keys = [f"{prefix}/{idx}" for idx in range(world_size)]
    timeout_s = int(
        _env(
            "DS4_NCCL_PREFLIGHT_STORE_BARRIER_TIMEOUT",
            _env("DS4_NCCL_PREFLIGHT_TIMEOUT", "90"),
        )
    )
    try:
        store.wait(keys, _dt.timedelta(seconds=timeout_s))
    except TypeError:
        store.wait(keys)


def _print_env() -> None:
    names = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "DS4_NCCL_PREFLIGHT_BACKEND",
        "DS4_200G_IFNAME",
        "DS4_200G_NCCL_IFNAME",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
        "VLLM_HOST_IP",
        "NCCL_NET",
        "NCCL_ALGO",
        "NCCL_IB_HCA",
        "NCCL_IB_DISABLE",
        "DS4_200G_VERIFIED_ROUTED_LOOPBACK_NCCL",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "DS4_NCCL_PREFLIGHT_BENCH_BYTES",
        "DS4_NCCL_PREFLIGHT_BENCH_ITERS",
        "DS4_NCCL_PREFLIGHT_MIN_BUSBW_GBPS",
        "DS4_NCCL_PREFLIGHT_WARN_BUSBW_GBPS",
        "DS4_NCCL_PREFLIGHT_GROUPS",
        "DS4_NCCL_PREFLIGHT_P2P_PAIRS",
        "DS4_NCCL_PREFLIGHT_P2P_METHOD",
        "DS4_NCCL_PREFLIGHT_MIN_P2P_GBPS",
        "DS4_NCCL_PREFLIGHT_WARN_P2P_GBPS",
    ]
    for name in names:
        print(f"{name}={_env(name, '<unset>')}", file=sys.stderr)


def _run_bandwidth_probe(
    rank: int,
    world_size: int,
    *,
    group=None,
    label: str = "world",
) -> int:
    bench_bytes = int(_env("DS4_NCCL_PREFLIGHT_BENCH_BYTES", "0"))
    if bench_bytes <= 0:
        return 0
    bench_iters = max(1, int(_env("DS4_NCCL_PREFLIGHT_BENCH_ITERS", "3")))
    min_busbw_gbps = float(_env("DS4_NCCL_PREFLIGHT_MIN_BUSBW_GBPS", "0"))
    warn_busbw_gbps = float(_env("DS4_NCCL_PREFLIGHT_WARN_BUSBW_GBPS", "0"))
    element_size = torch.empty((), dtype=torch.float32).element_size()
    numel = max(1, bench_bytes // element_size)
    actual_bytes = (numel * element_size)
    buf = torch.full((numel,), float(rank + 1), dtype=torch.float32, device="cuda")
    print(
        "DS4 NCCL preflight bandwidth begin: "
        f"group={label} bytes={actual_bytes} iters={bench_iters} "
        f"min_busbw_GBps={min_busbw_gbps:.3f} "
        f"warn_busbw_GBps={warn_busbw_gbps:.3f}",
        file=sys.stderr,
    )
    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(bench_iters):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    algbw_gbps = ((actual_bytes * bench_iters) / elapsed_s) / 1e9
    bus_factor = ((2.0 * (world_size - 1)) / world_size) if world_size > 1 else 1.0
    busbw_gbps = (algbw_gbps * bus_factor)
    print(
        "DS4 NCCL preflight bandwidth: "
        f"group={label} bytes={actual_bytes} iters={bench_iters} "
        f"elapsed_s={elapsed_s:.6f} "
        f"algbw_GBps={algbw_gbps:.3f} busbw_GBps={busbw_gbps:.3f} "
        f"busbw_Gbit_s={(busbw_gbps * 8.0):.3f} "
        f"min_busbw_GBps={min_busbw_gbps:.3f} "
        f"warn_busbw_GBps={warn_busbw_gbps:.3f}",
        file=sys.stderr,
    )
    if warn_busbw_gbps > 0 and busbw_gbps < warn_busbw_gbps:
        print(
            "WARNING: DS4 NCCL preflight below warning threshold: "
            f"group={label} measured {busbw_gbps:.3f} GB/s "
            f"({(busbw_gbps * 8.0):.3f} Gbit/s) < warning "
            f"{warn_busbw_gbps:.3f} GB/s "
            f"({(warn_busbw_gbps * 8.0):.3f} Gbit/s); launch continues unless "
            "the fail threshold is crossed",
            file=sys.stderr,
        )
    if min_busbw_gbps > 0 and busbw_gbps < min_busbw_gbps:
        print(
            "DS4 NCCL preflight failed: "
            f"measured busbw {busbw_gbps:.3f} GB/s < required {min_busbw_gbps:.3f} GB/s",
            file=sys.stderr,
        )
        return 68
    return 0


def _parse_rank_groups(world_size: int) -> list[list[int]]:
    raw = _env("DS4_NCCL_PREFLIGHT_GROUPS", "")
    if not raw:
        if world_size % 2 != 0:
            raise ValueError(
                "DS4_NCCL_PREFLIGHT_GROUPS is required for odd WORLD_SIZE"
            )
        return [[rank, rank + 1] for rank in range(0, world_size, 2)]
    groups: list[list[int]] = []
    for item in raw.split(";"):
        if not item.strip():
            continue
        group = [int(rank) for rank in item.split(",") if rank.strip()]
        if len(group) < 2:
            raise ValueError(f"invalid rank group {item!r}")
        for rank in group:
            if rank < 0 or rank >= world_size:
                raise ValueError(f"rank {rank} outside WORLD_SIZE={world_size}")
        groups.append(group)
    if not groups:
        raise ValueError("DS4_NCCL_PREFLIGHT_GROUPS did not contain any groups")
    return groups


def _parse_p2p_pairs(world_size: int) -> list[tuple[int, int]]:
    raw = _env("DS4_NCCL_PREFLIGHT_P2P_PAIRS", "")
    if not raw:
        return [(rank, rank + 1) for rank in range(world_size - 1)]
    pairs: list[tuple[int, int]] = []
    for item in raw.split(";"):
        text = item.strip()
        if not text:
            continue
        if "-" in text:
            left, right = text.split("-", 1)
        elif ":" in text:
            left, right = text.split(":", 1)
        elif "," in text:
            left, right = text.split(",", 1)
        else:
            raise ValueError(f"invalid P2P pair {item!r}")
        src = int(left.strip())
        dst = int(right.strip())
        if src == dst:
            raise ValueError(f"invalid self P2P pair {item!r}")
        for rank in (src, dst):
            if rank < 0 or rank >= world_size:
                raise ValueError(f"rank {rank} outside WORLD_SIZE={world_size}")
        pairs.append((src, dst))
    if not pairs:
        raise ValueError("DS4_NCCL_PREFLIGHT_P2P_PAIRS did not contain any pairs")
    return pairs


def _rank_loopback_ip(rank: int) -> str:
    raw = _env("DS4_NCCL_PREFLIGHT_RANK_IPS", "")
    if raw:
        parts = [part.strip() for part in raw.replace("|", ",").split(",")]
        parts = [part for part in parts if part]
        if len(parts) > rank:
            return parts[rank]
    return f"10.10.100.{10 + rank}"


def _route_dev_for_rank(peer_rank: int) -> str:
    peer_ip = _rank_loopback_ip(peer_rank)
    route = subprocess.check_output(
        ["ip", "route", "get", peer_ip],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    parts = route.split()
    for index, item in enumerate(parts):
        if item == "dev" and index + 1 < len(parts):
            return parts[index + 1]
    raise RuntimeError(f"could not find route device for rank {peer_rank} ip {peer_ip}")


def _pp_edge_dev_for_rank(rank: int, peer_rank: int) -> tuple[str, str]:
    rail = _env(
        "DS4_NCCL_PREFLIGHT_PP_EDGE_RAIL",
        _env("VLLM_DS4_PP_EDGE_RAIL", "enp"),
    ).strip()
    if rail in {"", "route", "routed", "loopback"} or abs(rank - peer_rank) != 1:
        return (_route_dev_for_rank(peer_rank), rail or "route")
    left_side = rank < peer_rank
    if rail in {"enp", "rail0", "lower", "odd"}:
        return ("enp1s0f1np1" if left_side else "enp1s0f0np0", rail)
    if rail in {"enP2p", "enp2p", "rail1", "upper", "even"}:
        return ("enP2p1s0f1np1" if left_side else "enP2p1s0f0np0", rail)
    return (rail, rail)


def _new_nccl_pair_group(src: int, dst: int):
    # Create every adjacent pair group collectively across the world.  The
    # local-synchronization shortcut can hang once the probe advances from
    # pair 0-1 to the overlapping pair 1-2 on this Spark topology.
    return dist.new_group(ranks=[src, dst], backend="nccl")


def _make_torch_pair_group(rank: int, src: int, dst: int):
    active = rank in {src, dst}
    peer = dst if rank == src else src
    route_ifname = ""
    edge_rail = _env(
        "DS4_NCCL_PREFLIGHT_PP_EDGE_RAIL",
        _env("VLLM_DS4_PP_EDGE_RAIL", "enp"),
    ).strip() or "enp"
    if active:
        route_ifname, edge_rail = _pp_edge_dev_for_rank(rank, peer)
    original_ifname = os.environ.get("NCCL_SOCKET_IFNAME")
    try:
        if active and route_ifname:
            os.environ["NCCL_SOCKET_IFNAME"] = route_ifname
        pair_group = _new_nccl_pair_group(src, dst)
        if not active:
            print(
                "DS4 NCCL P2P preflight torch pair group created on "
                f"non-member rank: pair={src}-{dst} rank={rank} "
                f"edge_rail={edge_rail}",
                file=sys.stderr,
            )
            if pair_group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(pair_group)
            return None
        send = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        recv = torch.empty_like(send)
        print(
            "DS4 NCCL P2P preflight torch pair group warmup: "
            f"pair={src}-{dst} rank={rank} route_ifname={route_ifname} "
            f"edge_rail={edge_rail} "
            f"pair_nccl_ifname={os.environ.get('NCCL_SOCKET_IFNAME', '<unset>')}",
            file=sys.stderr,
        )
        if rank == src:
            ops = [
                dist.P2POp(dist.isend, send, dst, pair_group),
                dist.P2POp(dist.irecv, recv, dst, pair_group),
            ]
        else:
            ops = [
                dist.P2POp(dist.irecv, recv, src, pair_group),
                dist.P2POp(dist.isend, send, src, pair_group),
            ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        torch.cuda.synchronize()
        expected = float(peer + 1)
        actual = float(recv[0].item())
        if actual != expected:
            raise RuntimeError(
                "DS4 NCCL P2P preflight torch pair warmup failed: "
                f"pair={src}-{dst} rank={rank} recv {actual} != expected {expected}"
            )
        return pair_group
    finally:
        if original_ifname is None:
            os.environ.pop("NCCL_SOCKET_IFNAME", None)
        else:
            os.environ["NCCL_SOCKET_IFNAME"] = original_ifname


def _run_p2p_pair_probe(
    rank: int,
    src: int,
    dst: int,
    *,
    cpu_group=None,
    torch_group=None,
) -> int:
    bench_bytes = int(_env("DS4_NCCL_PREFLIGHT_BENCH_BYTES", "0"))
    if bench_bytes <= 0:
        return 0
    bench_iters = max(1, int(_env("DS4_NCCL_PREFLIGHT_BENCH_ITERS", "3")))
    stripes = max(1, int(_env("DS4_NCCL_PREFLIGHT_P2P_STRIPES", "1")))
    method = _env("DS4_NCCL_PREFLIGHT_P2P_METHOD", "pynccl").lower()
    if method not in {"pynccl", "torch"}:
        raise ValueError("DS4_NCCL_PREFLIGHT_P2P_METHOD must be pynccl or torch")
    direction = _env("DS4_NCCL_PREFLIGHT_P2P_DIRECTION", "unidirectional").lower()
    if direction not in {"unidirectional", "oneway", "bidirectional", "bidir"}:
        raise ValueError(
            "DS4_NCCL_PREFLIGHT_P2P_DIRECTION must be unidirectional or bidirectional"
        )
    bidirectional = direction in {"bidirectional", "bidir"}
    credit = _bool_env("DS4_NCCL_PREFLIGHT_P2P_CREDIT", True)
    min_p2p_gbps = float(
        _env(
            "DS4_NCCL_PREFLIGHT_MIN_P2P_GBPS",
            _env("DS4_NCCL_PREFLIGHT_MIN_BUSBW_GBPS", "0"),
        )
    )
    warn_p2p_gbps = float(_env("DS4_NCCL_PREFLIGHT_WARN_P2P_GBPS", "0"))
    element_size = torch.empty((), dtype=torch.float32).element_size()
    numel = max(1, bench_bytes // element_size)
    actual_bytes = (numel * element_size)
    if rank not in {src, dst}:
        return 0
    peer = dst if rank == src else src
    send = torch.full((numel,), float(rank + 1), dtype=torch.float32, device="cuda")
    recv = torch.empty_like(send)
    send_credit = torch.full((1,), rank + 1, dtype=torch.uint8, device="cuda")
    recv_credit = torch.empty_like(send_credit)
    send_chunks = _split_p2p_tensor(send, stripes)
    recv_chunks = _split_p2p_tensor(recv, stripes)
    communicator = None

    def torch_exchange() -> None:
        ops = []
        if bidirectional and rank == src:
            for send_chunk, recv_chunk in zip(send_chunks, recv_chunks):
                ops.append(dist.P2POp(dist.isend, send_chunk, dst, torch_group))
                ops.append(dist.P2POp(dist.irecv, recv_chunk, dst, torch_group))
        elif bidirectional:
            for send_chunk, recv_chunk in zip(send_chunks, recv_chunks):
                ops.append(dist.P2POp(dist.irecv, recv_chunk, src, torch_group))
                ops.append(dist.P2POp(dist.isend, send_chunk, src, torch_group))
        elif rank == src:
            for send_chunk in send_chunks:
                ops.append(dist.P2POp(dist.isend, send_chunk, dst, torch_group))
        else:
            for recv_chunk in recv_chunks:
                ops.append(dist.P2POp(dist.irecv, recv_chunk, src, torch_group))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    def pynccl_exchange() -> None:
        assert communicator is not None
        group_rank = 0 if rank == src else 1
        peer_rank = 1 - group_rank
        communicator.group_start()
        if bidirectional or rank == src:
            for send_chunk in send_chunks:
                communicator.send(send_chunk, peer_rank)
        if bidirectional or rank != src:
            for recv_chunk in recv_chunks:
                communicator.recv(recv_chunk, peer_rank)
        if credit and not bidirectional:
            if rank == src:
                communicator.recv(recv_credit, peer_rank)
            else:
                communicator.send(send_credit, peer_rank)
        communicator.group_end()

    if method == "pynccl":
        if cpu_group is None:
            raise RuntimeError("PyNCCL P2P preflight requires a Gloo CPU pair group")
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        communicator = PyNcclCommunicator(cpu_group, torch.device("cuda:0"))
        exchange = pynccl_exchange
    else:
        if torch_group is None:
            raise RuntimeError("torch P2P preflight requires an NCCL pair group")
        exchange = torch_exchange

    print(
        "DS4 NCCL P2P preflight bandwidth begin: "
        f"pair={src}-{dst} local_rank={rank} peer={peer} "
        f"bytes={actual_bytes} iters={bench_iters} stripes={len(send_chunks)} "
        f"method={method} credit={1 if credit else 0} "
        f"direction={'bidirectional' if bidirectional else 'unidirectional'} "
        f"min_p2p_GBps={min_p2p_gbps:.3f} "
        f"warn_p2p_GBps={warn_p2p_gbps:.3f}",
        file=sys.stderr,
    )
    exchange()
    torch.cuda.synchronize()
    expected = float(peer + 1)
    if rank == dst or bidirectional:
        actual = float(recv[0].item())
        actual_tail = float(recv[-1].item())
    else:
        actual = expected
        actual_tail = expected
    if actual != expected or actual_tail != expected:
        print(
            "DS4 NCCL P2P preflight failed: "
            f"pair={src}-{dst} rank={rank} recv head/tail "
            f"{actual}/{actual_tail} != expected {expected}",
            file=sys.stderr,
        )
        return 66
    start = time.perf_counter()
    for _ in range(bench_iters):
        exchange()
    torch.cuda.synchronize()
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    direction_factor = 2 if bidirectional else 1
    measured_gbps = ((actual_bytes * direction_factor * bench_iters) / elapsed_s) / 1e9
    print(
        "DS4 NCCL P2P preflight bandwidth: "
        f"pair={src}-{dst} local_rank={rank} bytes={actual_bytes} "
        f"iters={bench_iters} stripes={len(send_chunks)} "
        f"direction={'bidirectional' if bidirectional else 'unidirectional'} "
        f"elapsed_s={elapsed_s:.6f} "
        f"p2p_GBps={measured_gbps:.3f} "
        f"p2p_Gbit_s={(measured_gbps * 8.0):.3f} "
        f"min_p2p_GBps={min_p2p_gbps:.3f} "
        f"warn_p2p_GBps={warn_p2p_gbps:.3f}",
        file=sys.stderr,
    )
    if warn_p2p_gbps > 0 and measured_gbps < warn_p2p_gbps:
        print(
            "WARNING: DS4 NCCL P2P preflight below warning threshold: "
            f"pair={src}-{dst} rank={rank} measured "
            f"{measured_gbps:.3f} GB/s ({(measured_gbps * 8.0):.3f} Gbit/s) "
            f"< warning {warn_p2p_gbps:.3f} GB/s "
            f"({(warn_p2p_gbps * 8.0):.3f} Gbit/s); launch continues unless "
            "the fail threshold is crossed",
            file=sys.stderr,
        )
    if min_p2p_gbps > 0 and measured_gbps < min_p2p_gbps:
        print(
            "DS4 NCCL P2P preflight failed: "
            f"pair={src}-{dst} rank={rank} measured "
            f"{measured_gbps:.3f} GB/s < required {min_p2p_gbps:.3f} GB/s",
            file=sys.stderr,
        )
        return 68
    if communicator is not None:
        destroy = getattr(communicator, "destroy", None)
        if destroy is not None:
            destroy()
    return 0


def _split_p2p_tensor(tensor: torch.Tensor, stripes: int) -> list[torch.Tensor]:
    stripes = min(max(1, stripes), tensor.numel())
    if stripes <= 1:
        return [tensor]
    base = tensor.numel() // stripes
    rem = tensor.numel() % stripes
    chunks: list[torch.Tensor] = []
    offset = 0
    for index in range(stripes):
        length = base + (1 if index < rem else 0)
        if length > 0:
            chunks.append(tensor.narrow(0, offset, length))
        offset += length
    return chunks


def _run_p2p_nccl_preflight(rank: int, world_size: int) -> int:
    pairs = _parse_p2p_pairs(world_size)
    method = _env("DS4_NCCL_PREFLIGHT_P2P_METHOD", "pynccl").lower()
    torch.cuda.set_device(0)
    print(
        "DS4 NCCL P2P preflight pairs: "
        + ";".join(f"{src}-{dst}" for src, dst in pairs),
        file=sys.stderr,
    )
    skip_world_all_reduce = _bool_env(
        "DS4_NCCL_PREFLIGHT_P2P_SKIP_WORLD_ALLREDUCE",
        False,
    )
    if skip_world_all_reduce:
        print(
            "DS4 NCCL P2P preflight stage: communicator all_reduce skipped "
            "by DS4_NCCL_PREFLIGHT_P2P_SKIP_WORLD_ALLREDUCE=1",
            file=sys.stderr,
        )
    else:
        value_device = "cuda" if dist.get_backend() == "nccl" else "cpu"
        value = torch.tensor([rank + 1], dtype=torch.float32, device=value_device)
        print("DS4 NCCL P2P preflight stage: communicator all_reduce begin", file=sys.stderr)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if value_device == "cuda":
            torch.cuda.synchronize()
        print("DS4 NCCL P2P preflight stage: communicator all_reduce complete", file=sys.stderr)
        expected = float((world_size * (world_size + 1)) // 2)
        actual = float(value.item())
        if actual != expected:
            print(
                "DS4 NCCL P2P preflight failed: "
                f"all_reduce sum {actual} != expected {expected}",
                file=sys.stderr,
            )
            return 66
    rank_groups = _env("DS4_NCCL_PREFLIGHT_GROUPS", "")
    if rank_groups and rank_groups != "<unused>":
        print(
            "DS4 NCCL P2P preflight stage: pairwise NCCL group probes begin",
            file=sys.stderr,
        )
        status = _run_pairwise_nccl_preflight(rank, world_size)
        if status != 0:
            return status
        print(
            "DS4 NCCL P2P preflight stage: pairwise NCCL group probes complete",
            file=sys.stderr,
        )
    for pair_index, (src, dst) in enumerate(pairs):
        print(
            f"DS4 NCCL P2P preflight pair begin: pair={src}-{dst} method={method}",
            file=sys.stderr,
        )
        _store_barrier(rank, world_size, f"p2p-{pair_index}-before-group")
        cpu_group = None
        torch_group = None
        if method == "pynccl":
            cpu_group = dist.new_group(ranks=[src, dst], backend="gloo")
        elif method == "torch":
            torch_group = _make_torch_pair_group(rank, src, dst)
        else:
            raise ValueError("DS4_NCCL_PREFLIGHT_P2P_METHOD must be pynccl or torch")
        _store_barrier(rank, world_size, f"p2p-{pair_index}-pre")
        try:
            status = _run_p2p_pair_probe(
                rank,
                src,
                dst,
                cpu_group=cpu_group,
                torch_group=torch_group,
            )
        finally:
            if cpu_group is not None and cpu_group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(cpu_group)
            if torch_group is not None and torch_group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(torch_group)
        _store_barrier(rank, world_size, f"p2p-{pair_index}-post")
        if status != 0:
            return status
        print(
            f"DS4 NCCL P2P preflight pair complete: pair={src}-{dst} method={method}",
            file=sys.stderr,
        )
    print(f"DS4 NCCL P2P preflight passed on rank {rank}", file=sys.stderr)
    return 0


def _run_pairwise_nccl_preflight(rank: int, world_size: int) -> int:
    groups = _parse_rank_groups(world_size)
    torch.cuda.set_device(0)
    print(
        "DS4 NCCL pairwise preflight groups: "
        + ";".join(",".join(str(rank) for rank in group) for group in groups),
        file=sys.stderr,
    )
    active_group = None
    active_ranks = None
    process_groups = []
    try:
        for ranks in groups:
            pg = dist.new_group(ranks=ranks, backend="nccl")
            if rank in ranks:
                process_groups.append(pg)
                active_group = pg
                active_ranks = ranks
        if active_group is None or active_ranks is None:
            print(
                f"DS4 NCCL pairwise preflight: rank {rank} is not in any NCCL group",
                file=sys.stderr,
            )
            _store_barrier(rank, world_size, "pairwise-no-active-group")
            return 0
        group_rank = active_ranks.index(rank)
        value = torch.tensor([group_rank + 1], dtype=torch.float32, device="cuda")
        label = ",".join(str(item) for item in active_ranks)
        print(
            f"DS4 NCCL pairwise preflight all_reduce begin: group={label}",
            file=sys.stderr,
        )
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=active_group)
        torch.cuda.synchronize()
        expected = float((len(active_ranks) * (len(active_ranks) + 1)) // 2)
        actual = float(value.item())
        if actual != expected:
            print(
                "DS4 NCCL pairwise preflight failed: "
                f"group={label} all_reduce sum {actual} != expected {expected}",
                file=sys.stderr,
            )
            return 66
        bw_status = _run_bandwidth_probe(
            group_rank,
            len(active_ranks),
            group=active_group,
            label=label,
        )
        if bw_status != 0:
            return bw_status
        _store_barrier(rank, world_size, "pairwise-complete")
        print(
            f"DS4 NCCL pairwise preflight passed on rank {rank}: group={label}",
            file=sys.stderr,
        )
        return 0
    finally:
        for pg in process_groups:
            dist.destroy_process_group(pg)


def main() -> int:
    rank = int(_env("RANK"))
    world_size = int(_env("WORLD_SIZE"))
    master_addr = _env("MASTER_ADDR")
    master_port = _env("MASTER_PORT")
    timeout_s = int(_env("DS4_NCCL_PREFLIGHT_TIMEOUT", "90"))
    backend = _env("DS4_NCCL_PREFLIGHT_BACKEND", "nccl")
    p2p_method = _env("DS4_NCCL_PREFLIGHT_P2P_METHOD", "pynccl").lower()
    process_group_backend = (
        "gloo"
        if backend in {"tp_pair_nccl", "p2p_nccl"}
        else backend
    )
    if backend not in {"gloo", "nccl", "tp_pair_nccl", "p2p_nccl"}:
        print(
            "DS4 NCCL preflight failed: "
            f"unsupported DS4_NCCL_PREFLIGHT_BACKEND={backend}",
            file=sys.stderr,
        )
        return 64
    print(
        "DS4 NCCL preflight starting "
        f"rank={rank}/{world_size} endpoint={master_addr}:{master_port} "
        f"backend={backend}",
        file=sys.stderr,
    )
    _print_env()
    if backend in {"nccl", "tp_pair_nccl", "p2p_nccl"} and not torch.cuda.is_available():
        print("DS4 NCCL preflight failed: CUDA is not available", file=sys.stderr)
        return 65
    try:
        if backend in {"nccl", "tp_pair_nccl", "p2p_nccl"}:
            torch.cuda.set_device(0)
        print("DS4 NCCL preflight stage: init_process_group begin", file=sys.stderr)
        dist.init_process_group(
            process_group_backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=_dt.timedelta(seconds=timeout_s),
        )
        print("DS4 NCCL preflight stage: init_process_group complete", file=sys.stderr)
        if backend == "tp_pair_nccl":
            return _run_pairwise_nccl_preflight(rank, world_size)
        if backend == "p2p_nccl":
            return _run_p2p_nccl_preflight(rank, world_size)
        device = "cuda" if backend == "nccl" else "cpu"
        value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        print("DS4 NCCL preflight stage: all_reduce begin", file=sys.stderr)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize()
        print("DS4 NCCL preflight stage: all_reduce complete", file=sys.stderr)
        expected = float((world_size * (world_size + 1)) // 2)
        actual = float(value.item())
        if actual != expected:
            print(
                "DS4 NCCL preflight failed: "
                f"all_reduce sum {actual} != expected {expected}",
                file=sys.stderr,
            )
            return 66
        if backend == "nccl":
            bw_status = _run_bandwidth_probe(rank, world_size)
            if bw_status != 0:
                return bw_status
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
