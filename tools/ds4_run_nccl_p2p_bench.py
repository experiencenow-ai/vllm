#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the isolated DS4 NCCL/PyNCCL P2P benchmark on Spark nodes."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys


DEFAULT_NODES = [f"spark{i}" for i in range(8)]


def _nodes(raw: str) -> list[str]:
    nodes = [item.strip() for item in raw.split(",") if item.strip()]
    if not nodes:
        raise SystemExit("no nodes selected")
    return nodes


def _shell_path(path: str) -> str:
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace('"', '\\"') + '"'
    return shlex.quote(path)


def _remote(node: str, command: str, *, dry_run: bool = False) -> int:
    print(f"== {node} == {command}")
    if dry_run:
        return 0
    return subprocess.run(["ssh", node, command], text=True).returncode


def _repo_command(source_root: str, inner: str) -> str:
    return f"cd {_shell_path(source_root)} && {inner}"


def _stop_command(args: argparse.Namespace) -> str:
    py = _shell_path(args.python)
    return _repo_command(
        args.source_root,
        f"{py} tools/ds4_stop_spark_processes.py --local --service {args.stop_service} "
        "--include-benchmarks --term-timeout-s 15 --kill-timeout-s 5",
    )


def _build_command(args: argparse.Namespace) -> str:
    py = _shell_path(args.python)
    return _repo_command(
        args.source_root,
        f"{py} -m py_compile tools/ds4_nccl_p2p_bench.py "
        "vllm/distributed/ds4_high_speed_channel.py "
        "vllm/distributed/device_communicators/pynccl.py",
    )


def _bench_command(args: argparse.Namespace, rank: int, log_tag: str) -> str:
    nccl_ifname = _rank_value(args.nccl_ifnames, rank) or args.nccl_ifname
    gloo_ifname = _rank_value(args.gloo_ifnames, rank) or args.gloo_ifname
    host_ip = _rank_value(args.host_ips, rank)
    env = {
        "RANK": str(rank),
        "WORLD_SIZE": str(args.nnodes),
        "MASTER_ADDR": args.master_addr,
        "MASTER_PORT": args.master_port,
        "NCCL_DEBUG": args.nccl_debug,
        "NCCL_SOCKET_IFNAME": nccl_ifname,
        "GLOO_SOCKET_IFNAME": gloo_ifname,
        "VLLM_HOST_IP": host_ip,
        "DS4_NCCL_P2P_BENCH_PAIRS": args.pairs,
        "DS4_NCCL_P2P_BENCH_METHODS": args.methods,
        "DS4_NCCL_P2P_BENCH_CONTROL_BACKEND": args.control_backend,
        "DS4_NCCL_P2P_BENCH_BYTES_LIST": args.bytes,
        "DS4_NCCL_P2P_BENCH_ITERS": str(args.iters),
        "DS4_NCCL_P2P_BENCH_WARMUP": str(args.warmup),
        "DS4_NCCL_P2P_BENCH_STRIPES": str(args.stripes),
        "DS4_NCCL_P2P_BENCH_DIRECTION": args.direction,
        "DS4_NCCL_P2P_BENCH_DTYPE": args.dtype,
        "DS4_NCCL_P2P_BENCH_CREDIT": "1" if args.pynccl_credit else "0",
        "DS4_NCCL_P2P_BENCH_STRIPED_STREAMS": "1" if args.striped_streams else "0",
        "VLLM_DS4_SKIP_PYNCCL_WARMUP_ALLREDUCE": "1",
    }
    if args.extra_env:
        for item in args.extra_env:
            key, value = item.split("=", 1)
            env[key] = value
    env_text = " ".join(
        f"{shlex.quote(key)}={shlex.quote(value)}"
        for key, value in env.items()
        if value != ""
    )
    py = _shell_path(args.python)
    log = f"~/ds4_logs/nccl_p2p_bench_{log_tag}-rank{rank}.log"
    return _repo_command(
        args.source_root,
        "mkdir -p ~/ds4_logs && "
        f"env {env_text} timeout --kill-after=10s {args.timeout_s}s "
        f"{py} tools/ds4_nccl_p2p_bench.py > {log} 2>&1",
    )


def _rank_value(raw: str, rank: int) -> str:
    if not raw:
        return ""
    values = raw.split("|")
    if len(values) == 1:
        values = raw.split(",")
    if rank >= len(values):
        raise SystemExit(f"rank {rank} has no value in {raw!r}")
    return values[rank].strip()


def _collect(node: str, rank: int, log_tag: str, dry_run: bool) -> None:
    log = f"~/ds4_logs/nccl_p2p_bench_{log_tag}-rank{rank}.log"
    command = f"printf '==== {node} rank={rank} {log}\\n'; cat {log}"
    _remote(node, command, dry_run=dry_run)


def _run_parallel(args: argparse.Namespace, nodes: list[str], log_tag: str) -> None:
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for rank, node in enumerate(nodes):
        command = _bench_command(args, rank, log_tag)
        print(f"== {node} == {command}")
        if args.dry_run:
            continue
        procs.append((node, subprocess.Popen(["ssh", node, command], text=True)))
    failed: list[str] = []
    for node, proc in procs:
        status = proc.wait()
        if status != 0:
            failed.append(f"{node}:{status}")
    if failed:
        raise SystemExit(f"benchmark failed on {', '.join(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--nnodes", type=int, default=8)
    parser.add_argument("--source-root", default="~/src/vllm")
    parser.add_argument("--python", default="~/ds4-vllm-local/bin/python")
    parser.add_argument("--master-addr", default="10.10.100.10")
    parser.add_argument("--master-port", default="30944")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--methods", default="pynccl,striped")
    parser.add_argument("--control-backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--bytes", default="1048576,16777216,67108864")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stripes", type=int, default=8)
    parser.add_argument("--direction", choices=("unidirectional", "bidirectional"), default="unidirectional")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--nccl-ifname", default="ds4ring0")
    parser.add_argument("--nccl-ifnames", default="", help="per-rank NCCL_SOCKET_IFNAME list separated by |")
    parser.add_argument("--gloo-ifname", default="ds4ring0")
    parser.add_argument("--gloo-ifnames", default="", help="per-rank GLOO_SOCKET_IFNAME list separated by |")
    parser.add_argument("--host-ips", default="", help="per-rank VLLM_HOST_IP list separated by |")
    parser.add_argument("--nccl-debug", default="WARN")
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--log-tag", default="")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--stop-service", default="")
    parser.add_argument("--striped-streams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pynccl-credit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nodes = _nodes(args.nodes)
    if len(nodes) != args.nnodes:
        raise SystemExit(f"--nodes has {len(nodes)} entries but --nnodes={args.nnodes}")
    if args.pairs == "":
        args.pairs = ";".join(f"{rank}-{rank + 1}" for rank in range(args.nnodes - 1))
    log_tag = args.log_tag or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.pull:
        for node in nodes:
            status = _remote(node, _repo_command(args.source_root, "git pull --ff-only origin main"), dry_run=args.dry_run)
            if status != 0:
                raise SystemExit(f"pull failed on {node}")
    if args.build:
        for node in nodes:
            status = _remote(node, _build_command(args), dry_run=args.dry_run)
            if status != 0:
                raise SystemExit(f"build failed on {node}")
    if args.stop_service:
        for node in nodes:
            status = _remote(node, _stop_command(args), dry_run=args.dry_run)
            if status != 0:
                raise SystemExit(f"stop failed on {node}")
    _run_parallel(args, nodes, log_tag)
    for rank, node in enumerate(nodes):
        _collect(node, rank, log_tag, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
