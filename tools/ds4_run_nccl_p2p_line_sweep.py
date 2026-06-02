#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run DS4 NCCL/PyNCCL P2P benchmarks one ring/line edge at a time.

The all-rank benchmark is useful for collective startup issues, but it is a
poor cable isolator because a single bad edge or multi-interface NCCL choice can
hide every other result.  This wrapper runs each physical edge as an independent
2-rank world with the exact source/destination rail interfaces for that cable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    name: str
    src_node: str
    dst_node: str
    src_ip: str
    dst_ip: str
    src_ifname: str
    dst_ifname: str


EDGES: dict[str, Edge] = {
    "0-1": Edge("0-1", "spark0", "spark1", "10.10.100.10", "10.10.100.11", "enP2p1s0f1np1", "enP2p1s0f0np0"),
    "1-2": Edge("1-2", "spark1", "spark2", "10.10.100.11", "10.10.100.12", "enP2p1s0f1np1", "enP2p1s0f0np0"),
    "2-3": Edge("2-3", "spark2", "spark3", "10.10.100.12", "10.10.100.13", "enP2p1s0f1np1", "enP2p1s0f0np0"),
    "3-4": Edge("3-4", "spark3", "spark4", "10.10.100.13", "10.10.100.14", "enP2p1s0f1np1", "enP2p1s0f0np0"),
    "4-5": Edge("4-5", "spark4", "spark5", "10.10.100.14", "10.10.100.15", "enP2p1s0f1np1", "enp1s0f0np0"),
    "5-6": Edge("5-6", "spark5", "spark6", "10.10.100.15", "10.10.100.16", "enP2p1s0f1np1", "enP2p1s0f0np0"),
    "6-7": Edge("6-7", "spark6", "spark7", "10.10.100.16", "10.10.100.17", "enP2p1s0f1np1", "enP2p1s0f0np0"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default=";".join(EDGES))
    parser.add_argument("--methods", default="torch")
    parser.add_argument("--bytes", default="16777216,67108864")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stripes", type=int, default=1)
    parser.add_argument("--direction", choices=("unidirectional", "bidirectional"), default="unidirectional")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--master-port-base", type=int, default=31020)
    parser.add_argument("--log-tag", default="p2p_line_sweep")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--stop-service", default="")
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def selected_edges(raw: str) -> list[Edge]:
    out: list[Edge] = []
    for item in raw.split(";"):
        name = item.strip()
        if not name:
            continue
        edge = EDGES.get(name)
        if edge is None:
            raise SystemExit(f"unknown edge {name!r}; known={','.join(EDGES)}")
        out.append(edge)
    if not out:
        raise SystemExit("no edges selected")
    return out


def main() -> int:
    args = parse_args()
    edges = selected_edges(args.edges)
    failed: list[str] = []
    for index, edge in enumerate(edges):
        command = [
            sys.executable,
            "tools/ds4_run_nccl_p2p_bench.py",
            "--nodes",
            f"{edge.src_node},{edge.dst_node}",
            "--nnodes",
            "2",
            "--master-addr",
            edge.src_ip,
            "--master-port",
            str(args.master_port_base + index),
            "--pairs",
            "0-1",
            "--methods",
            args.methods,
            "--bytes",
            args.bytes,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--stripes",
            str(args.stripes),
            "--direction",
            args.direction,
            "--dtype",
            args.dtype,
            "--nccl-ifnames",
            f"{edge.src_ifname}|{edge.dst_ifname}",
            "--gloo-ifnames",
            f"{edge.src_ifname}|{edge.dst_ifname}",
            "--host-ips",
            f"{edge.src_ip}|{edge.dst_ip}",
            "--timeout-s",
            str(args.timeout_s),
            "--log-tag",
            f"{args.log_tag}_{edge.name.replace('-', '_')}",
            "--extra-env",
            "NCCL_NET=Socket",
            "--extra-env",
            "NCCL_IB_DISABLE=1",
        ]
        if args.pull:
            command.append("--pull")
        if args.build:
            command.append("--build")
        if args.stop_service:
            command.extend(["--stop-service", args.stop_service])
        for item in args.extra_env:
            command.extend(["--extra-env", item])
        print(f"== edge {edge.name} {edge.src_node}->{edge.dst_node} ==")
        status = subprocess.run(command, text=True).returncode
        if status != 0:
            failed.append(f"{edge.name}:{status}")
    if failed:
        raise SystemExit(f"line P2P sweep failed on {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
