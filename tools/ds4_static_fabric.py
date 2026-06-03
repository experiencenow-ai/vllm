#!/usr/bin/env python3
"""Apply and verify the fixed DS4 Spark fabric route profile.

This is the post-power-cycle network setup path. Model launch should not invent
fabric topology; it should verify this profile and then use the fixed interface
bindings exported by the launcher/relauncher.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import shlex
import subprocess
import sys


DEFAULT_NODES = [f"spark{i}" for i in range(8)]
RAIL_DEVICES = {
    "enp": ("enp1s0f0np0", "enp1s0f1np1"),
    "rail0": ("enp1s0f0np0", "enp1s0f1np1"),
    "lower": ("enp1s0f0np0", "enp1s0f1np1"),
    "enp2p": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
    "enP2p": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
    "rail1": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
    "upper": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
}


@dataclass(frozen=True)
class LinkSpec:
    rank: int
    peer_rank: int
    dev: str
    local_ip: str
    prefix: int = 30

    @property
    def cidr(self) -> str:
        return f"{self.local_ip}/{self.prefix}"

    @property
    def label(self) -> str:
        return f"spark{self.rank}<->spark{self.peer_rank}"


@dataclass(frozen=True)
class RouteSpec:
    source_rank: int
    target_rank: int
    target_ip: str
    via: str
    dev: str
    source_ip: str

    @property
    def label(self) -> str:
        return f"spark{self.source_rank}->spark{self.target_rank}"


def fabric_ip(rank: int) -> str:
    return f"10.10.100.{10 + rank}"


def nodes_from_arg(raw: str) -> list[str]:
    nodes = [item.strip() for item in raw.split(",") if item.strip()]
    if not nodes:
        raise SystemExit("no Spark nodes configured")
    return nodes


def shell_path(path: str) -> str:
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace('"', '\\"') + '"'
    return shlex.quote(path)


def rail_devices(edge_rail: str) -> tuple[str, str]:
    if edge_rail in RAIL_DEVICES:
        return RAIL_DEVICES[edge_rail]
    if "," in edge_rail:
        left, right = [item.strip() for item in edge_rail.split(",", 1)]
        if left and right:
            return (left, right)
    raise SystemExit(
        "--edge-rail must be enp, enP2p, or a '<prev-dev>,<next-dev>' pair"
    )


def route_dev(source_rank: int, target_rank: int, edge_rail: str) -> str:
    prev_dev, next_dev = rail_devices(edge_rail)
    if target_rank > source_rank:
        return next_dev
    return prev_dev


def line_next_hop(source_rank: int, target_rank: int, edge_rail: str) -> tuple[str, str]:
    if source_rank == target_rank:
        raise ValueError("self route does not need a next hop")
    if target_rank > source_rank:
        subnet = ((source_rank + 1) * 2)
        return (f"10.10.{subnet}.2", route_dev(source_rank, target_rank, edge_rail))
    subnet = (source_rank * 2)
    return (f"10.10.{subnet}.1", route_dev(source_rank, target_rank, edge_rail))


def build_link_specs(rank: int, nodes: int, edge_rail: str) -> list[LinkSpec]:
    prev_dev, next_dev = rail_devices(edge_rail)
    specs: list[LinkSpec] = []
    if rank > 0:
        subnet = (rank * 2)
        specs.append(
            LinkSpec(
                rank=rank,
                peer_rank=(rank - 1),
                dev=prev_dev,
                local_ip=f"10.10.{subnet}.2",
            )
        )
    if rank < (nodes - 1):
        subnet = ((rank + 1) * 2)
        specs.append(
            LinkSpec(
                rank=rank,
                peer_rank=(rank + 1),
                dev=next_dev,
                local_ip=f"10.10.{subnet}.1",
            )
        )
    return specs


def build_specs(nodes: int, route_scope: str, edge_rail: str) -> list[RouteSpec]:
    specs: list[RouteSpec] = []
    for source_rank in range(nodes):
        for target_rank in range(nodes):
            if source_rank == target_rank:
                continue
            if route_scope == "adjacent" and abs(source_rank - target_rank) != 1:
                continue
            if route_scope == "head" and source_rank != 0 and target_rank != 0:
                continue
            via, dev = line_next_hop(source_rank, target_rank, edge_rail)
            specs.append(
                RouteSpec(
                    source_rank=source_rank,
                    target_rank=target_rank,
                    target_ip=fabric_ip(target_rank),
                    via=via,
                    dev=dev,
                    source_ip=fabric_ip(source_rank),
                )
            )
    return specs


def run_local(command: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )


def run_ssh(host: str, command: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return run_local(["ssh", host, command], timeout_s)


def sudo_prefix(enabled: bool) -> list[str]:
    if not enabled:
        return []
    return ["sudo", "-n"]


def sudo_available(args: argparse.Namespace) -> bool:
    if args.no_sudo:
        return True
    result = run_local(["sudo", "-n", "true"], args.timeout_s)
    if result.returncode == 0:
        return True
    print(
        "FAIL static fabric sudo: passwordless sudo is required for ip/sysctl "
        "setup; run as root, configure a constrained sudoers rule, or pass "
        "--no-sudo only from a root-owned service",
        file=sys.stderr,
    )
    return False


def apply_command(args: argparse.Namespace, specs: list[RouteSpec]) -> list[list[str]]:
    rank = args.rank
    if rank is None:
        raise SystemExit("--local --apply requires --rank")
    prefix = sudo_prefix(not args.no_sudo)
    commands: list[list[str]] = []
    commands.append(prefix + ["ip", "link", "show", args.loopback_dev])
    commands.append(prefix + ["ip", "link", "set", args.loopback_dev, "up"])
    commands.append(
        prefix
        + [
            "ip",
            "addr",
            "replace",
            f"{fabric_ip(rank)}/32",
            "dev",
            args.loopback_dev,
        ]
    )
    for link in build_link_specs(rank, args.nnodes, args.edge_rail):
        commands.append(prefix + ["ip", "link", "set", link.dev, "up"])
        commands.append(prefix + ["ip", "addr", "replace", link.cidr, "dev", link.dev])
    if args.route_scope != "adjacent":
        commands.append(prefix + ["sysctl", "-w", "net.ipv4.ip_forward=1"])
    for spec in specs:
        if spec.source_rank != rank:
            continue
        commands.append(
            prefix
            + [
                "ip",
                "route",
                "replace",
                spec.target_ip,
                "via",
                spec.via,
                "dev",
                spec.dev,
                "src",
                spec.source_ip,
            ]
        )
    return commands


def local_apply(args: argparse.Namespace) -> int:
    rank = args.rank
    if rank is None:
        raise SystemExit("--local --apply requires --rank")
    if not sudo_available(args):
        return 1
    specs = build_specs(args.nnodes, args.route_scope, args.edge_rail)
    commands = apply_command(args, specs)
    if args.dry_run:
        for command in commands:
            print(" ".join(shlex.quote(item) for item in command))
        return 0
    failures = 0
    for command in commands:
        result = run_local(command, args.timeout_s)
        if result.returncode == 0:
            continue
        if command[-1] == args.loopback_dev and "does not exist" in result.stderr:
            create = sudo_prefix(not args.no_sudo) + [
                "ip",
                "link",
                "add",
                args.loopback_dev,
                "type",
                "dummy",
            ]
            create_result = run_local(create, args.timeout_s)
            if create_result.returncode == 0:
                continue
            result = create_result
        failures += 1
        print(
            "FAIL apply "
            + " ".join(shlex.quote(item) for item in command)
            + f": {(result.stderr or result.stdout).strip()}",
            file=sys.stderr,
        )
    if failures:
        return 1
    print(
        f"PASS static fabric apply rank={rank} loopback={fabric_ip(rank)} "
        f"routes={len([item for item in specs if item.source_rank == rank])}",
        file=sys.stderr,
    )
    return 0


def extract_field(route: str, field: str) -> str | None:
    parts = route.split()
    for index, part in enumerate(parts[:-1]):
        if part == field:
            return parts[index + 1]
    return None


def dev_is_up(dev: str, timeout_s: int) -> bool:
    result = run_local(["cat", f"/sys/class/net/{dev}/operstate"], timeout_s)
    return result.returncode == 0 and result.stdout.strip() == "up"


def link_addr_present(link: LinkSpec, timeout_s: int) -> bool:
    result = run_local(["ip", "-o", "-4", "addr", "show", "dev", link.dev], timeout_s)
    return result.returncode == 0 and f" {link.cidr} " in f" {result.stdout} "


def local_verify(args: argparse.Namespace) -> int:
    rank = args.rank
    if rank is None:
        raise SystemExit("--local --verify requires --rank")
    failures = 0
    local_ip = fabric_ip(rank)
    addr = run_local(
        ["ip", "-o", "-4", "addr", "show", "dev", args.loopback_dev],
        args.timeout_s,
    )
    if addr.returncode != 0 or f" {local_ip}/32 " not in f" {addr.stdout} ":
        print(
            f"FAIL static fabric loopback rank={rank}: "
            f"{args.loopback_dev} lacks {local_ip}/32",
            file=sys.stderr,
        )
        failures += 1
    for link in build_link_specs(rank, args.nnodes, args.edge_rail):
        if dev_is_up(link.dev, args.timeout_s) and link_addr_present(link, args.timeout_s):
            print(f"PASS {link.label:<15} {link.cidr:<15} dev {link.dev}")
            continue
        print(
            f"FAIL static fabric link rank={rank}: "
            f"{link.dev} lacks {link.cidr} or is not up",
            file=sys.stderr,
        )
        failures += 1
    specs = build_specs(args.nnodes, args.route_scope, args.edge_rail)
    for spec in specs:
        if spec.source_rank != rank:
            continue
        result = run_local(["ip", "route", "get", spec.target_ip], args.timeout_s)
        route = (result.stdout + result.stderr).splitlines()[0].strip()
        dev = extract_field(route, "dev")
        via = extract_field(route, "via")
        src = extract_field(route, "src")
        ok = (
            result.returncode == 0
            and dev == spec.dev
            and via == spec.via
            and src == spec.source_ip
            and dev_is_up(spec.dev, args.timeout_s)
            and " dev wl" not in route
            and " via 192.168." not in route
        )
        status = "PASS" if ok else "FAIL"
        print(f"{status} {spec.label:<15} {spec.target_ip:<13} :: {route}")
        if not ok:
            failures += 1
    if args.route_scope != "adjacent":
        forwarded = run_local(["sysctl", "-n", "net.ipv4.ip_forward"], args.timeout_s)
        if forwarded.returncode != 0 or forwarded.stdout.strip() != "1":
            print(
                f"FAIL static fabric forwarding rank={rank}: "
                "net.ipv4.ip_forward is not 1",
                file=sys.stderr,
            )
            failures += 1
    if failures:
        print(f"static fabric verify failed: rank={rank} failures={failures}")
        return 1
    print(f"static fabric verify passed: rank={rank}")
    return 0


def remote_script_command(args: argparse.Namespace, rank: int, mode: str) -> str:
    py = shell_path(args.python)
    pieces = [
        py,
        "tools/ds4_static_fabric.py",
        "--local",
        mode,
        "--rank",
        str(rank),
        "--nnodes",
        str(args.nnodes),
        "--route-scope",
        args.route_scope,
        "--edge-rail",
        shlex.quote(args.edge_rail),
        "--loopback-dev",
        shlex.quote(args.loopback_dev),
        "--timeout-s",
        str(args.timeout_s),
    ]
    if args.no_sudo:
        pieces.append("--no-sudo")
    return f"cd {shell_path(args.source_root)} && " + " ".join(pieces)


def fleet_run(args: argparse.Namespace, mode: str) -> int:
    nodes = nodes_from_arg(args.nodes)
    if len(nodes) != args.nnodes:
        raise SystemExit(f"--nodes has {len(nodes)} entries but --nnodes={args.nnodes}")
    failures = 0
    for rank, node in enumerate(nodes):
        command = remote_script_command(args, rank, mode)
        print(f"== {node} == {command}")
        if args.dry_run:
            continue
        result = run_ssh(node, command, max(args.timeout_s * 4, 20))
        output = (result.stdout + result.stderr).strip()
        if output:
            print(output)
        if result.returncode != 0:
            failures += 1
    return 1 if failures else 0


def print_env(args: argparse.Namespace) -> int:
    prev_dev, next_dev = rail_devices(args.edge_rail)
    if prev_dev == next_dev:
        ifnames = prev_dev
    else:
        ifnames = f"{prev_dev},{next_dev}"
    env = {
        "DS4_200G_IFNAME": ifnames,
        "DS4_200G_SOCKET_IFNAME": ifnames,
        "DS4_200G_NCCL_IFNAME": ifnames,
        "DS4_CONTROL_IFNAME": args.loopback_dev,
        "DS4_GLOO_SOCKET_IFNAME": args.gloo_ifname,
        "DS4_200G_ADVERTISE_LOOPBACK": "1",
        "DS4_200G_NCCL_TRANSPORT": "socket",
        "VLLM_DS4_PP_EDGE_RAIL": args.edge_rail,
        "DS4_NCCL_PREFLIGHT_PP_EDGE_RAIL": args.edge_rail,
    }
    for key, value in env.items():
        print(f"{key}={value}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", action="store_true", help="run across all nodes by ssh")
    parser.add_argument("--local", action="store_true", help="run on the current node")
    parser.add_argument("--apply", action="store_true", help="apply static routes")
    parser.add_argument("--verify", action="store_true", help="verify static routes")
    parser.add_argument("--print-launch-env", action="store_true")
    parser.add_argument("--nodes", default=os.getenv("DS4_SPARK_NODES", ",".join(DEFAULT_NODES)))
    parser.add_argument("--nnodes", type=int, default=int(os.getenv("NNODES", "8")))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--source-root", default=os.getenv("DS4_VLLM_SOURCE_ROOT", "~/src/vllm"))
    parser.add_argument("--python", default=os.getenv("DS4_VLLM_PYTHON", "~/ds4-vllm-local/bin/python"))
    parser.add_argument("--edge-rail", default=os.getenv("DS4_STATIC_FABRIC_EDGE_RAIL", "enp"))
    parser.add_argument(
        "--route-scope",
        choices=("all", "adjacent", "head"),
        default=os.getenv("DS4_STATIC_FABRIC_ROUTE_SCOPE", "all"),
    )
    parser.add_argument("--loopback-dev", default=os.getenv("DS4_STATIC_FABRIC_LOOPBACK_DEV", "ds4ring0"))
    parser.add_argument("--gloo-ifname", default=os.getenv("DS4_STATIC_FABRIC_GLOO_IFNAME", "enP7s7"))
    parser.add_argument("--timeout-s", type=int, default=int(os.getenv("DS4_STATIC_FABRIC_TIMEOUT_S", "8")))
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_launch_env:
        return print_env(args)
    if args.apply == args.verify:
        raise SystemExit("select exactly one of --apply or --verify")
    if args.fleet == args.local:
        raise SystemExit("select exactly one of --fleet or --local")
    if args.local and args.apply:
        return local_apply(args)
    if args.local and args.verify:
        return local_verify(args)
    if args.fleet and args.apply:
        return fleet_run(args, "--apply")
    if args.fleet and args.verify:
        return fleet_run(args, "--verify")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
