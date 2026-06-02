#!/usr/bin/env python3
"""Stop DS4 Spark service processes before relaunch.

The relaunch path should not depend on ad-hoc pgrep/pkill fragments. This
script finds exact DS4/vLLM process classes, sends SIGTERM, waits, then sends
SIGKILL to anything still alive. It can run on one Spark or over the whole
Spark fleet through ssh.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_NODES = ",".join(f"spark{i}" for i in range(8))

SERVICE_PATTERNS = {
    "dsv4": [
        r"\bvllm\.entrypoints\.cli\.main serve .*DeepSeek-V4-Flash\b",
        r"\bds4_launch_dsv4_flash_(?:pp8|pp4_tp2_ep|tp2_native_benchmark)\.sh\b",
        r"\bds4_nccl_preflight\.py\b",
    ],
    "qwen": [
        r"\bvllm\.entrypoints\.cli\.main serve .*(?:Qwen|qwen|sakamakismile)\b",
        r"\bds4_launch_qwen27_.*\.sh\b",
    ],
    "vllm": [
        r"\bvllm\.entrypoints\.cli\.main serve\b",
        r"\bds4_launch_(?:dsv4|qwen).*\.sh\b",
        r"\bds4_nccl_preflight\.py\b",
    ],
    "benchmarks": [
        r"\bds4_api_queue_benchmark\.py\b",
        r"\bds4_queue_saturation\.py\b",
        r"\bds4_nccl_p2p_bench\.py\b",
        r"\bds4_run_nccl_p2p_bench\.py\b",
    ],
    "coordinator": [
        r"\bpython[0-9.]* -m ds4_infer\.api\b",
        r"\bds4_coordinator_api\.sh\b",
    ],
}


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", action="store_true", help="run on all nodes via ssh")
    parser.add_argument("--local", action="store_true", help="run only on this host")
    parser.add_argument(
        "--service",
        choices=("dsv4", "qwen", "vllm", "all"),
        default="dsv4",
        help="service process class to stop",
    )
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help="also stop DS4 benchmark clients",
    )
    parser.add_argument(
        "--include-coordinator",
        action="store_true",
        help="also stop the spark0 coordinator API",
    )
    parser.add_argument(
        "--nodes",
        default=os.getenv("DS4_SPARK_NODES", DEFAULT_NODES),
        help="comma-separated fleet nodes for --fleet",
    )
    parser.add_argument(
        "--remote-source-root",
        default=os.getenv("DS4_VLLM_SOURCE_ROOT", "~/src/vllm"),
        help="vLLM checkout path on remote nodes",
    )
    parser.add_argument(
        "--remote-python",
        default=os.getenv("DS4_VLLM_PYTHON", "~/ds4-vllm-local/bin/python"),
        help="Python executable on remote nodes",
    )
    parser.add_argument("--term-timeout-s", type=float, default=10.0)
    parser.add_argument("--kill-timeout-s", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="do not escalate to SIGKILL after SIGTERM timeout",
    )
    return parser.parse_args()


def process_table() -> list[ProcessInfo]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    processes: list[ProcessInfo] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            processes.append(ProcessInfo(int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return processes


def selected_patterns(args: argparse.Namespace) -> list[re.Pattern[str]]:
    names = ["dsv4", "qwen", "vllm"] if args.service == "all" else [args.service]
    if args.include_benchmarks:
        names.append("benchmarks")
    if args.include_coordinator:
        names.append("coordinator")
    patterns: list[re.Pattern[str]] = []
    for name in names:
        patterns.extend(re.compile(pattern) for pattern in SERVICE_PATTERNS[name])
    return patterns


def find_matches(
    processes: list[ProcessInfo], patterns: list[re.Pattern[str]]
) -> dict[int, ProcessInfo]:
    ignored = {os.getpid(), os.getppid()}
    matches: dict[int, ProcessInfo] = {}
    for proc in processes:
        if proc.pid in ignored:
            continue
        if "ds4_stop_spark_processes.py" in proc.command:
            continue
        if any(pattern.search(proc.command) for pattern in patterns):
            matches[proc.pid] = proc
    return matches


def add_orphan_vllm_workers(
    processes: list[ProcessInfo],
    matches: dict[int, ProcessInfo],
) -> dict[int, ProcessInfo]:
    """Catch worker processes left behind after their vLLM parent dies.

    Multiprocessing workers rewrite argv to names such as ``VLLM::Worker_PP0``.
    Once the parent API/EngineCore process is gone, there is no model path left
    in argv for the normal service regex to match.  Restrict this to ppid=1 so
    an active, healthy service with live children is not collected by accident.
    """

    orphan_pattern = re.compile(r"^VLLM::(?:Worker(?:_|\b)|EngineCore\b)")
    for proc in processes:
        if proc.ppid == 1 and orphan_pattern.search(proc.command):
            matches[proc.pid] = proc
    return matches


def add_descendants(
    processes: list[ProcessInfo], matches: dict[int, ProcessInfo]
) -> dict[int, ProcessInfo]:
    by_parent: dict[int, list[ProcessInfo]] = {}
    by_pid = {proc.pid: proc for proc in processes}
    for proc in processes:
        by_parent.setdefault(proc.ppid, []).append(proc)
    pending = list(matches)
    while pending:
        parent = pending.pop()
        for child in by_parent.get(parent, []):
            if child.pid not in matches:
                matches[child.pid] = child
                pending.append(child.pid)
    return {pid: by_pid.get(pid, proc) for pid, proc in matches.items()}


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_dead(pids: list[int], timeout_s: float) -> list[int]:
    deadline = time.monotonic() + timeout_s
    alive = [pid for pid in pids if pid_alive(pid)]
    while alive and time.monotonic() < deadline:
        time.sleep(0.25)
        alive = [pid for pid in alive if pid_alive(pid)]
    return alive


def kill_pids(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"ERROR cannot signal pid={pid}: {exc}", file=sys.stderr)


def run_local(args: argparse.Namespace) -> int:
    processes = process_table()
    matches = find_matches(processes, selected_patterns(args))
    matches = add_orphan_vllm_workers(processes, matches)
    matches = add_descendants(processes, matches)
    host = socket.gethostname()
    if not matches:
        print(f"{host}: no matching {args.service} processes")
        return 0
    ordered = sorted(matches.values(), key=lambda proc: proc.pid, reverse=True)
    print(f"{host}: matched {len(ordered)} process(es)")
    for proc in ordered:
        print(f"{host}: pid={proc.pid} ppid={proc.ppid} cmd={proc.command}")
    if args.dry_run:
        return 0
    pids = [proc.pid for proc in ordered]
    kill_pids(pids, signal.SIGTERM)
    alive = wait_dead(pids, args.term_timeout_s)
    if alive and not args.no_force:
        print(f"{host}: SIGTERM left {len(alive)} process(es); sending SIGKILL")
        kill_pids(alive, signal.SIGKILL)
        alive = wait_dead(alive, args.kill_timeout_s)
    if alive:
        print(f"{host}: failed to stop pid(s): {alive}", file=sys.stderr)
        return 2
    print(f"{host}: stopped {len(pids)} process(es)")
    return 0


def shell_path(path: str) -> str:
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace('"', '\\"') + '"'
    return shlex.quote(path)


def remote_command(args: argparse.Namespace) -> str:
    remote_args = [
        "--local",
        "--service",
        args.service,
        "--term-timeout-s",
        str(args.term_timeout_s),
        "--kill-timeout-s",
        str(args.kill_timeout_s),
    ]
    if args.include_benchmarks:
        remote_args.append("--include-benchmarks")
    if args.include_coordinator:
        remote_args.append("--include-coordinator")
    if args.dry_run:
        remote_args.append("--dry-run")
    if args.no_force:
        remote_args.append("--no-force")
    quoted_args = " ".join(shlex.quote(item) for item in remote_args)
    return (
        f"cd {shell_path(args.remote_source_root)} && "
        f"{shell_path(args.remote_python)} tools/ds4_stop_spark_processes.py "
        f"{quoted_args}"
    )


def run_fleet(args: argparse.Namespace) -> int:
    nodes = [node.strip() for node in args.nodes.split(",") if node.strip()]
    command = remote_command(args)
    failed = 0
    for node in nodes:
        print(f"== {node} ==")
        result = subprocess.run(["ssh", node, command], text=True)
        if result.returncode != 0:
            failed += 1
            print(f"{node}: stop failed with status {result.returncode}", file=sys.stderr)
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    if args.fleet and args.local:
        print("choose only one of --fleet or --local", file=sys.stderr)
        return 2
    if args.fleet:
        return run_fleet(args)
    return run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
