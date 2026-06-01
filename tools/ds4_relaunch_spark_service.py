#!/usr/bin/env python3
"""Pull, build, stop, and relaunch a DS4 Spark vLLM service.

This is the repeatable deployment path for Spark bring-up. It intentionally
routes through the repo checkout on each node:

1. git pull the selected branch,
2. run the repo build/validation step,
3. stop old matching service processes with escalation,
4. start all ranks with one coherent launch profile,
5. poll the head service until it is ready.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time


DEFAULT_NODES = [f"spark{i}" for i in range(8)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=("dsv4-pp8",), default="dsv4-pp8")
    parser.add_argument("--nodes", default=os.getenv("DS4_SPARK_NODES", ",".join(DEFAULT_NODES)))
    parser.add_argument("--head-node", default=os.getenv("DS4_HEAD_NODE", "spark0"))
    parser.add_argument("--head-addr", default=os.getenv("HEAD_ADDR", "10.10.100.10"))
    parser.add_argument("--source-root", default=os.getenv("DS4_VLLM_SOURCE_ROOT", "~/src/vllm"))
    parser.add_argument("--python", default=os.getenv("DS4_VLLM_PYTHON", "~/ds4-vllm-local/bin/python"))
    parser.add_argument("--remote", default=os.getenv("DS4_VLLM_REMOTE", "origin"))
    parser.add_argument("--branch", default=os.getenv("DS4_VLLM_BRANCH", "main"))
    parser.add_argument("--master-port", default=os.getenv("MASTER_PORT", "29944"))
    parser.add_argument("--api-port", default=os.getenv("API_PORT", "8102"))
    parser.add_argument(
        "--profile",
        default=os.getenv("DS4_DSV4_PIPELINE_RAM_PROFILE", "max-throughput"),
        help="DS4_DSV4_PIPELINE_RAM_PROFILE for dsv4-pp8",
    )
    parser.add_argument("--nnodes", type=int, default=int(os.getenv("NNODES", "8")))
    parser.add_argument("--log-tag", default="")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-stop", action="store_true")
    parser.add_argument(
        "--build-command",
        default=os.getenv("DS4_VLLM_BUILD_COMMAND", "auto"),
        help="'auto', 'make', or an explicit shell command",
    )
    parser.add_argument("--stop-timeout-s", type=float, default=15.0)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--health-poll-s", type=float, default=5.0)
    parser.add_argument(
        "--health-url",
        default="",
        help="direct health URL; by default health is checked through ssh head-node",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable passed to every service rank; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nodes_from_arg(raw: str) -> list[str]:
    nodes = [node.strip() for node in raw.split(",") if node.strip()]
    if not nodes:
        raise SystemExit("no Spark nodes specified")
    return nodes


def shell_path(path: str) -> str:
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace('"', '\\"') + '"'
    return shlex.quote(path)


def remote_run(node: str, command: str, dry_run: bool = False) -> int:
    print(f"== {node} == {command}")
    if dry_run:
        return 0
    result = subprocess.run(["ssh", node, command], text=True)
    return result.returncode


def remote_output(node: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", node, command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def repo_command(args: argparse.Namespace, inner: str) -> str:
    return f"cd {shell_path(args.source_root)} && {inner}"


def pull_command(args: argparse.Namespace) -> str:
    return repo_command(args, f"git pull {shlex.quote(args.remote)} {shlex.quote(args.branch)}")


def build_command(args: argparse.Namespace) -> str:
    py = shell_path(args.python)
    if args.build_command == "make":
        return repo_command(args, "make")
    if args.build_command != "auto":
        return repo_command(args, args.build_command)
    return repo_command(
        args,
        "if [ -f Makefile ]; then "
        "make; "
        "else "
        f"{py} -m py_compile vllm/envs.py vllm/v1/worker/workspace.py "
        "tools/ds4_stop_spark_processes.py tools/ds4_relaunch_spark_service.py "
        "tools/ds4_workspace_prealloc_audit.py && "
        "bash -n tools/ds4_launch_dsv4_flash_pp8.sh && "
        f"{py} tools/ds4_workspace_prealloc_audit.py && "
        f"{py} tools/ds4_no_marlin_static_audit.py; "
        "fi",
    )


def stop_command(args: argparse.Namespace) -> str:
    py = shell_path(args.python)
    return repo_command(
        args,
        f"{py} tools/ds4_stop_spark_processes.py --local --service dsv4 "
        f"--term-timeout-s {args.stop_timeout_s:g} --kill-timeout-s 5",
    )


def launch_env(args: argparse.Namespace, rank: int) -> dict[str, str]:
    env = {
        "NODE_RANK": str(rank),
        "HEAD_ADDR": args.head_addr,
        "NNODES": str(args.nnodes),
        "MASTER_PORT": args.master_port,
        "API_PORT": args.api_port,
        "DS4_DSV4_PIPELINE_RAM_PROFILE": args.profile,
        "VLLM_DEBUG_WORKSPACE": os.getenv("VLLM_DEBUG_WORKSPACE", "0"),
        "DS4_VLLM_SOURCE_ROOT": args.source_root.replace("~", f"/home/{node_user(rank)}", 1)
        if args.source_root.startswith("~/")
        else args.source_root,
        "DS4_VLLM_PYTHON": args.python.replace("~", f"/home/{node_user(rank)}", 1)
        if args.python.startswith("~/")
        else args.python,
    }
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env has empty key: {item!r}")
        env[key] = value
    return env


def node_user(rank: int) -> str:
    return f"spark{rank}"


def launch_command(args: argparse.Namespace, rank: int, log_tag: str) -> str:
    env = " ".join(
        f"{shlex.quote(key)}={shlex.quote(value)}"
        for key, value in launch_env(args, rank).items()
    )
    log = f"~/ds4_logs/dsv4_pp8_{log_tag}-rank{rank}.log"
    return repo_command(
        args,
        "mkdir -p ~/ds4_logs && "
        f"setsid -f env {env} tools/ds4_launch_dsv4_flash_pp8.sh "
        f"> {log} 2>&1 < /dev/null && "
        f"echo started rank={rank} log={log}",
    )


def service_url(args: argparse.Namespace) -> str:
    return args.health_url or f"http://127.0.0.1:{args.api_port}/v1/models"


def poll_health(args: argparse.Namespace) -> bool:
    deadline = time.monotonic() + args.health_timeout_s
    url = service_url(args)
    last_error = ""
    while time.monotonic() < deadline:
        try:
            if args.health_url:
                result = subprocess.run(
                    ["curl", "-sS", "--max-time", "3", url],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            else:
                result = subprocess.run(
                    [
                        "ssh",
                        args.head_node,
                        f"curl -sS --max-time 3 {shlex.quote(url)}",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            data = json.loads(result.stdout)
            print(f"health ok: {data}")
            return True
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            print(f"health waiting: {last_error}")
            time.sleep(args.health_poll_s)
    print(f"health failed after {args.health_timeout_s:g}s: {last_error}", file=sys.stderr)
    return False


def run_on_nodes(args: argparse.Namespace, nodes: list[str], command_factory) -> None:
    failed: list[str] = []
    for rank, node in enumerate(nodes):
        command = command_factory(rank, node)
        if remote_run(node, command, args.dry_run) != 0:
            failed.append(node)
    if failed:
        raise SystemExit(f"failed on node(s): {', '.join(failed)}")


def run_parallel_start(args: argparse.Namespace, nodes: list[str], log_tag: str) -> None:
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for rank, node in enumerate(nodes):
        command = launch_command(args, rank, log_tag)
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
        raise SystemExit(f"launch command failed on node(s): {', '.join(failed)}")


def main() -> int:
    args = parse_args()
    nodes = nodes_from_arg(args.nodes)
    if len(nodes) != args.nnodes:
        raise SystemExit(f"--nodes has {len(nodes)} entries but --nnodes={args.nnodes}")
    log_tag = args.log_tag or (
        "pr44_workspace_"
        + args.profile.replace("-", "_")
        + "_"
        + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    if not args.skip_pull:
        run_on_nodes(args, nodes, lambda _rank, _node: pull_command(args))
    if not args.skip_build:
        run_on_nodes(args, nodes, lambda _rank, _node: build_command(args))
    if not args.skip_stop:
        run_on_nodes(args, nodes, lambda _rank, _node: stop_command(args))
    run_parallel_start(args, nodes, log_tag)
    if args.dry_run:
        return 0
    return 0 if poll_health(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
