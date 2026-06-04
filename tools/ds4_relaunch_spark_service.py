#!/usr/bin/env python3
"""Pull, build, stop, and relaunch a DS4 Spark vLLM service.

This is the repeatable deployment path for Spark bring-up. It intentionally
routes through the repo checkout on each node:

1. git pull the selected branch,
2. run the repo build/validation step,
3. verify/apply the fixed static Spark fabric profile,
4. stop old matching service processes with escalation,
5. start all ranks with one coherent launch profile,
6. poll the head service until it is ready.

Use --setup-only after a Spark power cycle when the nodes need the checked-in
repo, build validation, and static fabric restored, but no model should be
started yet.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import signal
import shlex
import subprocess
import sys
import time


DEFAULT_NODES = [f"spark{i}" for i in range(8)]


class LaunchTimer:
    def __init__(self) -> None:
        self._started = time.monotonic()
        self._rows: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def phase(self, name: str):
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self._rows.append((name, elapsed))
            print(f"[launch-timer] {name}: {elapsed:.1f}s")

    def summary(self) -> None:
        total = time.monotonic() - self._started
        print("[launch-timer] summary:")
        for name, elapsed in self._rows:
            print(f"[launch-timer]   {name}: {elapsed:.1f}s")
        print(f"[launch-timer]   total: {total:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service",
        choices=("dsv4-pp8", "dsv4-pp4-tp2-ep"),
        default="dsv4-pp8",
    )
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
        default=os.getenv("DS4_DSV4_PIPELINE_RAM_PROFILE", "resident3"),
        help="DS4_DSV4_PIPELINE_RAM_PROFILE for the selected DSV4 service",
    )
    parser.add_argument("--nnodes", type=int, default=int(os.getenv("NNODES", "8")))
    parser.add_argument("--log-tag", default="")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-stop", action="store_true")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="pull/build and run static-fabric setup, then exit before stopping or launching service",
    )
    parser.add_argument(
        "--static-fabric",
        choices=("off", "verify", "apply"),
        default=os.getenv("DS4_STATIC_FABRIC_MODE", "verify"),
        help="verify or apply the fixed Spark fabric profile before stopping/running service",
    )
    parser.add_argument(
        "--static-fabric-edge-rail",
        default=os.getenv("DS4_STATIC_FABRIC_EDGE_RAIL", os.getenv("VLLM_DS4_PP_EDGE_RAIL", "enp")),
    )
    parser.add_argument(
        "--static-fabric-route-scope",
        choices=("all", "adjacent", "head"),
        default=os.getenv("DS4_STATIC_FABRIC_ROUTE_SCOPE", "all"),
    )
    parser.add_argument(
        "--static-fabric-timeout-s",
        type=int,
        default=int(os.getenv("DS4_STATIC_FABRIC_TIMEOUT_S", "8")),
    )
    parser.add_argument(
        "--skip-local-controller-cleanup",
        action="store_true",
        help="do not terminate stale local ds4_relaunch_spark_service.py controllers for this service",
    )
    parser.add_argument("--local-controller-stop-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--build-command",
        default=os.getenv("DS4_VLLM_BUILD_COMMAND", "auto"),
        help="'auto', 'make', or an explicit shell command",
    )
    parser.add_argument("--stop-timeout-s", type=float, default=15.0)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--health-poll-s", type=float, default=5.0)
    parser.add_argument(
        "--startup-fail-fast-s",
        type=float,
        default=30.0,
        help="after this grace period, fail health polling early if no head-node launch, preflight, or vLLM process is alive",
    )
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


def _command_service_matches(command: str, service: str) -> bool:
    if "--service" not in command:
        return service == "dsv4-pp8"
    return f"--service {service}" in command or f"--service={service}" in command


def cleanup_stale_local_controllers(args: argparse.Namespace) -> None:
    if args.skip_local_controller_cleanup or args.dry_run:
        return
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: could not inspect local relaunch controllers: {exc}", file=sys.stderr)
        return
    own_pid = os.getpid()
    victims: list[tuple[int, str]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        if "ds4_relaunch_spark_service.py" not in command:
            continue
        if not _command_service_matches(command, args.service):
            continue
        victims.append((pid, command))
    if not victims:
        return
    for pid, command in victims:
        print(f"terminating stale local relaunch controller pid={pid}: {command}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + args.local_controller_stop_timeout_s
    remaining = [pid for pid, _command in victims]
    while remaining and time.monotonic() < deadline:
        next_remaining: list[int] = []
        for pid in remaining:
            try:
                os.kill(pid, 0)
                next_remaining.append(pid)
            except ProcessLookupError:
                pass
        remaining = next_remaining
        if remaining:
            time.sleep(0.2)
    for pid in remaining:
        print(f"force killing stale local relaunch controller pid={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def repo_command(args: argparse.Namespace, inner: str) -> str:
    return f"cd {shell_path(args.source_root)} && {inner}"


def pull_command(args: argparse.Namespace) -> str:
    return repo_command(args, f"git pull --ff-only {shlex.quote(args.remote)} {shlex.quote(args.branch)}")


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
        f"{py} -m py_compile vllm/envs.py "
        "vllm/entrypoints/openai/ds4_kv_cache.py "
        "vllm/entrypoints/openai/completion/protocol.py "
        "vllm/entrypoints/openai/chat_completion/protocol.py "
        "vllm/distributed/parallel_state.py "
        "vllm/distributed/ds4_high_speed_channel.py "
        "vllm/distributed/ds4_tcp_tensor_channel.py "
        "vllm/v1/engine/core.py "
        "vllm/v1/core/sched/scheduler.py "
        "vllm/v1/worker/gpu_worker.py "
        "vllm/v1/worker/gpu_model_runner.py "
        "vllm/model_executor/kernels/mhc/tilelang.py "
        "vllm/model_executor/layers/mhc.py "
        "vllm/models/deepseek_v4/nvidia/model.py "
        "vllm/models/deepseek_v4/nvidia/flashmla.py "
        "vllm/config/compilation.py "
        "vllm/v1/attention/backends/mla/flashmla_sparse.py "
        "vllm/v1/attention/backends/mla/sparse_swa.py "
        "vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py "
        "vllm/v1/simple_kv_offload/capacity.py "
        "vllm/v1/simple_kv_offload/manager.py "
        "vllm/v1/simple_kv_offload/persistent_disk.py "
        "vllm/v1/simple_kv_offload/worker.py "
        "vllm/v1/worker/workspace.py "
        "tools/ds4_stop_spark_processes.py tools/ds4_relaunch_spark_service.py "
        "tools/ds4_static_fabric.py "
        "tools/ds4_nccl_preflight.py tools/ds4_nccl_p2p_bench.py "
        "tools/ds4_run_nccl_p2p_bench.py "
        "tools/ds4_mhc_correctness_probe.py "
        "tools/ds4_cudagraph_support_audit.py "
        "tools/ds4_simple_kv_offload_audit.py "
        "tools/ds4_workspace_prealloc_audit.py "
        "tools/ds4_cohort_admission_audit.py "
        "tools/ds4_high_speed_channel_audit.py "
        "tools/ds4_pp_wave_admission_audit.py tools/ds4_speed_path_audit.py "
        "tools/ds4_pp_boundary_trace_audit.py "
        "tools/ds4_dsv4_pp_hc_boundary_audit.py "
        "tools/ds4_dsv4_layer_backend_audit.py "
        "tools/ds4_dsv4_hc_head_backend_audit.py "
        "tools/ds4_dsv4_weight_audit.py "
        "tools/ds4_dsv4_hash_moe_router_audit.py "
        "tools/ds4_dsv4_mxfp4_swiglu_audit.py "
        "tools/ds4_dsv4_mxfp4_layout_audit.py "
        "tools/ds4_sparse_mla_correctness_audit.py "
        "tools/ds4_mhc_large_prefill_audit.py "
        "tools/ds4_nccl_p2p_bench_audit.py && "
        "bash -n tools/ds4_200g_guard.sh && "
        "bash -n tools/ds4_install_static_fabric_sudoers.sh && "
        "bash -n tools/ds4_launch_dsv4_flash_pp8.sh && "
        "bash -n tools/ds4_launch_dsv4_flash_pp4_tp2_ep.sh && "
        f"{py} tools/ds4_workspace_prealloc_audit.py && "
        f"{py} tools/ds4_cudagraph_support_audit.py && "
        f"{py} tools/ds4_simple_kv_offload_audit.py && "
        f"{py} tools/ds4_cohort_admission_audit.py && "
        f"{py} tools/ds4_high_speed_channel_audit.py && "
        f"{py} tools/ds4_pp_wave_admission_audit.py && "
        f"{py} tools/ds4_speed_path_audit.py && "
        f"{py} tools/ds4_pp_boundary_trace_audit.py && "
        f"{py} tools/ds4_dsv4_pp_hc_boundary_audit.py && "
        f"{py} tools/ds4_dsv4_layer_backend_audit.py && "
        f"{py} tools/ds4_dsv4_hc_head_backend_audit.py && "
        f"{py} tools/ds4_dsv4_weight_audit.py && "
        f"{py} tools/ds4_dsv4_hash_moe_router_audit.py && "
        f"{py} tools/ds4_dsv4_mxfp4_swiglu_audit.py && "
        f"{py} tools/ds4_dsv4_mxfp4_layout_audit.py && "
        f"{py} tools/ds4_sparse_mla_correctness_audit.py && "
        f"{py} tools/ds4_mhc_large_prefill_audit.py && "
        f"{py} tools/ds4_nccl_p2p_bench_audit.py && "
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
    fabric_ifnames = static_fabric_ifnames(args.static_fabric_edge_rail)
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
        "DS4_200G_IFNAME": fabric_ifnames,
        "DS4_CONTROL_IFNAME": "ds4ring0",
        "DS4_GLOO_SOCKET_IFNAME": "enP7s7",
        "DS4_200G_ADVERTISE_LOOPBACK": "1",
        "DS4_200G_NCCL_TRANSPORT": "socket",
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


def static_fabric_ifnames(edge_rail: str) -> str:
    explicit = [item.strip() for item in edge_rail.split(",") if item.strip()]
    if len(explicit) > 2:
        return ",".join(explicit)
    rails = {
        "enp": ("enp1s0f0np0", "enp1s0f1np1"),
        "rail0": ("enp1s0f0np0", "enp1s0f1np1"),
        "lower": ("enp1s0f0np0", "enp1s0f1np1"),
        "enp2p": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
        "enP2p": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
        "rail1": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
        "upper": ("enP2p1s0f0np0", "enP2p1s0f1np1"),
    }
    if edge_rail in rails:
        left, right = rails[edge_rail]
    elif "," in edge_rail:
        left, right = [item.strip() for item in edge_rail.split(",", 1)]
    else:
        return edge_rail
    if left == right:
        return left
    return f"{left},{right}"


def node_user(rank: int) -> str:
    return f"spark{rank}"


def launch_command(args: argparse.Namespace, rank: int, log_tag: str) -> str:
    env = " ".join(
        f"{shlex.quote(key)}={shlex.quote(value)}"
        for key, value in launch_env(args, rank).items()
    )
    if args.service == "dsv4-pp4-tp2-ep":
        launch_script = "tools/ds4_launch_dsv4_flash_pp4_tp2_ep.sh"
        log_prefix = "dsv4_pp4_tp2_ep"
    else:
        launch_script = "tools/ds4_launch_dsv4_flash_pp8.sh"
        log_prefix = "dsv4_pp8"
    log = f"~/ds4_logs/{log_prefix}_{log_tag}-rank{rank}.log"
    return repo_command(
        args,
        "mkdir -p ~/ds4_logs && "
        f"setsid -f env {env} {launch_script} "
        f"> {log} 2>&1 < /dev/null && "
        f"echo started rank={rank} log={log}",
    )


def service_url(args: argparse.Namespace) -> str:
    return args.health_url or f"http://127.0.0.1:{args.api_port}/v1/models"


def poll_health(args: argparse.Namespace) -> bool:
    deadline = time.monotonic() + args.health_timeout_s
    started = time.monotonic()
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
            if (
                time.monotonic() - started >= args.startup_fail_fast_s
                and not head_service_process_alive(args)
            ):
                print(
                    "health failed early: no head-node launch/preflight/vLLM process is alive",
                    file=sys.stderr,
                )
                print_head_diagnostics(args)
                return False
            time.sleep(args.health_poll_s)
    print(f"health failed after {args.health_timeout_s:g}s: {last_error}", file=sys.stderr)
    return False


def head_service_process_alive(args: argparse.Namespace) -> bool:
    pattern = (
        "[v]llm.entrypoints.cli.main serve|"
        "[V]LLM::|"
        "[d]s4_launch_dsv4_flash_|"
        "[d]s4_nccl_preflight.py"
    )
    command = f"pgrep -af {shlex.quote(pattern)} >/dev/null"
    try:
        result = subprocess.run(["ssh", args.head_node, command], text=True)
    except OSError:
        return True
    return result.returncode == 0


def print_head_diagnostics(args: argparse.Namespace) -> None:
    command = (
        "printf 'processes:\\n'; "
        "pgrep -af '[v]llm.entrypoints.cli.main serve|[V]LLM::|[d]s4_launch_dsv4_flash_|[d]s4_nccl_preflight.py' || true; "
        "printf '\\nrecent dsv4 logs:\\n'; "
        "ls -td ~/ds4_logs/dsv4_pp*_rank0.log ~/ds4_logs/dsv4_pp*-rank0.log 2>/dev/null | head -3 | "
        "xargs -r -I{} sh -c 'echo ==== {}; tail -80 {}'"
    )
    try:
        result = subprocess.run(
            ["ssh", args.head_node, command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout[-12000:], file=sys.stderr)
    except OSError as exc:
        print(f"failed to collect head diagnostics: {exc}", file=sys.stderr)


def run_on_nodes(args: argparse.Namespace, nodes: list[str], command_factory) -> None:
    failed: list[str] = []
    for rank, node in enumerate(nodes):
        command = command_factory(rank, node)
        if remote_run(node, command, args.dry_run) != 0:
            failed.append(node)
    if failed:
        raise SystemExit(f"failed on node(s): {', '.join(failed)}")


def static_fabric_command(args: argparse.Namespace, mode: str) -> str:
    py = shell_path(args.python)
    return repo_command(
        args,
        f"{py} tools/ds4_static_fabric.py --fleet --{mode} "
        f"--nodes {shlex.quote(','.join(nodes_from_arg(args.nodes)))} "
        f"--nnodes {args.nnodes} "
        f"--edge-rail {shlex.quote(args.static_fabric_edge_rail)} "
        f"--route-scope {shlex.quote(args.static_fabric_route_scope)} "
        f"--timeout-s {args.static_fabric_timeout_s}",
    )


def run_static_fabric(args: argparse.Namespace, nodes: list[str]) -> None:
    if args.static_fabric == "off":
        return
    command = static_fabric_command(args, args.static_fabric)
    if remote_run(args.head_node, command, args.dry_run) != 0:
        if args.static_fabric == "verify":
            bootstrap = static_fabric_command(args, "apply")
            print(
                "static fabric verification failed before service stop; "
                "after a power cycle run this bootstrap once:",
                file=sys.stderr,
            )
            print(f"ssh {args.head_node} {shlex.quote(bootstrap)}", file=sys.stderr)
        raise SystemExit("static fabric setup failed")
    if len(nodes) != args.nnodes:
        raise SystemExit(f"--nodes has {len(nodes)} entries but --nnodes={args.nnodes}")


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
    timer = LaunchTimer()
    args = parse_args()
    try:
        with timer.phase("cleanup_local_controllers"):
            cleanup_stale_local_controllers(args)
        with timer.phase("parse_nodes"):
            nodes = nodes_from_arg(args.nodes)
            if len(nodes) != args.nnodes:
                raise SystemExit(
                    f"--nodes has {len(nodes)} entries but --nnodes={args.nnodes}"
                )
            log_tag = args.log_tag or (
                "pr44_workspace_"
                + args.profile.replace("-", "_")
                + "_"
                + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            )
        if not args.skip_pull:
            with timer.phase("git_pull"):
                run_on_nodes(args, nodes, lambda _rank, _node: pull_command(args))
        if not args.skip_build:
            with timer.phase("build_validate"):
                run_on_nodes(args, nodes, lambda _rank, _node: build_command(args))
        with timer.phase("static_fabric"):
            run_static_fabric(args, nodes)
        if args.setup_only:
            print(
                "setup-only complete: pull/build/static fabric finished; "
                "service was not stopped or launched"
            )
            return 0
        if not args.skip_stop:
            with timer.phase("stop_old_service"):
                run_on_nodes(args, nodes, lambda _rank, _node: stop_command(args))
        with timer.phase("start_ranks"):
            run_parallel_start(args, nodes, log_tag)
        if args.dry_run:
            return 0
        with timer.phase("health_ready"):
            return 0 if poll_health(args) else 1
    finally:
        timer.summary()


if __name__ == "__main__":
    raise SystemExit(main())
