#!/usr/bin/env python3
"""Rail-aware TCP preflight for the DS4 Spark 200G fabric.

This intentionally mirrors the ds4_transfer.fast_copy data-plane shape:
explicit source-rail binding, destination next-hop IPs from `ip route show`,
and many unencrypted TCP streams per edge. It is not an NCCL collective test.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import sys
import threading
import time


@dataclass(frozen=True)
class Rail:
    source_ip: str
    destination_ip: str
    dev: str


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _rank() -> int:
    return int(_env("RANK", _env("NODE_RANK")))


def _world_size() -> int:
    return int(_env("WORLD_SIZE", _env("NNODES")))


def _fabric_ips(world_size: int) -> list[str]:
    raw = _env("DS4_RAIL_TCP_FABRIC_IPS", "")
    if raw:
        ips = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        ips = [f"10.10.100.{10 + rank}" for rank in range(world_size)]
    if len(ips) != world_size:
        raise ValueError(
            "DS4_RAIL_TCP_FABRIC_IPS must contain one IP per rank "
            f"({len(ips)} != WORLD_SIZE={world_size})"
        )
    return ips


def _parse_pairs(world_size: int) -> list[tuple[int, int]]:
    raw = _env(
        "DS4_RAIL_TCP_PREFLIGHT_PAIRS",
        _env("DS4_NCCL_PREFLIGHT_P2P_PAIRS", ""),
    )
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
            raise ValueError(f"invalid rail TCP pair {item!r}")
        src = int(left.strip())
        dst = int(right.strip())
        if src == dst:
            raise ValueError(f"invalid self rail TCP pair {item!r}")
        for rank in (src, dst):
            if rank < 0 or rank >= world_size:
                raise ValueError(f"rank {rank} outside WORLD_SIZE={world_size}")
        pairs.append((src, dst))
    if not pairs:
        raise ValueError("DS4_RAIL_TCP_PREFLIGHT_PAIRS did not contain any pairs")
    return pairs


def _run(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(argv)} failed with {result.returncode}: {result.stderr}"
        )
    return result.stdout


def _source_ip_for_dev(dev: str) -> str:
    output = _run(["ip", "-4", "-o", "addr", "show", "dev", dev, "scope", "global"])
    for line in output.splitlines():
        for token in line.split():
            if "/" in token and token[0].isdigit():
                return token.split("/", 1)[0]
    raise RuntimeError(f"no source IP for device {dev}")


def _discover_rails(destination_fabric_ip: str) -> list[Rail]:
    route = _run(["ip", "route", "show", destination_fabric_ip])
    rails: list[Rail] = []
    tokens = route.replace("\n", " ").split()
    for index, token in enumerate(tokens):
        if token == "via" and index + 3 < len(tokens) and tokens[index + 2] == "dev":
            dst_ip = tokens[index + 1]
            dev = tokens[index + 3]
            rails.append(
                Rail(
                    source_ip=_source_ip_for_dev(dev),
                    destination_ip=dst_ip,
                    dev=dev,
                )
            )
    if rails:
        return rails

    fallback = _run(["ip", "route", "get", destination_fabric_ip])
    words = fallback.split()
    if "dev" not in words:
        raise RuntimeError(
            f"could not discover 200G rail to {destination_fabric_ip}: {fallback}"
        )
    dev = words[words.index("dev") + 1]
    if "via" in words:
        dst_ip = words[words.index("via") + 1]
    else:
        dst_ip = destination_fabric_ip
    if "src" in words:
        source_ip = words[words.index("src") + 1]
    else:
        source_ip = _source_ip_for_dev(dev)
    return [Rail(source_ip=source_ip, destination_ip=dst_ip, dev=dev)]


def _server_stream(port: int, expected_bytes: int, timeout_s: float) -> int:
    bind_ip = _env("DS4_RAIL_TCP_PREFLIGHT_SERVER_BIND", "0.0.0.0")
    script = (
        f"set -o pipefail; "
        f"nc -l -s {bind_ip} -p {port} | wc -c"
    )
    result = subprocess.run(
        ["bash", "-lc", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rail TCP nc server failed on port {port}: {result.stderr[-500:]}"
        )
    try:
        received = int(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"rail TCP nc server on port {port} produced invalid byte count: "
            f"{result.stdout!r}"
        ) from exc
    if received != expected_bytes:
        raise RuntimeError(
            f"rail TCP nc server on port {port} received {received}, "
            f"expected {expected_bytes}"
        )
    return received


def _client_stream(rail: Rail, port: int, bytes_to_send: int, timeout_s: float) -> int:
    script = (
        "set -o pipefail; "
        f"head -c {bytes_to_send} /dev/zero | "
        f"nc -N -s {rail.source_ip} {rail.destination_ip} {port}"
    )
    deadline = time.monotonic() + timeout_s
    last_stderr = ""
    while time.monotonic() < deadline:
        remaining = max(1.0, deadline - time.monotonic())
        result = subprocess.run(
            ["bash", "-lc", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=remaining,
            check=False,
        )
        if result.returncode == 0:
            return bytes_to_send
        last_stderr = result.stderr[-500:]
        time.sleep(0.1)
    raise RuntimeError(
        f"rail TCP nc client failed via {rail.source_ip}->{rail.destination_ip}:"
        f"{port}: {last_stderr}"
    )


def _preflight_tool() -> str:
    tool = _env("DS4_RAIL_TCP_PREFLIGHT_TOOL", "iperf3").strip().lower()
    if tool not in {"iperf3", "nc"}:
        raise ValueError("DS4_RAIL_TCP_PREFLIGHT_TOOL must be iperf3 or nc")
    if tool == "iperf3" and shutil.which("iperf3") is None:
        raise RuntimeError(
            "DS4_RAIL_TCP_PREFLIGHT_TOOL=iperf3 but iperf3 is not installed"
        )
    if tool == "nc" and shutil.which("nc") is None:
        raise RuntimeError("DS4_RAIL_TCP_PREFLIGHT_TOOL=nc but nc is not installed")
    return tool


def _run_iperf3_server(pair_index: int, src: int, dst: int) -> int:
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    port = port_base + (pair_index * 100)
    bind_ip = _env("DS4_RAIL_TCP_PREFLIGHT_SERVER_BIND", "0.0.0.0")
    duration_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_DURATION_S", "5"))
    timeout_s = duration_s + float(_env("DS4_RAIL_TCP_PREFLIGHT_TIMEOUT", "30"))
    result = subprocess.run(
        ["iperf3", "-s", "-B", bind_ip, "-p", str(port), "-1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rail TCP iperf3 server failed for pair={src}-{dst} "
            f"port={port}: {result.stderr[-500:] or result.stdout[-500:]}"
        )
    print(
        "DS4 rail TCP preflight iperf3 server complete: "
        f"pair={src}-{dst} port={port}",
        file=sys.stderr,
    )
    return 0


def _run_iperf3_client(
    pair_index: int,
    src: int,
    dst: int,
    destination_ip: str,
) -> int:
    streams = max(1, int(_env("DS4_RAIL_TCP_PREFLIGHT_STREAMS", "16")))
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    port = port_base + (pair_index * 100)
    duration_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_DURATION_S", "5"))
    timeout_s = duration_s + float(_env("DS4_RAIL_TCP_PREFLIGHT_TIMEOUT", "30"))
    min_gbps = float(
        _env(
            "DS4_RAIL_TCP_PREFLIGHT_MIN_GBPS",
            _env("DS4_NCCL_PREFLIGHT_MIN_P2P_GBPS", "0"),
        )
    )
    rails = _discover_rails(destination_ip)
    rail = rails[0]
    time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_CLIENT_DELAY_S", "0.5")))
    result = subprocess.run(
        [
            "iperf3",
            "-c",
            rail.destination_ip,
            "-B",
            rail.source_ip,
            "-p",
            str(port),
            "-P",
            str(streams),
            "-t",
            str(duration_s),
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rail TCP iperf3 client failed for pair={src}-{dst} "
            f"{rail.source_ip}->{rail.destination_ip}:{port}: "
            f"{result.stderr[-500:] or result.stdout[-500:]}"
        )
    data = json.loads(result.stdout)
    end = data.get("end", {})
    summary = end.get("sum_sent") or end.get("sum") or {}
    bits_per_second = float(summary.get("bits_per_second", 0.0))
    gbps = bits_per_second / 8.0 / 1e9
    rails_text = ",".join(
        f"{item.source_ip}->{item.destination_ip}/{item.dev}" for item in rails
    )
    print(
        "DS4 rail TCP preflight bandwidth: "
        f"pair={src}-{dst} role=iperf3-client streams={streams} "
        f"duration_s={duration_s:.3f} rails={rails_text} "
        f"GBps={gbps:.3f} Gbit_s={(gbps * 8.0):.3f} "
        f"min_GBps={min_gbps:.3f}",
        file=sys.stderr,
    )
    if min_gbps > 0 and gbps < min_gbps:
        return 68
    return 0


def _run_server(pair_index: int, src: int, dst: int) -> int:
    if _preflight_tool() == "iperf3":
        return _run_iperf3_server(pair_index, src, dst)
    streams = max(1, int(_env("DS4_RAIL_TCP_PREFLIGHT_STREAMS", "16")))
    total_bytes = max(streams, int(_env("DS4_RAIL_TCP_PREFLIGHT_BYTES", "268435456")))
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    timeout_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_TIMEOUT", "30"))
    per_stream = total_bytes // streams
    remainder = total_bytes % streams
    received = 0
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(slot: int) -> None:
        nonlocal received
        expected = per_stream + (1 if slot < remainder else 0)
        try:
            count = _server_stream(
                port_base + (pair_index * 100) + slot,
                expected,
                timeout_s,
            )
            with lock:
                received += count
        except BaseException as exc:  # noqa: BLE001 - report exact thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(streams)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout_s + 5.0)
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    for thread in threads:
        if thread.is_alive():
            errors.append(RuntimeError("server stream thread timed out"))
    if errors:
        raise RuntimeError("; ".join(str(error) for error in errors))
    gbps = (received / elapsed_s) / 1e9
    print(
        "DS4 rail TCP preflight bandwidth: "
        f"pair={src}-{dst} role=server bytes={received} streams={streams} "
        f"elapsed_s={elapsed_s:.6f} GBps={gbps:.3f} "
        f"Gbit_s={(gbps * 8.0):.3f}",
        file=sys.stderr,
    )
    return 0


def _run_client(pair_index: int, src: int, dst: int, destination_ip: str) -> int:
    if _preflight_tool() == "iperf3":
        return _run_iperf3_client(pair_index, src, dst, destination_ip)
    streams = max(1, int(_env("DS4_RAIL_TCP_PREFLIGHT_STREAMS", "16")))
    total_bytes = max(streams, int(_env("DS4_RAIL_TCP_PREFLIGHT_BYTES", "268435456")))
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    timeout_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_TIMEOUT", "30"))
    min_gbps = float(
        _env(
            "DS4_RAIL_TCP_PREFLIGHT_MIN_GBPS",
            _env("DS4_NCCL_PREFLIGHT_MIN_P2P_GBPS", "0"),
        )
    )
    rails = _discover_rails(destination_ip)
    per_stream = total_bytes // streams
    remainder = total_bytes % streams
    sent = 0
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(slot: int) -> None:
        nonlocal sent
        rail = rails[slot % len(rails)]
        expected = per_stream + (1 if slot < remainder else 0)
        try:
            count = _client_stream(
                rail,
                port_base + (pair_index * 100) + slot,
                expected,
                timeout_s,
            )
            with lock:
                sent += count
        except BaseException as exc:  # noqa: BLE001 - report exact thread failure
            errors.append(exc)

    time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_CLIENT_DELAY_S", "0.5")))
    start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout_s + 5.0)
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    for thread in threads:
        if thread.is_alive():
            errors.append(RuntimeError("client stream thread timed out"))
    if errors:
        raise RuntimeError("; ".join(str(error) for error in errors))
    gbps = (sent / elapsed_s) / 1e9
    rails_text = ",".join(
        f"{rail.source_ip}->{rail.destination_ip}/{rail.dev}" for rail in rails
    )
    print(
        "DS4 rail TCP preflight bandwidth: "
        f"pair={src}-{dst} role=client bytes={sent} streams={streams} "
        f"rails={rails_text} elapsed_s={elapsed_s:.6f} GBps={gbps:.3f} "
        f"Gbit_s={(gbps * 8.0):.3f} min_GBps={min_gbps:.3f}",
        file=sys.stderr,
    )
    if min_gbps > 0 and gbps < min_gbps:
        return 68
    return 0


def main() -> int:
    rank = _rank()
    world_size = _world_size()
    fabric_ips = _fabric_ips(world_size)
    pairs = _parse_pairs(world_size)
    print(
        "DS4 rail TCP preflight starting: "
        f"rank={rank}/{world_size} pairs="
        + ";".join(f"{src}-{dst}" for src, dst in pairs),
        file=sys.stderr,
    )
    try:
        for pair_index, (src, dst) in enumerate(pairs):
            if rank == dst:
                status = _run_server(pair_index, src, dst)
                if status != 0:
                    return status
            elif rank == src:
                status = _run_client(pair_index, src, dst, fabric_ips[dst])
                if status != 0:
                    return status
        print(f"DS4 rail TCP preflight passed on rank {rank}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(
            f"DS4 rail TCP preflight failed on rank {rank}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 67


if __name__ == "__main__":
    raise SystemExit(main())
