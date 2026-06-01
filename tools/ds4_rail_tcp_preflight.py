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
import signal
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
    tool = _env("DS4_RAIL_TCP_PREFLIGHT_TOOL", "iperf").strip().lower()
    if tool not in {"iperf", "iperf3", "nc"}:
        raise ValueError("DS4_RAIL_TCP_PREFLIGHT_TOOL must be iperf, iperf3, or nc")
    if tool == "iperf" and shutil.which("iperf") is None:
        raise RuntimeError("DS4_RAIL_TCP_PREFLIGHT_TOOL=iperf but iperf is not installed")
    if tool == "iperf3" and shutil.which("iperf3") is None:
        raise RuntimeError(
            "DS4_RAIL_TCP_PREFLIGHT_TOOL=iperf3 but iperf3 is not installed"
        )
    if tool == "nc" and shutil.which("nc") is None:
        raise RuntimeError("DS4_RAIL_TCP_PREFLIGHT_TOOL=nc but nc is not installed")
    return tool


def _gbit_threshold(primary: str, legacy_gbytes: str, default: str = "0") -> float:
    raw = _env(primary, "")
    if raw:
        return float(raw)
    raw = _env(legacy_gbytes, "")
    if raw:
        return (float(raw) * 8.0)
    return float(default)


def _fail_gbit_s() -> float:
    raw = _env("DS4_RAIL_TCP_PREFLIGHT_MIN_GBIT_S", "")
    if raw:
        return float(raw)
    raw = _env("DS4_RAIL_TCP_PREFLIGHT_MIN_GBPS", "")
    if raw:
        return (float(raw) * 8.0)
    raw = _env("DS4_NCCL_PREFLIGHT_MIN_P2P_GBPS", "")
    if raw:
        return (float(raw) * 8.0)
    return 0.0


def _warn_gbit_s() -> float:
    return _gbit_threshold(
        "DS4_RAIL_TCP_PREFLIGHT_WARN_GBIT_S",
        "DS4_RAIL_TCP_PREFLIGHT_WARN_GBPS",
    )


def _report_client_bandwidth(
    *,
    pair: str,
    role: str,
    streams: int,
    duration_s: float | None,
    rails_text: str,
    gbps: float,
    extra: str = "",
) -> int:
    measured_gbit_s = (gbps * 8.0)
    fail_gbit_s = _fail_gbit_s()
    warn_gbit_s = _warn_gbit_s()
    duration_text = "" if duration_s is None else f"duration_s={duration_s:.3f} "
    print(
        "DS4 rail TCP preflight bandwidth: "
        f"pair={pair} role={role} streams={streams} "
        f"{duration_text}rails={rails_text} {extra}"
        f"GBps={gbps:.3f} Gbit_s={measured_gbit_s:.3f} "
        f"fail_Gbit_s={fail_gbit_s:.3f} warn_Gbit_s={warn_gbit_s:.3f}",
        file=sys.stderr,
    )
    if warn_gbit_s > 0 and measured_gbit_s < warn_gbit_s:
        print(
            "WARNING: DS4 rail TCP preflight below warning threshold: "
            f"pair={pair} measured_Gbit_s={measured_gbit_s:.3f} "
            f"warn_Gbit_s={warn_gbit_s:.3f} fail_Gbit_s={fail_gbit_s:.3f} "
            "launch continues unless the fail threshold is crossed",
            file=sys.stderr,
        )
    if fail_gbit_s > 0 and measured_gbit_s < fail_gbit_s:
        print(
            "DS4 rail TCP preflight failed: "
            f"pair={pair} measured_Gbit_s={measured_gbit_s:.3f} "
            f"< required {fail_gbit_s:.3f} Gbit/s",
            file=sys.stderr,
        )
        return 68
    return 0


def _run_iperf_server(pair_index: int, src: int, dst: int) -> int:
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    port = port_base + (pair_index * 100)
    bind_ip = _env("DS4_RAIL_TCP_PREFLIGHT_SERVER_BIND", "0.0.0.0")
    duration_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_DURATION_S", "5"))
    timeout_s = duration_s + float(_env("DS4_RAIL_TCP_PREFLIGHT_TIMEOUT", "30"))
    result = subprocess.run(
        ["iperf", "-s", "-1", "-B", bind_ip, "-p", str(port), "-f", "g", "-y", "C"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rail TCP iperf server failed for pair={src}-{dst} "
            f"port={port}: {result.stderr[-500:] or result.stdout[-500:]}"
        )
    print(
        "DS4 rail TCP preflight iperf server complete: "
        f"pair={src}-{dst} port={port}",
        file=sys.stderr,
    )
    return 0


def _iperf_server_command(pair_index: int) -> list[str]:
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    port = port_base + (pair_index * 100)
    bind_ip = _env("DS4_RAIL_TCP_PREFLIGHT_SERVER_BIND", "0.0.0.0")
    return ["iperf", "-s", "-B", bind_ip, "-p", str(port), "-f", "g", "-y", "C"]


def _parse_iperf_csv_bits_per_second(stdout: str) -> float:
    duration_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_DURATION_S", "5"))
    min_interval_s = float(
        _env(
            "DS4_RAIL_TCP_PREFLIGHT_MIN_REPORT_INTERVAL_S",
            str(max(1.0, duration_s * 0.5)),
        )
    )
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 9:
            continue
        if fields[5] != "-1":
            continue
        try:
            left, right = fields[6].split("-", 1)
            interval_s = float(right) - float(left)
        except ValueError:
            continue
        if interval_s < min_interval_s:
            continue
        try:
            summary = float(fields[-1])
        except ValueError:
            continue
        if summary > 0:
            return summary
    raise RuntimeError(
        "could not parse iperf CSV summary bandwidth from a real -1 interval: "
        f"{stdout!r}"
    )


def _run_iperf_client(
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
    rails = _discover_rails(destination_ip)
    rail = rails[0]
    time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_CLIENT_DELAY_S", "0.5")))
    deadline = time.monotonic() + timeout_s
    last_error = ""
    bits_per_second = 0.0
    while time.monotonic() < deadline:
        remaining_s = max(1.0, deadline - time.monotonic())
        result = subprocess.run(
            [
                "iperf",
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
                "-f",
                "g",
                "-y",
                "C",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=remaining_s,
            check=False,
        )
        if result.returncode == 0:
            try:
                bits_per_second = _parse_iperf_csv_bits_per_second(result.stdout)
                break
            except RuntimeError as exc:
                last_error = str(exc)
        else:
            last_error = result.stderr[-500:] or result.stdout[-500:]
        time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_CLIENT_RETRY_SLEEP_S", "0.5")))
    if bits_per_second <= 0:
        raise RuntimeError(
            f"rail TCP iperf client failed for pair={src}-{dst} "
            f"{rail.source_ip}->{rail.destination_ip}:{port}: {last_error}"
        )
    gbps = bits_per_second / 8.0 / 1e9
    rails_text = ",".join(
        f"{item.source_ip}->{item.destination_ip}/{item.dev}" for item in rails
    )
    return _report_client_bandwidth(
        pair=f"{src}-{dst}",
        role="iperf-client",
        streams=streams,
        duration_s=duration_s,
        rails_text=rails_text,
        gbps=gbps,
    )


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


def _iperf3_server_command(pair_index: int) -> list[str]:
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    port = port_base + (pair_index * 100)
    bind_ip = _env("DS4_RAIL_TCP_PREFLIGHT_SERVER_BIND", "0.0.0.0")
    return ["iperf3", "-s", "-B", bind_ip, "-p", str(port)]


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
    rails = _discover_rails(destination_ip)
    rail = rails[0]
    time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_CLIENT_DELAY_S", "0.5")))
    deadline = time.monotonic() + timeout_s
    last_error = ""
    bits_per_second = 0.0
    while time.monotonic() < deadline:
        remaining_s = max(1.0, deadline - time.monotonic())
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
            timeout=remaining_s,
            check=False,
        )
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                end = data.get("end", {})
                summary = end.get("sum_sent") or end.get("sum") or {}
                bits_per_second = float(summary.get("bits_per_second", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = str(exc)
                bits_per_second = 0.0
            if bits_per_second > 0:
                break
        else:
            last_error = result.stderr[-500:] or result.stdout[-500:]
        time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_CLIENT_RETRY_SLEEP_S", "0.5")))
    if bits_per_second <= 0:
        raise RuntimeError(
            f"rail TCP iperf3 client failed for pair={src}-{dst} "
            f"{rail.source_ip}->{rail.destination_ip}:{port}: {last_error}"
        )
    gbps = bits_per_second / 8.0 / 1e9
    rails_text = ",".join(
        f"{item.source_ip}->{item.destination_ip}/{item.dev}" for item in rails
    )
    return _report_client_bandwidth(
        pair=f"{src}-{dst}",
        role="iperf3-client",
        streams=streams,
        duration_s=duration_s,
        rails_text=rails_text,
        gbps=gbps,
    )


def _run_server(pair_index: int, src: int, dst: int) -> int:
    tool = _preflight_tool()
    if tool == "iperf":
        return _run_iperf_server(pair_index, src, dst)
    if tool == "iperf3":
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
    tool = _preflight_tool()
    if tool == "iperf":
        return _run_iperf_client(pair_index, src, dst, destination_ip)
    if tool == "iperf3":
        return _run_iperf3_client(pair_index, src, dst, destination_ip)
    streams = max(1, int(_env("DS4_RAIL_TCP_PREFLIGHT_STREAMS", "16")))
    total_bytes = max(streams, int(_env("DS4_RAIL_TCP_PREFLIGHT_BYTES", "268435456")))
    port_base = int(_env("DS4_RAIL_TCP_PREFLIGHT_PORT_BASE", "49400"))
    timeout_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_TIMEOUT", "30"))
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
    return _report_client_bandwidth(
        pair=f"{src}-{dst}",
        role="client",
        streams=streams,
        duration_s=None,
        rails_text=rails_text,
        gbps=gbps,
        extra=f"bytes={sent} elapsed_s={elapsed_s:.6f} ",
    )


def _start_bandwidth_server(tool: str, pair_index: int, src: int, dst: int) -> subprocess.Popen[str]:
    if tool == "iperf":
        argv = _iperf_server_command(pair_index)
    elif tool == "iperf3":
        argv = _iperf3_server_command(pair_index)
    else:
        raise ValueError(f"background server is not supported for {tool}")
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    print(
        "DS4 rail TCP preflight server started: "
        f"pair={src}-{dst} tool={tool} pid={proc.pid}",
        file=sys.stderr,
    )
    return proc


def _check_background_servers(
    servers: list[tuple[int, int, int, subprocess.Popen[str]]],
) -> None:
    for pair_index, src, dst, proc in servers:
        status = proc.poll()
        if status is None:
            continue
        stdout, stderr = proc.communicate(timeout=1)
        raise RuntimeError(
            "rail TCP preflight server exited before client phase: "
            f"pair={src}-{dst} index={pair_index} status={status}: "
            f"{(stderr or stdout)[-500:]}"
        )


def _stop_background_servers(
    servers: list[tuple[int, int, int, subprocess.Popen[str]]],
) -> None:
    stop_timeout_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_SERVER_STOP_TIMEOUT_S", "2"))
    for pair_index, src, dst, proc in servers:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            stdout, stderr = proc.communicate(timeout=stop_timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate(timeout=stop_timeout_s)
        text = (stderr or stdout or "").strip()
        if text:
            text = " " + text.splitlines()[-1][-300:]
        print(
            "DS4 rail TCP preflight server stopped: "
            f"pair={src}-{dst} index={pair_index} status={proc.returncode}{text}",
            file=sys.stderr,
        )


def _run_bandwidth_preflight(
    rank: int,
    fabric_ips: list[str],
    pairs: list[tuple[int, int]],
    tool: str,
) -> int:
    servers: list[tuple[int, int, int, subprocess.Popen[str]]] = []
    server_started_at = time.monotonic()
    client_phase_passed = False
    try:
        for pair_index, (src, dst) in enumerate(pairs):
            if rank == dst:
                servers.append(
                    (pair_index, src, dst, _start_bandwidth_server(tool, pair_index, src, dst))
                )
        server_started_at = time.monotonic()
        if servers:
            time.sleep(float(_env("DS4_RAIL_TCP_PREFLIGHT_SERVER_READY_S", "1.0")))
            _check_background_servers(servers)
        for pair_index, (src, dst) in enumerate(pairs):
            if rank == src:
                status = _run_client(pair_index, src, dst, fabric_ips[dst])
                if status != 0:
                    return status
                _check_background_servers(servers)
        client_phase_passed = True
        return 0
    finally:
        if servers and client_phase_passed:
            duration_s = float(_env("DS4_RAIL_TCP_PREFLIGHT_DURATION_S", "5"))
            hold_extra_s = float(
                _env("DS4_RAIL_TCP_PREFLIGHT_SERVER_HOLD_EXTRA_S", "10.0")
            )
            hold_until = server_started_at + duration_s + hold_extra_s
            remaining_s = hold_until - time.monotonic()
            if remaining_s > 0:
                time.sleep(remaining_s)
        _stop_background_servers(servers)


def main() -> int:
    rank = _rank()
    world_size = _world_size()
    fabric_ips = _fabric_ips(world_size)
    pairs = _parse_pairs(world_size)
    tool = _preflight_tool()
    print(
        "DS4 rail TCP preflight starting: "
        f"rank={rank}/{world_size} pairs="
        + ";".join(f"{src}-{dst}" for src, dst in pairs)
        + f" tool={tool}",
        file=sys.stderr,
    )
    try:
        if tool in {"iperf", "iperf3"}:
            status = _run_bandwidth_preflight(rank, fabric_ips, pairs, tool)
            if status != 0:
                return status
            print(f"DS4 rail TCP preflight passed on rank {rank}", file=sys.stderr)
            return 0
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
