#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static audit for the isolated DS4 NCCL/PyNCCL P2P benchmark."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def check(label: str, ok: bool) -> None:
    print(("PASS" if ok else "FAIL") + ": " + label)
    if not ok:
        raise SystemExit(1)


bench = read("tools/ds4_nccl_p2p_bench.py")
runner = read("tools/ds4_run_nccl_p2p_bench.py")

check("rank-local benchmark exists", "DS4 NCCL/PyNCCL point-to-point benchmark" in bench)
check("benchmark compares torch P2P", "method == \"torch\"" in bench and "dist.batch_isend_irecv" in bench)
check("benchmark compares single PyNCCL P2P", "method == \"pynccl\"" in bench and "PyNcclCommunicator" in bench)
check("benchmark compares striped PyNCCL P2P", "method == \"striped\"" in bench and "Ds4StripedNcclTensorChannel" in bench)
check("benchmark emits JSON rows", "json.dumps(row, sort_keys=True)" in bench)
check("benchmark supports adjacent pair selection", "DS4_NCCL_P2P_BENCH_PAIRS" in bench)
check("runner launches all ranks over ssh", "subprocess.Popen([\"ssh\", node, command]" in runner)
check("runner can stop a service before isolated testing", "--stop-service" in runner and "ds4_stop_spark_processes.py" in runner)
check("runner can pull/build before testing", "--pull" in runner and "--build" in runner)
check("runner supports per-rank interface lists", "--nccl-ifnames" in runner and "--gloo-ifnames" in runner and "_rank_value" in runner)
