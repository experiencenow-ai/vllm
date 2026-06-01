#!/usr/bin/env python3
"""Static checks for DS4 workspace preallocation.

The c512 DSV4 path must not discover a larger MoE scratch workspace after CUDA
graph capture has locked the v1 workspace manager. Production launchers should
reserve the intended scratch size before lock and still fail closed if a later
hot-path allocation exceeds it.
"""

from pathlib import Path
import sys


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
workspace = (root / "vllm/v1/worker/workspace.py").read_text()
dsv4_pp8 = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
dsv4_tp2 = (
    root / "tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh"
).read_text()

checks = [
    (
        "workspace prealloc env is registered",
        "VLLM_WORKSPACE_PREALLOC_BYTES" in envs
        and 'os.getenv("VLLM_WORKSPACE_PREALLOC_BYTES", "0")' in envs,
    ),
    (
        "workspace manager has explicit pre-lock reserve path",
        "def reserve_bytes(self, required_bytes: int) -> None:" in workspace
        and "Cannot reserve workspace bytes after workspace is locked." in workspace,
    ),
    (
        "lock reserves before locking",
        "workspace_manager.reserve_bytes(envs.VLLM_WORKSPACE_PREALLOC_BYTES)"
        in workspace
        and "workspace_manager.lock()" in workspace
        and workspace.index(
            "workspace_manager.reserve_bytes(envs.VLLM_WORKSPACE_PREALLOC_BYTES)"
        )
        < workspace.index("workspace_manager.lock()"),
    ),
    (
        "locked workspace still fails closed on late growth",
        "Workspace is locked but allocation from" in workspace
        and "Workspace growth is not allowed after locking." in workspace,
    ),
    (
        "DSV4 PP8 profiles set workspace prealloc defaults",
        "DSV4_WORKSPACE_PREALLOC_BYTES:=1073741824" in dsv4_pp8
        and "DSV4_WORKSPACE_PREALLOC_BYTES:=805306368" in dsv4_pp8
        and "DSV4_WORKSPACE_PREALLOC_BYTES:=536870912" in dsv4_pp8,
    ),
    (
        "DSV4 PP8 launcher exports vLLM workspace prealloc",
        'export VLLM_WORKSPACE_PREALLOC_BYTES="${VLLM_WORKSPACE_PREALLOC_BYTES:-$DSV4_WORKSPACE_PREALLOC_BYTES}"'
        in dsv4_pp8,
    ),
    (
        "DSV4 TP2 launcher exports vLLM workspace prealloc",
        'export VLLM_WORKSPACE_PREALLOC_BYTES="${VLLM_WORKSPACE_PREALLOC_BYTES:-$DSV4_WORKSPACE_PREALLOC_BYTES}"'
        in dsv4_tp2,
    ),
]

failed = 0
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
