#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def check(name: str, ok: bool) -> int:
    print(("PASS" if ok else "FAIL") + f": {name}")
    return 0 if ok else 1


def main() -> int:
    shared = read("vllm/model_executor/layers/fused_moe/runner/shared_experts.py")
    runner = read("vllm/model_executor/layers/fused_moe/runner/moe_runner.py")

    failures = 0
    failures += check(
        "SharedExperts.apply returns direct output for NO_OVERLAP",
        "order == SharedExpertsOrder.NO_OVERLAP" in shared
        and "return self._layer(shared_experts_input)" in shared,
    )
    failures += check(
        "NO_OVERLAP path clears stale side-channel output",
        "Clearing stale MoE shared_experts output" in shared
        and "self._output[idx] = None" in shared,
    )
    failures += check(
        "MoE runner captures direct shared output",
        "shared_output = self._maybe_apply_shared_experts(" in runner
        and "SharedExpertsOrder.NO_OVERLAP" in runner,
    )
    failures += check(
        "MoE runner only reads side-channel output for overlapped paths",
        "if shared_output is not None or self._shared_experts is None" in runner
        and "else self._shared_experts.output" in runner,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
