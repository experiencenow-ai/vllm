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
    serving = read("vllm/entrypoints/openai/completion/serving.py")
    envs = read("vllm/envs.py")
    dsv4 = read("tools/ds4_launch_dsv4_flash_pp8.sh")

    failures = 0
    failures += check(
        "env exposes DS4 cohort admission switch",
        "VLLM_DS4_COHORT_ADMISSION" in envs
        and "VLLM_DS4_COHORT_ADMISSION_MIN_PROMPTS" in envs,
    )
    failures += check(
        "OpenAI completions use cohort admission guard",
        "use_ds4_cohort_admission" in serving
        and "envs.VLLM_DS4_COHORT_ADMISSION" in serving
        and "envs.VLLM_DS4_COHORT_ADMISSION_MIN_PROMPTS" in serving,
    )
    failures += check(
        "cohort admission pauses and resumes scheduler",
        'pause_generation(mode="keep", clear_cache=False)' in serving
        and "resume_generation()" in serving,
    )
    failures += check(
        "cohort admission calls add_request directly",
        'getattr(self.engine_client, "add_request", None)' in serving
        and "collector = await add_request(" in serving,
    )
    failures += check(
        "missing direct add_request fails closed",
        "refusing to split" in serving
        and "DS4 cohort admission failed before generation" in serving,
    )
    failures += check(
        "collector generators drain RequestOutput directly",
        "collector.get_nowait() or await collector.get()" in serving
        and "STREAM_FINISHED" in serving,
    )
    failures += check(
        "DSV4 PP8 enables cohort admission by default",
        'VLLM_DS4_COHORT_ADMISSION="${VLLM_DS4_COHORT_ADMISSION:-1}"'
        in dsv4
        and "VLLM_DS4_COHORT_ADMISSION_MIN_PROMPTS" in dsv4,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
