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
    envs = read("vllm/envs.py")
    scheduler = read("vllm/v1/core/sched/scheduler.py")
    dsv4 = read("tools/ds4_launch_dsv4_flash_pp8.sh")
    qwen = read("tools/ds4_launch_qwen27_nvfp4_pp8.sh")
    relaunch = read("tools/ds4_relaunch_spark_service.py")
    runner = read("vllm/v1/worker/gpu_model_runner.py")

    failures = 0
    failures += check(
        "env exposes DS4 PP new-request wave cap",
        "VLLM_DS4_SCHED_MAX_NEW_REQS_PER_STEP" in envs
        and "deterministic PP fill waves" in envs,
    )
    failures += check(
        "scheduler imports envs for DS4 wave cap",
        "import vllm.envs as envs" in scheduler,
    )
    failures += check(
        "scheduler caps waiting request admission only for PP services",
        "if self.use_pp:" in scheduler
        and "envs.VLLM_DS4_SCHED_MAX_NEW_REQS_PER_STEP" in scheduler
        and "ds4_new_reqs_scheduled_this_step" in scheduler,
    )
    failures += check(
        "scheduler caps new prefill tokens separately from decode capacity",
        "VLLM_DS4_SCHED_MAX_NEW_PREFILL_TOKENS_PER_STEP" in envs
        and "envs.VLLM_DS4_SCHED_MAX_NEW_PREFILL_TOKENS_PER_STEP" in scheduler
        and "ds4_new_prefill_tokens_scheduled_this_step" in scheduler
        and "remaining_prefill_tokens" in scheduler,
    )
    failures += check(
        "scheduler increments new request wave count after admission",
        "self.running.append(request)" in scheduler
        and "ds4_new_reqs_scheduled_this_step += 1" in scheduler,
    )
    failures += check(
        "DSV4 PP launcher uses conveyor-shaped request waves for throughput profiles",
        'VLLM_DS4_COHORT_PAUSE_DURING_ADMISSION="${VLLM_DS4_COHORT_PAUSE_DURING_ADMISSION:-1}"' in dsv4
        and 'DSV4_SCHED_MAX_NEW_REQS_PER_STEP:=64' in dsv4
        and 'DSV4_SCHED_MAX_NEW_PREFILL_TOKENS_PER_STEP:=131072' in dsv4
        and 'DSV4_SCHED_MAX_NEW_PREFILL_TOKENS_PER_STEP:=262144' in dsv4,
    )
    failures += check(
        "DS4 fused execute+sample fast path is wired",
        "VLLM_DS4_FUSED_EXECUTE_SAMPLE" in envs
        and "execute_model_and_sample_tokens" in read("vllm/v1/engine/core.py")
        and "execute_model_and_sample_tokens" in read("vllm/v1/worker/gpu_worker.py")
        and 'DSV4_FUSED_EXECUTE_SAMPLE:-1' in dsv4,
    )
    failures += check(
        "DS4 fused execute+sample is decode-only",
        "iteration_details.num_ctx_requests == 0" in read("vllm/v1/engine/core.py")
        and "DS4 fused execute+sample is decode-only" in read("vllm/v1/worker/gpu_worker.py"),
    )
    failures += check(
        "DS4 PP worker timing separates recv, forward, and send",
        "DS4 PP worker timing" in read("vllm/v1/worker/gpu_worker.py")
        and "DS4 PP recv timing" in read("vllm/v1/worker/gpu_worker.py")
        and "prev_send_wait_ms" in read("vllm/v1/worker/gpu_worker.py")
        and "recv_setup_ms" in read("vllm/v1/worker/gpu_worker.py")
        and "send_setup_ms" in read("vllm/v1/worker/gpu_worker.py"),
    )
    failures += check(
        "Qwen PP launcher exposes optional new-request wave cap without forcing it",
        'QWEN27_SCHED_MAX_NEW_REQS_PER_STEP:-0' in qwen,
    )
    failures += check(
        "DSV4 throughput launcher keeps profile debug off by default",
        'VLLM_DS4_PROFILE_DEBUG="${VLLM_DS4_PROFILE_DEBUG:-0}"' in dsv4,
    )
    failures += check(
        "DSV4 iteration-detail logging is opt-in for benchmarks",
        "DSV4_ENABLE_LOGGING_ITERATION_DETAILS" in dsv4
        and "--enable-logging-iteration-details" in dsv4
        and '"${LOGGING_ITERATION_ARGS[@]}"' in dsv4,
    )
    failures += check(
        "Spark relaunch can pass explicit rank env without editing scripts",
        '"--env"' in relaunch
        and "KEY=VALUE" in relaunch
        and '"VLLM_DEBUG_WORKSPACE": os.getenv("VLLM_DEBUG_WORKSPACE", "0")' in relaunch,
    )
    failures += check(
        "padded graph token lanes are zeroed before embedding",
        "_zero_padded_input_token_lanes" in runner
        and "self.input_ids.gpu[num_scheduled_tokens:num_input_tokens].zero_()" in runner
        and "self._zero_padded_input_token_lanes(" in runner
        and "if is_first_rank:" in runner,
    )
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
