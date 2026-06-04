#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "vllm" / "entrypoints" / "openai" / "completion" / "serving.py"


def main() -> int:
    text = SERVING.read_text(encoding="utf-8")
    checks = {
        "lock is initialized": "_ds4_completion_cohort_admission_lock = asyncio.Lock()" in text,
        "admission path takes lock": "async with self._ds4_completion_cohort_admission_lock:" in text,
        "scheduler pause remains explicit": "pause_generation(" in text and "mode=\"keep\"" in text and "clear_cache=False" in text,
        "scheduler wake remains verified": "DS4 cohort admission left the scheduler paused after wake_up" in text,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        for name in failed:
            print(f"FAIL {name}")
        return 1
    for name in checks:
        print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
