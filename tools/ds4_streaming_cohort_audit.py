#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "vllm" / "entrypoints" / "openai" / "completion" / "serving.py"


def main() -> int:
    text = SERVING.read_text()
    marker = "use_ds4_cohort_admission = ("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing DS4 cohort admission block in {SERVING}")
    end = text.find("\n        for i, engine_input in enumerate(engine_inputs):", start)
    if end < 0:
        raise SystemExit(f"unterminated DS4 cohort admission block in {SERVING}")
    block = text[start:end]
    required = [
        "envs.VLLM_DS4_COHORT_ADMISSION",
        "not request.use_beam_search",
        "len(engine_inputs) >= envs.VLLM_DS4_COHORT_ADMISSION_MIN_PROMPTS",
    ]
    for needle in required:
        if needle not in block:
            raise SystemExit(f"DS4 cohort admission block missing {needle!r}")
    if "not request.stream" in block:
        raise SystemExit(
            "DS4 cohort admission must apply to streaming prompt-array "
            "completions; DS API uses internal SSE for row-level completion."
        )
    print("PASS DS4 streaming prompt-array completions use cohort admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
