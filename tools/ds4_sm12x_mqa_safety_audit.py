#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static audit for DS4 SM12x paged-MQA safety rails."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SM12X_MQA = ROOT / "vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py"
PP8 = ROOT / "tools/ds4_launch_dsv4_flash_pp8.sh"
TP2 = ROOT / "tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh"

text = SM12X_MQA.read_text()
checks = [
    ("context_lens_normalizer", "def _normalize_context_lens_for_next_n" in text),
    ("rowwise_gate", "def _should_use_rowwise_paged_mqa" in text),
    ("rowwise_max_rows_env", "VLLM_DS4_SM12X_MQA_ROWWISE_MAX_ROWS" in text),
    ("kv_block_bounds_param", "num_kv_blocks: tl.constexpr" in text),
    ("block_table_bounds_param", "max_blocks_per_seq: tl.constexpr" in text),
    ("negative_block_sentinel", "other=-1" in text),
    ("valid_block_mask", "& (block_idx >= 0)" in text and "& (block_idx < num_kv_blocks)" in text),
    ("bounded_generic_path", "using bounded 2D Triton path" in text),
]
fallback_text = (ROOT / "vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py").read_text()
checks.append(("topk_chunk_env", "VLLM_DS4_SM12X_PAGED_MQA_TOPK_CHUNK_SIZE" in fallback_text))

for script in (PP8, TP2):
    script_text = script.read_text()
    checks.append((f"{script.name}_rowwise_max_rows",
                   "VLLM_DS4_SM12X_MQA_ROWWISE_MAX_ROWS" in script_text))

failed = False
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name}")
    failed = failed or not ok

if failed:
    raise SystemExit(1)
