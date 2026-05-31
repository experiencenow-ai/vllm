#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static audit for DSV4 prefill K-cache safety rails."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_UTILS = ROOT / "vllm/models/deepseek_v4/common/ops/cache_utils.py"
ENVS = ROOT / "vllm/envs.py"
PP8 = ROOT / "tools/ds4_launch_dsv4_flash_pp8.sh"
TP2 = ROOT / "tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh"

cache_utils = CACHE_UTILS.read_text()
envs = ENVS.read_text()
checks = [
    (
        "dequant_gather_shape_validation",
        "def _validate_dequantize_and_gather_k_cache_inputs" in cache_utils
        and "block_table expected" not in cache_utils
        and "block_table to be" in cache_utils,
    ),
    (
        "cutedsl_failclosed_gate",
        "def _should_use_cutedsl_dequantize_and_gather_k_cache" in cache_utils
        and "VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL_MAX_ROWS" in cache_utils
        and "CuteDSL is not importable" in cache_utils
        and "refusing Triton fallback" in cache_utils,
    ),
    (
        "cutedsl_gate_precedes_import",
        "if _should_use_cutedsl_dequantize_and_gather_k_cache(out):"
        in cache_utils,
    ),
    (
        "bounded_triton_path_remains",
        "def dequantize_and_gather_k_cache_triton" in cache_utils
        and "dequantize_and_gather_k_cache_triton(" in cache_utils,
    ),
    (
        "triton_gather_masks_block_table_and_kv_blocks",
        "num_kv_blocks: tl.constexpr" in cache_utils
        and "valid_table_entry = (block_in_seq >= 0)" in cache_utils
        and "& (physical_block_idx < num_kv_blocks)" in cache_utils
        and "physical_block_idx = tl.where(valid_block, physical_block_idx, 0)"
        in cache_utils,
    ),
    (
        "triton_gather_masks_output_bounds",
        "out_max_tokens: tl.constexpr" in cache_utils
        and "valid_output = (out_token_idx >= 0) & (out_token_idx < out_max_tokens)"
        in cache_utils
        and "valid_token = valid_block & valid_output" in cache_utils,
    ),
    (
        "envs_register_dequant_gate",
        "VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL" in envs
        and "VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL_MAX_ROWS" in envs
        and 'os.environ.get("VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL", "1")' in envs
        and 'os.environ.get("VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL_MAX_ROWS", "-1")'
        in envs,
    ),
    (
        "envs_register_sm12x_mqa_vars",
        "VLLM_DS4_SM12X_MQA_ROWWISE" in envs
        and "VLLM_DS4_SM12X_PAGED_MQA_TOPK_CHUNK_SIZE" in envs,
    ),
]

for script in (PP8, TP2):
    script_text = script.read_text()
    checks.append((
        f"{script.name}_exports_dequant_gate",
        'VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL:-1' in script_text
        and 'VLLM_DS4_DEQUANT_GATHER_K_CUTEDSL_MAX_ROWS:--1' in script_text,
    ))

failed = False
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name}")
    failed = failed or not ok

if failed:
    raise SystemExit(1)
