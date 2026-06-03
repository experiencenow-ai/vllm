#!/usr/bin/env python3
"""Static audit for DS4 DeepSeek V4 hash-MoE router dtype safety."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
router = (
    root / "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
).read_text()
envs = (root / "vllm/envs.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "hash router casts token ids to hash table dtype before CUDA op",
        "if hash_indices_table is not None and input_tokens is not None:" in router
        and "input_tokens.dtype != hash_indices_table.dtype" in router
        and "input_tokens = input_tokens.to(dtype=hash_indices_table.dtype)" in router
        and "read the int64 token buffer through an int32 pointer" in router,
    ),
    (
        "CUDA hash-router slices graph-padded rows before CUDA op",
        "_ds4_slice_router_rows_for_output(" in router
        and router.find("_ds4_slice_router_rows_for_output(")
        < router.find("ops.topk_hash_softplus_sqrt(")
        and "gating_output[:num_rows]" in router
        and "input_tokens[:num_rows]" in router,
    ),
    (
        "CUDA hash-router op is reached after dtype normalization",
        router.find("input_tokens = input_tokens.to(dtype=hash_indices_table.dtype)")
        < router.find("ops.topk_hash_softplus_sqrt("),
    ),
    (
        "hash router reference-check envs are registered",
        "VLLM_DS4_DSV4_HASH_ROUTER_REF_CHECK" in envs
        and "VLLM_DS4_DSV4_HASH_ROUTER_REF_MAX_TOKENS" in envs
        and "VLLM_DS4_DSV4_HASH_ROUTER_REF_ATOL" in envs,
    ),
    (
        "hash router has a compile-disabled torch reference check",
        "@torch.compiler.disable" in router
        and "_ds4_check_hash_softplus_sqrt_against_torch(" in router
        and "_topk_softplus_sqrt_torch(" in router
        and "DS4 DSV4 hash-router CUDA op disagrees with torch reference" in router,
    ),
    (
        "hash router reference-check skips CUDA graph capture",
        "torch.cuda.is_current_stream_capturing()" in router
        and "if torch.cuda.is_current_stream_capturing():" in router
        and router.find("torch.cuda.is_current_stream_capturing()")
        < router.find("ref_weights = torch.empty_like(topk_weights)"),
    ),
    (
        "hash router reference-check slices graph-padded rows",
        "_ds4_slice_router_rows_for_output(" in router
        and "gating_output[:num_rows]" in router
        and "input_tokens[:num_rows]" in router
        and router.find("_ds4_slice_router_rows_for_output(")
        < router.find("ref_weights = torch.empty_like(topk_weights)"),
    ),
    (
        "DSV4 launcher logs hash router reference-check knobs",
        "hash_router_ref_check=$VLLM_DS4_DSV4_HASH_ROUTER_REF_CHECK" in launcher
        and "hash_router_ref_max_tokens=$VLLM_DS4_DSV4_HASH_ROUTER_REF_MAX_TOKENS"
        in launcher
        and "hash_router_ref_atol=$VLLM_DS4_DSV4_HASH_ROUTER_REF_ATOL" in launcher,
    ),
    (
        "relaunch build validates hash-MoE router audit",
        "tools/ds4_dsv4_hash_moe_router_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
