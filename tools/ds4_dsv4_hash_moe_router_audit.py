#!/usr/bin/env python3
"""Static audit for DS4 DeepSeek V4 hash-MoE router dtype safety."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
router = (
    root / "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
).read_text()
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
        "CUDA hash-router op is reached after dtype normalization",
        router.find("input_tokens = input_tokens.to(dtype=hash_indices_table.dtype)")
        < router.find("ops.topk_hash_softplus_sqrt("),
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
