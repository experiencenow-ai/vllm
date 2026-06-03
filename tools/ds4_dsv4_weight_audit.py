#!/usr/bin/env python3
"""Static checks for DS4 DSV4 PP weight-load auditing."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
envs = (root / "vllm/envs.py").read_text()
model = (root / "vllm/models/deepseek_v4/nvidia/model.py").read_text()
launcher = (root / "tools/ds4_launch_dsv4_flash_pp8.sh").read_text()
relaunch = (root / "tools/ds4_relaunch_spark_service.py").read_text()

checks = [
    (
        "DSV4 weight audit env is registered",
        "VLLM_DS4_DSV4_WEIGHT_AUDIT: bool = False" in envs
        and "\"VLLM_DS4_DSV4_WEIGHT_AUDIT\": lambda:" in envs,
    ),
    (
        "expert loads are only counted after success",
        "loaded_expert_param = None" in model
        and "if loaded_expert_param is not None:" in model
        and "loaded_params.add(loaded_expert_param)" in model,
    ),
    (
        "owned edge weights are required",
        "model.embed_tokens.weight" in model
        and "lm_head.weight" in model
        and "model.norm.weight" in model
        and "model.hc_head_fn" in model
        and "model.hc_head_base" in model
        and "model.hc_head_scale" in model,
    ),
    (
        "final lm_head uses DeepSeek reference fp32 logits path",
        "params_dtype=torch.float32" in model
        and "hidden_states.to(torch.float32)" in model
        and "final PP rank must keep " in model
        and "lm_head.weight in torch.float32" in model,
    ),
    (
        "owned local layer coverage is checked",
        "expected_layers = set(range(self.model.start_layer, self.model.end_layer))"
        in model
        and "missing_layers = sorted(expected_layers - loaded_layers)" in model,
    ),
    (
        "all rank-owned parameters must be reported loaded",
        "owned_params = set(dict(self.named_parameters()))" in model
        and "missing_owned_params = sorted(owned_params - loaded_params)" in model
        and "unexpected_loaded_params = sorted(loaded_params - owned_params)" in model
        and "missing_owned_params[:64]" in model,
    ),
    (
        "expert coverage is tracked by layer, tensor, shard, and expert id",
        "self._ds4_expert_coverage: dict[int, dict[tuple[str, str], set[int]]] = {}"
        in model
        and "self._record_ds4_expert_coverage(" in model
        and r"(?:model\.)?layers\." in model
        and "self._ds4_expert_coverage.setdefault(layer_idx, {})" in model
        and "layer_coverage.setdefault((tensor_kind, shard_id), set()).add(expert_id)"
        in model,
    ),
    (
        "expert coverage is audited once after AutoWeightsLoader completes",
        "loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)"
        in model
        and "self.model._audit_ds4_expert_coverage()" in model
        and "self._audit_ds4_expert_coverage()" not in model.split(
            "def _record_ds4_expert_coverage", 1
        )[0],
    ),
    (
        "FP4 expert weights and scales are required per owned layer",
        '("weight", "w1")' in model
        and '("weight", "w2")' in model
        and '("weight", "w3")' in model
        and '("weight_scale", "w1")' in model
        and '("weight_scale", "w2")' in model
        and '("weight_scale", "w3")' in model
        and "expected_experts = set(range(self.config.n_routed_experts))" in model,
    ),
    (
        "expert coverage audit fails closed",
        "DS4 DSV4 expert coverage audit failed" in model
        and "raise RuntimeError" in model,
    ),
    (
        "weight audit fails closed",
        "DS4 DSV4 weight audit failed" in model
        and "raise RuntimeError" in model,
    ),
    (
        "launcher enables and logs weight audit",
        'export VLLM_DS4_DSV4_WEIGHT_AUDIT="${VLLM_DS4_DSV4_WEIGHT_AUDIT:-1}"'
        in launcher
        and "weight_audit=$VLLM_DS4_DSV4_WEIGHT_AUDIT" in launcher,
    ),
    (
        "relaunch build validates weight audit",
        "tools/ds4_dsv4_weight_audit.py" in relaunch,
    ),
]

failed = 0
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed += 0 if ok else 1

if failed:
    raise SystemExit(1)
