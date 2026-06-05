#!/usr/bin/env python3
"""Static checks for DeepSeek V4 attention leading-dimension handling."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "vllm/models/deepseek_v4/nvidia/ops/attention.py"


def require(*needles: str) -> None:
    text = ATTENTION.read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(
            f"FAIL: {ATTENTION.relative_to(ROOT)} missing shape markers: "
            f"{missing}")
    print(f"PASS: {ATTENTION.relative_to(ROOT)}")


require(
    "def _num_leading_rows(shape: torch.Size) -> int:",
    "def _positions_for_flat_hidden_rows(",
    "def _reshape_attention_projection(",
    "def _forward_context_batch_rows(",
    "def _attention_target_rows(",
    "def _pad_flat_attention_input(",
    "if len(leading_shape) == 1:",
    "return output",
    "hidden_states = hidden_states.reshape(num_tokens, hidden_size)",
    "target_rows = _attention_target_rows(positions, num_tokens)",
    "hidden_states = _pad_flat_attention_input(hidden_states, target_rows)",
    "leading_shape = torch.Size((target_rows,))",
    "positions = _positions_for_flat_hidden_rows(",
    "o_padded = torch.ops.vllm.deepseek_v4_attention(",
    "actual_rows = int(o_padded.shape[0])",
    "DS4 DSV4 attention q shape mismatch",
    "out = q.new_empty(q.shape)",
    "return out",
    "DS4 DSV4 attention wrapper/op shape metadata mismatch",
    "return hidden_states.new_empty(",
    "mutates_args=[]",
    "return _reshape_attention_projection(",
)

text = ATTENTION.read_text()

if "out.resize_(q.shape)" in text:
    raise SystemExit(
        "FAIL: vllm/models/deepseek_v4/nvidia/ops/attention.py still "
        "silently resizes the DSV4 attention output buffer")

if "mutates_args=[\"out\"]" in text:
    raise SystemExit(
        "FAIL: vllm/models/deepseek_v4/nvidia/ops/attention.py still "
        "registers DSV4 attention as an out-mutating custom op")

print("PASS: DS4 DSV4 attention shape audit")
