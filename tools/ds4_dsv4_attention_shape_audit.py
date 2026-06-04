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
    "if len(leading_shape) == 1:",
    "return output",
    "hidden_states = hidden_states.reshape(num_tokens, hidden_size)",
    "positions = _positions_for_flat_hidden_rows(",
    "out.resize_(q.shape)",
    "return _reshape_attention_projection(",
)

print("PASS: DS4 DSV4 attention shape audit")
