#!/usr/bin/env python3
"""Static audit for PP async sampled-token handoff hardening."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "vllm/v1/worker/gpu_model_runner.py").read_text()

CHECKS = [
    (
        "sender broadcasts request-count/payload metadata before token payload",
        "meta = torch.tensor(\n            [num_reqs, 1 if has_payload else 0]" in RUNNER
        and "torch.distributed.broadcast(meta, src=pp.rank, group=pp.device_group)"
        in RUNNER,
    ),
    (
        "sender skips chunked-prefill payload after metadata",
        "has_payload = not self._is_all_reqs_chunked_prefill()" in RUNNER
        and "if not has_payload:\n            return" in RUNNER,
    ),
    (
        "sender broadcasts private int32 contiguous token buffer",
        "sampled_token_ids.to(dtype=torch.int32).contiguous().clone()" in RUNNER,
    ),
    (
        "receiver obtains metadata from last PP rank",
        "torch.distributed.broadcast(meta, src=pp.last_rank, group=pp.device_group)"
        in RUNNER,
    ),
    (
        "receiver hardfails request-count mismatch",
        "PP+async sampled-token broadcast shape mismatch" in RUNNER,
    ),
    (
        "receiver clears previous token mapping when no payload exists",
        "self.input_batch.prev_sampled_token_ids = None" in RUNNER
        and "self.input_batch.prev_req_id_to_index = {}" in RUNNER,
    ),
]


def main() -> int:
    failed = False
    for name, ok in CHECKS:
        if ok:
            print(f"PASS: {name}")
            continue
        print(f"FAIL: {name}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
