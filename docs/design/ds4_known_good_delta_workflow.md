# DS4 GB10 known-good delta workflow

The DS4 native GB10 path should be ported by mechanical comparison against a
known-good vLLM tree, not by adding one patch per observed crash.

Known-good references currently worth comparing:

- `jasl/vllm@dda4668b59567416f86956cfe7bbc1eab371a61e` — forum/r0b0tlab
  pinned TP=2 DSV4 reference.
- `jasl/vllm@27fd665b` or the current `ds4-sm120-preview-dev` branch — later
  SM12x DSV4 work that includes the `fp8_einsum` SM12x Triton dispatch pattern.
- `jasl/DeepGEMM@7a7a41a1` — DeepGEMM fork discussed in the SM12x DSV4 tracking
  issue. Use only for the kernels it actually covers; do not force DeepGEMM
  MXFP4 on SM12x unless the FP8xFP4 GEMM path is proven.

Run from the vLLM checkout:

```bash
python3 tools/ds4_known_good_delta_audit.py \
    --repo https://github.com/jasl/vllm.git \
    --ref dda4668b59567416f86956cfe7bbc1eab371a61e \
    --out /tmp/ds4_delta_dda

python3 tools/ds4_known_good_delta_audit.py \
    --repo https://github.com/jasl/vllm.git \
    --ref 27fd665b \
    --out /tmp/ds4_delta_27fd
```

The report focuses on the files that decide whether DSV4/Qwen take the native
GB10 path:

- CUDA capability / Blackwell-family detection.
- DeepSeek V4 sparse MLA, Lightning Indexer, MQA logits, mHC, and FP8 einsum.
- MXFP4 MoE backend selection.
- FP8 / NVFP4 dense linear kernel selection.
- E8M0 scale handling before stable ABI boundaries.
- Torch functionalization pass coverage for DSV4 custom ops.
- CMake / build pins that decide whether SM121 kernels are actually built.

Porting rule:

1. Keep local DS4 external-KV, LMCache/HMA, API, queue, telemetry, and pipeline
   modifications.
2. Port native GB10 fast-path code from the known-good tree into those local
   integration points.
3. No Marlin/emulation fallback on GB10 production paths.
4. Do not force DeepGEMM MXFP4 on SM12x unless a known-good DeepGEMM wheel proves
   FP8xFP4 GEMM support.
5. Rerun this audit after every port. The final report should show only
   intentional DS4-local differences.

High-value deltas to inspect first:

- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py`
- `vllm/models/deepseek_v4/nvidia/ops/cutedsl_utils.py`
- `vllm/v1/attention/ops/deepseek_v4_ops/sm12x_deep_gemm_fallbacks.py`
- `vllm/utils/deep_gemm.py`
- `vllm/models/deepseek_v4/nvidia/ops/attention.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`
- `vllm/model_executor/kernels/linear/__init__.py`

The audit is read-only. It exists to prevent symptom-by-symptom patching.
