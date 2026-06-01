# DS4 CUDA fast-path notes

This patch targets the current PP throughput bottleneck after the coordinator
cohort fixes: DSV4 sparse-MQA/top-k/indexer work still used generic PyTorch
selection and repeated logits allocation inside the SM12x native path.

## Changes

- `sm12x_mqa.py` now accepts caller-provided logits buffers for dense and
  paged MQA logits. The paged top-k path can reuse one chunk logits buffer
  instead of allocating a new `[rows, chunk]` tensor every chunk.
- `sm12x_deep_gemm_fallbacks.py` adds a CUDA selection path using vLLM's
  `_C.top_k_per_row_prefill` and small Triton gather kernels so chunk top-k
  and candidate-merge top-k no longer have to call generic `torch.topk` in the
  hot path.
- `VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT=1` is the production default. In strict
  native mode, requesting this path without the `_C` CUDA extension fails
  instead of silently falling back.
- Non-streaming completion benchmarks can set
  `VLLM_DS4_FINAL_ONLY_NONSTREAMING=1` so the server emits only final request
  outputs instead of draining every intermediate token through the OpenAI layer.
- `VLLM_DS4_ITERATION_TIMING=1` adds engine timing logs that separate scheduler,
  submission, future wait, and scheduler update time.

## Runtime knobs

```bash
export VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT=1
export VLLM_DS4_FINAL_ONLY_NONSTREAMING=1
export VLLM_DS4_ITERATION_TIMING=1
export VLLM_DS4_ITERATION_TIMING_EVERY=10
```

For diagnosis only:

```bash
export VLLM_DS4_SM12X_MQA_TOPK_CUDA_SELECT=0
```

That returns to the generic PyTorch top-k path and should not be used for
production throughput measurements.
