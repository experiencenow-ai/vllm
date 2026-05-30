# DS4 / GB10 DSV4 native routing notes

The DSV4 GB10 path is fail-closed. Marlin, CPU, Triton fallback, and emulation
are not production paths.

## Current SM12x native routing rule

Do not force `--moe-backend deep_gemm` on GB10/SM121 unless the runtime
DeepGEMM wheel has proven SM12x `fp8_fp4_gemm_nt` support.

The known failure shape is:

```text
Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend
...
csrc/apis/gemm.hpp:99: Unsupported architecture or scaling factor types
```

That is not Marlin and not fabric. It means DeepGEMM reached its FP8xFP4 GEMM
path and the runtime wheel/source still does not provide the required SM12x
kernel/scale-factor dispatch.

The default DS4 launch scripts therefore leave `--moe-backend` and
`--linear-backend` at `auto`. In Blackwell-family-120 strict mode, auto tries
native FlashInfer TRTLLM/CUTLASS MXFP4/MXFP8 backends and excludes
`DEEPGEMM_MXFP4` unless explicitly enabled.

## Explicit opt-in for DeepGEMM MXFP4

Only use this after a known-good wheel/container proves the SM12x FP8xFP4 GEMM
path:

```bash
export VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X=1
export DSV4_MOE_BACKEND=deep_gemm
```

Without that opt-in, an explicit DeepGEMM MXFP4 request on GB10 fails before
serving.

## r0b0tlab-style TP2 launcher

Use:

```bash
tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh
```

Defaults:

```text
TP=2
expert parallel enabled
FP8 KV
block size 256
MTP enabled
CUDA graphs enabled
moe_backend auto
linear_backend auto
Marlin disabled/fail-closed
DeepGEMM MXFP4 SM12x opt-in disabled
```

This intentionally matches the public recipe shape more closely than the older
DS4 scripts that forced `--moe-backend deep_gemm`.

## Cutlass FP8 / E8M0 scale boundary

The current native path can select:

```text
CutlassFp8BlockScaledMMKernel
FLASHINFER_TRTLLM_MXFP4_MXFP8
```

DeepSeek-V4 checkpoints may carry persistent block-scale tensors as
`torch.float8_e8m0fnu`. Those scale tensors must be decoded once after weight
load and before the stable ABI Cutlass op path. If an E8M0 tensor reaches
`_C.cutlass_scaled_mm`, startup can fail with:

```text
Not yet supported ScalarType 44
```

The DS4 recipe keeps the native Cutlass path and normalizes only the persistent
scale tensor to `float32` during `process_weights_after_loading()`. This is not
a Marlin fallback and not a per-token workaround.
