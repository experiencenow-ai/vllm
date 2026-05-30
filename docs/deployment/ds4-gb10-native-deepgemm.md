# DS4 GB10 native DeepGEMM requirement

DeepSeek-V4 on DGX Spark / GB10 is not unlocked by the vLLM selector alone.
The runtime also needs a DeepGEMM build with SM12x kernel coverage for the
non-MoE DeepSeek-V4 paths:

- `sm120_tf32_hc_prenorm_gemm` for mHC pre-norm GEMM;
- `sm120_fp8_paged_mqa_logits` for the Lightning Indexer / paged-MQA path;
- SM12x dispatch for the FP8/MXFP4 MoE path.

If the process fails with:

```text
DeepGEMM tf32_hc_prenorm_gemm rejected this GPU architecture
```

then the no-Marlin fail-closed guard is doing the right thing. The installed
DeepGEMM is not the GB10-capable stack. Do not bypass this and do not fall back
to Marlin.

## Install the DS4 GB10 DeepGEMM fork

```bash
cd /home/$USER/src/vllm
export CUDA_VERSION=13.0
export TORCH_CUDA_ARCH_LIST=12.1a
export DEEPGEMM_GIT_REPO=https://github.com/jasl/DeepGEMM.git
export DEEPGEMM_GIT_REF=7a7a41a1

tools/ds4_install_deepgemm_gb10_native.sh
```

For a full wheel/container rebuild, export the same variables before building
vLLM. `cmake/external_projects/deepgemm.cmake` reads `DEEPGEMM_GIT_REPO` and
`DEEPGEMM_GIT_REF`, and includes `12.0a` / `12.1a` in the DeepGEMM architecture
set when CUDA is 13.0 or newer.

## Preflight

Run this inside the exact runtime/container that will serve DSV4:

```bash
python3 tools/ds4_dsv4_native_preflight.py
```

Use an active mHC kernel call only when debugging the installed DeepGEMM binary:

```bash
python3 tools/ds4_dsv4_native_preflight.py --active-kernel-probe
```

The launch scripts run the non-active preflight by default. To bypass it during
development only:

```bash
export DS4_SKIP_NATIVE_PREFLIGHT=1
```

Production should leave the preflight enabled.

## Expected forbidden log lines

```text
Using 'MARLIN' Mxfp4 MoE backend
EmulationNvFp4LinearKernel
MarlinNvFp4LinearKernel
```

Any of those means the process is not on the native GB10 path.
