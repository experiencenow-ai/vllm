# DS4 GB10 native MXFP4/FP8 path

DS4 production on DGX Spark / GB10 must not silently run DeepSeek-V4 Flash or Qwen NVFP4 through Marlin or emulation. GB10 reports CUDA capability SM121. That is Blackwell-family hardware, but it is not in the SM100 datacenter family. Backend gates must therefore use `current_platform.is_device_capability_blackwell()` when the intended meaning is "Blackwell-family", and reserve `is_device_capability_family(100)` for kernels that are actually compiled only for SM100/SM10x.

## Required startup result

Acceptable DeepSeek-V4 MXFP4 MoE backend logs on GB10 are native backends such as:

```text
Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend.
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend.
```

`DEEPGEMM_MXFP4` is not a default acceptable path on GB10 / SM12x. The current
known DeepGEMM stack can pass Marlin, SF-layout, mHC, model-load, and RoCE
checks, then fail during profile/first-forward in the FP8xFP4 GEMM dispatch:

```text
csrc/apis/gemm.hpp:99: Unsupported architecture or scaling factor types
```

For that reason the DSV4 launchers default both `DSV4_LINEAR_BACKEND` and
`DSV4_MOE_BACKEND` to `auto`, and the Blackwell-family-120 MXFP4 auto list
keeps `DEEPGEMM_MXFP4` out unless it is explicitly enabled:

```bash
VLLM_DS4_ALLOW_DEEPGEMM_MXFP4_SM12X=1
```

Only use that opt-in after the runtime DeepGEMM wheel has proven SM12x
`fp8_fp4_gemm_nt` support. Do not force `--moe-backend deep_gemm` or
`--linear-backend deep_gemm` for production GB10 tests until then.

The same `gemm.hpp:99` wall can appear through FP8 block-scaled linear layers,
even when MoE correctly selects FlashInfer. On GB10 / SM12x, auto linear
routing therefore also removes DeepGEMM-bearing FP8 linear candidates by
default:

```text
DeepGemmFp8BlockScaledMMKernel
FlashInferFp8DeepGEMMDynamicBlockScaledKernel
```

Those candidates are restored only with:

```bash
VLLM_DS4_ALLOW_DEEPGEMM_FP8_LINEAR_SM12X=1
```

The non-DeepGEMM native candidates, such as FlashInfer or CUTLASS FP8
block-scaled linear kernels, must be selected or startup must fail.

Live TP2 validation on Spark0/Spark1 reached the next boundary after these
guards were added. The acceptable routing evidence was:

```text
Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod
Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend
```

That run loaded the DeepSeek-V4 Flash model and MTP weights, then failed during
profile compilation in Torch Inductor with:

```text
AssertionError: auto_functionalized was not removed
```

Setting `enable_auto_functionalized_v2=true` in `DSV4_COMPILATION_CONFIG`
changes the failure to `auto_functionalized_v2 was not removed`; it does not
fix the compile path. Treat that as a Torch/vLLM compilation-stack issue after
native routing, not as permission to re-enable Marlin or force DeepGEMM MXFP4
on SM12x.

Unacceptable production logs:

```text
Using 'MARLIN' Mxfp4 MoE backend.
Using 'BATCHED_MARLIN' Mxfp4 MoE backend.
Using 'EMULATION' Mxfp4 MoE backend.
```

For DeepSeek-V4 on Blackwell, the auto backend priority list no longer contains Marlin. This is structural: it does not depend on `VLLM_DS4_STRICT_NATIVE_FP4` being present in the runtime. If all native backends reject the deployment, startup fails and reports the last native rejection reason.

For Qwen NVFP4 on Blackwell, acceptable linear backends are native Blackwell FP4 paths such as FlashInfer/CUTLASS or CUTLASS. `MarlinNvFp4LinearKernel`, `FbgemmNvFp4LinearKernel`, and `EmulationNvFp4LinearKernel` are rejected on Blackwell even when strict-env plumbing is missing. ModelOpt `W4A16_NVFP4` is also rejected on Blackwell because this tree currently implements that checkpoint shape through Marlin; use W4A4 ModelOpt NVFP4 with native FlashInfer/CUTLASS, or BF16.

Run the static no-Marlin audit before building an image:

```bash
python3 tools/ds4_no_marlin_static_audit.py
```

## Probe before serving

Run this inside the same container and Python environment that will serve the model:

```bash
python3 tools/ds4_dsv4_native_preflight.py
python3 tools/ds4_native_blackwell_probe.py --strict-dsv4
```

The minimum useful result is:

```text
sm12x: yes
deep_gemm_symbol_tf32_hc_prenorm_gemm: yes
deep_gemm_marker_sm120_tf32_hc_prenorm_gemm: <path>
deep_gemm_marker_sm120_fp8_paged_mqa_logits: <path>
active_sf_layout_moe_recipe_1x32: yes:<shape>:<dtype>
active_sf_layout_fp8_recipe_1x128x128: yes:<shape>:<dtype>
is_blackwell: yes
family_120: yes
native_mxfp4_candidate_ready: yes
native_mhc_tilelang_ready: yes
```

If `deep_gemm_marker_sm120_tf32_hc_prenorm_gemm` or `deep_gemm_marker_sm120_fp8_paged_mqa_logits` is `no`, install the GB10-capable DeepGEMM fork before serving:

```bash
export CUDA_VERSION=13.0
export TORCH_CUDA_ARCH_LIST=12.1a
export DEEPGEMM_GIT_REPO=https://github.com/jasl/DeepGEMM.git
export DEEPGEMM_GIT_REF=7a7a41a1
tools/ds4_install_deepgemm_gb10_native.sh
```

`tools/ds4_install_deepgemm_gb10_native.sh` exports
`VLLM_DS4_PATCH_DEEPGEMM_SM12X=1`, so `tools/install_deepgemm.sh` patches the
DeepGEMM source before compiling the wheel. The same source patch is applied
when vLLM vendors/builds DeepGEMM through CMake. The patch extends the SM100
scale-factor layout dispatch to SM12x; without it, SM121 can pass backend
selection, load `DEEPGEMM_MXFP4`, and still crash during FP8 scale-factor
post-processing with `Unknown SF transformation`.

You can verify a DeepGEMM source tree directly:

```bash
python3 tools/ds4_patch_deepgemm_sm12x_layout.py /path/to/DeepGEMM --check-only
```

If `native_mxfp4_candidate_ready` is `no`, the problem is not the model weights. It is the serving environment: missing or wrong `deep_gemm`, missing FlashInfer native fused-MoE support, wrong CUDA/Torch build, or SM121 not being recognized as Blackwell.

If `native_mhc_tilelang_ready` is `no`, the run must not continue as a hidden fallback. The DeepSeek-V4 MHC path is performance-critical; serving should either use a working native TileLang/DeepGEMM path or stop with the probe's concrete error.

## TP2 reproduction profile

Use the TP2 script for an apples-to-apples native-path check before debugging PP8 scheduling:

```bash
# worker
NODE_RANK=1 HEAD_ADDR=<spark3-200g-ip> DS4_200G_IFNAME=<worker-200g-if> \
  DSV4_FLASH_MODEL=/path/to/DeepSeek-V4-Flash \
  tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh

# head
NODE_RANK=0 HEAD_ADDR=<spark3-200g-ip> DS4_200G_IFNAME=<head-200g-if> \
  DSV4_FLASH_MODEL=/path/to/DeepSeek-V4-Flash \
  tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh
```

The launcher hard-fails if `DS4_200G_IFNAME` is missing, the selected
interface is not link-up at `200000Mb/s`, routing to `HEAD_ADDR` would use a
non-200G device, or the selected interface does not map to a RoCE HCA. It pins
`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `VLLM_HOST_IP`,
`NCCL_NET=IB`, and `NCCL_IB_HCA` to the selected fabric interface, and rejects
`NCCL_IB_DISABLE=1`. Before loading weights it runs a CUDA/NCCL all-reduce
preflight on the same interface and HCA, then runs the DSV4 DeepGEMM native
package preflight and strict native Blackwell probe above. This is intentional:
a slow 10G/WiFi/socket fallback, broken
200G/RDMA path, Marlin request, or missing native MHC path is a failed run, not
a degraded benchmark.

Default latency profile:

```text
max_model_len=65000
max_num_seqs=2
max_num_batched_tokens=4096
```

Throughput profile override:

```bash
DSV4_MAX_MODEL_LEN=65536 DSV4_MAX_NUM_SEQS=16 DSV4_MAX_NUM_BATCHED_TOKENS=8192 \
  tools/ds4_launch_dsv4_flash_tp2_native_benchmark.sh
```

## PP8 production profile

The PP8 script also sets strict native FP4 mode and CUDA graph compilation. It does not use `--enforce-eager`.

```bash
NODE_RANK=<0-7> NNODES=8 HEAD_ADDR=<spark0-200g-ip> DS4_200G_IFNAME=<rank-local-200g-if> \
  DSV4_FLASH_MODEL=/path/to/DeepSeek-V4-Flash \
  tools/ds4_launch_dsv4_flash_pp8.sh
```

If the head address is a routed `10.10.100.x` loopback fabric address on rank 0,
set `DS4_200G_ALLOW_LOOPBACK_HEAD=1` on rank 0 only. Workers still must route to
that address through their selected 200G interface.
