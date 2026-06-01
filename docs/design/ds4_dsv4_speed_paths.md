# DS4 DSV4 Speed Paths

DSV4 PP8 is the simplest cache and pipeline topology, but it has no useful
expert-parallel group because each pipeline stage is TP1.  For high-concurrency
throughput testing, the next all-8 topology is PP4 x TP2 x EP2:

```text
stage 0: spark0 + spark1
stage 1: spark2 + spark3
stage 2: spark4 + spark5
stage 3: spark6 + spark7
```

The launcher is:

```bash
tools/ds4_launch_dsv4_flash_pp4_tp2_ep.sh
```

It keeps the production fail-closed policy:

```text
no Marlin
no forced DeepGEMM MXFP4 on SM12x
native Blackwell preflight
200G fabric guard
FlashInfer autotune disabled by default
bounded prefill waves
```

The physical rail TCP guard is temporarily relaxed while the weak back-half
ring links are investigated:

```text
rail TCP fail floor:     10 Gbit/s
rail TCP warning floor:  64 Gbit/s
NCCL collective fail:    disabled during cable debug
NCCL collective warning: 8 GB/s     (64 Gbit/s)
PP4 NCCL P2P fail floor: disabled during cable debug
PP4 NCCL P2P warning:    8 GB/s     (64 Gbit/s)
```

Each restart still prints the measured rail result and emits a warning when a
link falls below the warning floor.  Do not remove the warning or silently lower
it; the relaxed fail floor is only to keep model work moving while the cabling
or crosstalk issue is fixed.

DSV4 launchers skip the profile-only dummy sampler/logits warmup:

```text
VLLM_DS4_PROFILE_SKIP_DUMMY_SAMPLER=1
```

This is not a serving-path shortcut.  The model body still runs during
`profile_run()` so torch compile, PP/TP setup, and native kernels are exercised.
The skipped part is the startup dummy sampler's full-vocab TP logits all-gather,
which is not needed when `kv_cache_memory_bytes` is explicit and can kill weak
TP links before the service becomes healthy.  Real requests still execute the
normal logits and sampling path.

The PP PyNCCL tensor-dict path remains diagnostic for PP8.  A live test showed
the current generic tensor-dict PyNCCL path regressed, so PP8 keeps the torch PP
process-group path by default until the fast path is narrowed to known DSV4
boundary tensors.
