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

The PP PyNCCL tensor-dict path remains diagnostic for PP8.  A live test showed
the current generic tensor-dict PyNCCL path regressed, so PP8 keeps the torch PP
process-group path by default until the fast path is narrowed to known DSV4
boundary tensors.
