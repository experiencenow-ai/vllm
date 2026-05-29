# DS4 Local KV Recipes

This page records the local DS4 KV-cache recipes used by the Spark runtime
fork. Keep benchmark claims tied to the source commit, hardware, launch flags,
and validation logs.

## DeepSeek V4 Flash

The current DS4 production lane uses the host-local vLLM fork rather than the
older Docker copy-patch runtime:

```text
repository: https://github.com/experiencenow-ai/vllm
branch: main
required code: PR #7 DeepSeek V4 Cutlass cache-mode fix, or newer
runtime: /home/spark*/ds4-vllm-local
source:  /home/spark*/src/vllm
model:   /home/spark*/models/hf/deepseek-ai/DeepSeek-V4-Flash
```

Use one grouped vLLM service across the Spark tensor-parallel lane. The live
requalification shape used spark4 as head and spark7 as worker while spark5 was
unavailable; the normal lane is spark4 plus spark5.

Do not use `/home/spark*/src/vllm-b55c3b6-docker-lineage` for the standard
DSV4 lane. That tree was a Docker-lineage rescue runtime; the documented path is
the source-built host-local checkout at `/home/spark*/src/vllm`.

### Source Runtime Build

Build vLLM from the host-local source checkout into the host-local runtime on
every DSV4 node. Use `git checkout`; some Spark images have an older Git that
does not support `git switch`.

```bash
cd /home/$USER/src/vllm
git fetch experiencenow
git checkout main
git pull --ff-only experiencenow main

/home/$USER/ds4-vllm-local/bin/python -m pip install setuptools-rust pytest

export VLLM_TARGET_DEVICE=cuda
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8
export TORCH_CUDA_ARCH_LIST="8.0+PTX;12.1a"
export CPATH=/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include:/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include/python3.12:${CPATH:-}
export C_INCLUDE_PATH=/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include:/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include/python3.12:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include:/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include/python3.12:${CPLUS_INCLUDE_PATH:-}
export CMAKE_ARGS="-DPython_INCLUDE_DIR=/home/$USER/standard-runtimes/python3.12-dev-extract/usr/include/python3.12 -DPython_EXECUTABLE=/home/$USER/ds4-vllm-local/bin/python"

/home/$USER/ds4-vllm-local/bin/python -m pip install -e . --no-build-isolation
```

The `TORCH_CUDA_ARCH_LIST` line is required. The Spark CUDA 13 / PyTorch 2.11
stack warns that PyTorch ignores `CMAKE_CUDA_ARCHITECTURES`; use
`TORCH_CUDA_ARCH_LIST` instead. A correct configure prints:

```text
Added CUDA NVCC flags for: -gencode;arch=compute_80,code=sm_80;...;arch=compute_121a,code=sm_121a;...
CUDA target architectures: 8.0;12.1a
Building Marlin kernels for archs: 8.0+PTX
Building Marlin MOE kernels for archs: 8.0+PTX
```

Do not use `-DCMAKE_CUDA_ARCHITECTURES=native` on this stack. It can be
rewritten by PyTorch/CMake into `compute_20,code=sm_121`, which vLLM parses as
`CUDA target architectures: 2.0`; that skips the DSV4 kernels.

Without the `8.0+PTX` target, CMake can skip the Marlin source set that
registers the CUDA implementation for `_C::gptq_marlin_repack`. Python imports
still pass, but DSV4 fails during model load with:

```text
NotImplementedError: Could not run '_C::gptq_marlin_repack' with arguments from the 'CUDA' backend.
```

Do not force `DS4_DSV4_MOE_BACKEND=triton` for this lane. The MXFP4 Triton path
rejects the Spark CUDA device during DeepSeek V4 model initialization. Use the
default backend selection and verify that Marlin's CUDA op exists.

Keep `pytest` installed in the serving runtime. Qwen torch.compile profiling on
the Spark CUDA 13 / PyTorch 2.11 stack can import `cupy.testing`, which imports
`pytest`; without it, Qwen can fail after checkpoint load with
`ModuleNotFoundError: No module named 'pytest'`.

Prove the installed runtime is the source checkout, not an old editable install:

```bash
/home/$USER/ds4-vllm-local/bin/python -m pip show vllm | grep -E 'Version|Editable project location'
PYTHONPATH=/home/$USER/src/vllm /home/$USER/ds4-vllm-local/bin/python - <<'PY'
import pathlib
import vllm
import vllm.vllm_flash_attn
print("vllm_file", pathlib.Path(vllm.__file__).resolve())
print("version", getattr(vllm, "__version__", None))
print("fa_file", pathlib.Path(vllm.vllm_flash_attn.__file__).resolve())
PY
```

Then smoke the Marlin repack CUDA dispatch before spending time on a full DSV4
load:

```bash
PYTHONPATH=/home/$USER/src/vllm /home/$USER/ds4-vllm-local/bin/python - <<'PY'
import torch
import vllm._C
b = torch.zeros((32, 64), device="cuda", dtype=torch.int32)
perm = torch.empty((0,), device="cuda", dtype=torch.int32)
out = torch.ops._C.gptq_marlin_repack(b, perm, 256, 64, 4, False)
torch.cuda.synchronize()
print("gptq_marlin_repack_cuda_ok", tuple(out.shape), out.dtype, out.device)
PY
```

Critical launch settings:

```bash
export VLLM_USE_SIMPLE_KV_OFFLOAD=1
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_ROOT=/var/tmp/ds4_hma_store/dsv4/simple_cpu_offload
export VLLM_SIMPLE_KV_OFFLOAD_PERSIST_STRICT=1

vllm serve /home/spark4/models/hf/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --max-model-len 262144 \
  --block-size 256 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --kv-offloading-size 8 \
  --kv-offloading-backend native \
  --kv-cache-metrics \
  --enable-logging-iteration-details \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --no-disable-hybrid-kv-cache-manager \
  --enforce-eager
```

Do not disable the hybrid KV cache manager for DSV4. Keep
`VLLM_USE_SIMPLE_KV_OFFLOAD=1`; otherwise the runtime can select the generic
offloading connector and miss the persistent SimpleCPUOffload store hooks.

Keep `--block-size 256`. DSV4 native offload uses the model-specific group hash
size internally while the scheduler block size remains 256.

The 2026-05-28 replay benchmark used a 6733-token shared prefix and observed:

```text
cold elapsed: 31.621346s
warm elapsed: 3.455483s
speedup: 9.151064x
DS4 persistent SimpleCPUOffload scheduler hit: 6144 tokens
warm replay computed context: 589 tokens
external prefix cache hit rate: 45.6%
```

Validation checklist:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  --data-binary @/tmp/ds4_chat_probe.json
curl -fsS -X POST http://127.0.0.1:8000/v1/trim_memory \
  -H 'content-type: application/json' \
  --data '{"reset_prefix_cache":true,"malloc_trim":true}'
```

The model response should include `deepseek-v4-flash` and the configured
`max_model_len`. The route list should include `POST /v1/trim_memory`.

## Qwen27 LMCache HMA

Qwen3.5/Qwen3.6-style models are hybrid GDN models. They do not have one
uniform attention-only KV layout, so do not launch Qwen27 LMCache with
`--disable-hybrid-kv-cache-manager`.

The required stack is:

```text
vLLM: LMCacheConnectorV1 implements SupportsHMA and forwards the adapter-selected KV group
vLLM: Mamba/GDN scheduler and prefix-cache fixes for external computed tokens
LMCache: companion hybrid-state restore branch selecting FullAttentionSpec groups
```

Install the Qwen runtime dependencies into the exact Python environment that
will launch vLLM. `pytest` is a serving dependency for this stack: Qwen
torch.compile profiling can import `cupy.testing`, which imports `pytest`.
Without it, Qwen can load all checkpoint shards and then fail during engine-core
initialization with `ModuleNotFoundError: No module named 'pytest'`.

```bash
/home/$USER/ds4-vllm-local/bin/python -m pip install setuptools-rust pytest
```

Install the LMCache companion branch into the same runtime environment used by
the vLLM server:

```bash
git clone https://github.com/LMCache/LMCache.git /home/$USER/src/LMCache-qwen-hma
cd /home/$USER/src/LMCache-qwen-hma
git fetch origin pull/3284/head:qwen-hybrid-state-cache
git checkout qwen-hybrid-state-cache
/home/$USER/ds4-vllm-local/bin/python -m pip install -e .
```

When installing on Spark runtimes without system Python development headers in
the default include path, set the CUDA and Python include paths before running
`pip install`:

```bash
export CUDA_HOME=/usr/local/cuda
export PYHDR_HOME=/home/$USER/standard-runtimes/python3.12-dev-extract
test -f "$PYHDR_HOME/usr/include/python3.12/Python.h"
test -f "$PYHDR_HOME/usr/include/aarch64-linux-gnu/python3.12/pyconfig.h"
export CPATH=$PYHDR_HOME/usr/include/python3.12:$PYHDR_HOME/usr/include/aarch64-linux-gnu/python3.12:$PYHDR_HOME/usr/include:${CPATH:-}
```

After installing vLLM and LMCache, prove the serving runtime has the complete
Qwen dependency set before loading the model:

```bash
/home/$USER/ds4-vllm-local/bin/python -c 'import pytest, cupy.testing, vllm, lmcache; print("qwen-runtime-deps-ok")'
```

Launch Qwen27 with HMA enabled:

```bash
export LMCACHE_CONFIG_FILE=/tmp/lmcache_qwen27.yaml
export LMCACHE_ROOT=/home/$USER/ds4_lmcache/qwen27
export PYTHONHASHSEED=0
mkdir -p "$LMCACHE_ROOT"

cat > "$LMCACHE_CONFIG_FILE" <<YAML
chunk_size: 784
local_cpu: true
max_local_cpu_size: 16.0
local_disk: file://$LMCACHE_ROOT
max_local_disk_size: 1024.0
YAML

export PATH=/home/$USER/ds4-vllm-local/bin:$PATH
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
/home/$USER/ds4-vllm-local/bin/python -m vllm.entrypoints.cli.main serve \
  /home/$USER/models/hf/Qwen/Qwen3.6-27B-FP8 \
  --served-model-name qwen27 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.50 \
  --dtype bfloat16 \
  --language-model-only \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --no-async-scheduling \
  --reasoning-parser qwen3 \
  --no-disable-hybrid-kv-cache-manager \
  --disable-log-stats \
  --mamba-cache-mode align \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

For Qwen3.6-27B-FP8 on spark7, vLLM sets the HMA attention page size to 784
tokens. Keep the LMCache `chunk_size` aligned to that page size. The first
HMA smoke used `chunk_size: 784` and a smaller
`--max-num-batched-tokens 7840`; production Qwen27 lanes should use the normal
resident-lane caps above: `--max-num-seqs 12`,
`--max-num-batched-tokens 32768`, and `--gpu-memory-utilization 0.50`.
Do not fall back to uncapped vLLM defaults for a 262k qual launch; that can push
Spark nodes into swap before the API is ready. Keep
`max_local_cpu_size: 16.0` unless the node's host-memory budget has been
rechecked. A Spark0 gate run with `max_local_cpu_size: 64.0` reached the
FullAttentionSpec and hybrid-state initialization path, then collapsed
`MemAvailable` below 1 GiB before the API became healthy. The earlier
`chunk_size: 256` launch started successfully but LMCache rejected external hits
because matching hybrid-state pages were not available at the same token
boundary. Keep `--no-async-scheduling` for the rollout gate; enable
`--async-scheduling` only as a separate performance experiment after the capped
LMCache recipe is healthy.

Acceptance requires all of the following:

```text
Qwen27 starts with HMA plus LMCache.
No "LMCacheConnectorV1 does not support HMA" error.
No "failed to convert the KV cache specs to one unified type" error.
Logs show LMCache selecting a FullAttentionSpec group.
Logs show hybrid state groups detected or restored.
Repeated long-prefix request has materially lower TTFT.
After resetting vLLM prefix cache, a repeated request logs "Loaded hybrid state"
and "Retrieved ... out of ... required tokens".
```

The spark7 smoke on 2026-05-28 used a 16206-token prompt. With the aligned
recipe, the first request stored hybrid state at 7840 and 15680 tokens. After
`POST /v1/trim_memory` reset the local vLLM prefix cache, the next request
logged:

```text
LMCache hit tokens: 15680, need to load: 15680
Loaded hybrid state ... at 15680 token(s)
Retrieved 15680 out of 15680 required tokens
External prefix cache hit rate: 93.7%
```

## Spark Rollout

Keep the vLLM source tree and runtime branch identical on every node that may
host the service:

```bash
cd /home/$USER/src/vllm
git fetch experiencenow
git checkout main
git pull --ff-only experiencenow main
/home/$USER/ds4-vllm-local/bin/python -m py_compile \
  vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/core/single_type_kv_cache_manager.py \
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
```

For DSV4 nodes, no LMCache install is required. Verify the runtime still uses
SimpleCPUOffload and MTP by checking the launch command for:

```text
VLLM_USE_SIMPLE_KV_OFFLOAD=1
--kv-offloading-backend native
--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
--no-disable-hybrid-kv-cache-manager
```

For Qwen nodes, install the companion LMCache branch into the exact Python
environment used to start vLLM, then use the aligned Qwen recipe above. The
`/home/spark*/ds4-vllm-local/bin/vllm` wrapper may point at a different
interpreter on some Spark images; prefer:

```bash
export PATH=/home/$USER/ds4-vllm-local/bin:$PATH
/home/$USER/ds4-vllm-local/bin/python -m vllm.entrypoints.cli.main serve ...
```
