# DS4 PP Conveyor Fast Path

DS4 pipeline-parallel services should not treat every PP boundary send as a
hard barrier before the next scheduler wave can execute.  That shape makes the
hot GPU move rank-by-rank and leaves upstream/downstream ranks idle while one
large scheduled batch drains over the PP edge.

The DS4 conveyor path keeps the vLLM `IntermediateTensors` contract, but changes
the lifetime and synchronization policy:

```text
forward produces PP tensors
  -> copy CUDA tensor payloads into owned send-buffer slot
  -> enqueue PP tensor-dict send from that owned slot
  -> next scheduler wave may run immediately
  -> wait only when every reusable send-buffer slot is still in flight
```

The producer tensors can be released or reused by the model runner because the
PP transfer owns its CUDA buffers.  This lets PP transfer latency overlap with
receive/setup/compute for later scheduler waves.

Default DS4 service knobs:

```bash
VLLM_DS4_PP_DIRECT_CUDA_TENSOR_DICT=1
VLLM_DS4_PP_OVERLAP_SEND=1
VLLM_DS4_PP_SEND_BUFFER_SLOTS=4
VLLM_DS4_PP_SEND_BUFFER_MAX_BYTES=1073741824
VLLM_DS4_PP_GANTT_TRACE=0
VLLM_DS4_PP_GANTT_TRACE_EVERY=10
```

DSV4 throughput profiles also admit requests in conveyor-shaped waves:

```bash
DSV4_SCHED_MAX_NEW_REQS_PER_STEP=64
```

This still allows high in-flight concurrency such as c256/c512, but avoids
turning one prompt-array request into one monolithic scheduler batch.

For diagnosis:

```bash
VLLM_DS4_PP_GANTT_TRACE=1
VLLM_DS4_PP_GANTT_TRACE_EVERY=10
```

Healthy traces should show `recv_post`, `forward_start`,
`forward_done_intermediate`, and `send_enqueue_buffered` events overlapping
across PP ranks with few `send_buffer_wait` events.
