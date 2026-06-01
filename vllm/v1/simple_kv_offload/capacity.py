# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CPU-block capacity helpers for SimpleCPUOffloadConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


def derive_logical_cpu_block_count(
    kv_cache_config: "KVCacheConfig", cpu_capacity_bytes: int
) -> int:
    """Return the logical CPU block-id space shared by scheduler and workers.

    SimpleCPUOffload metadata uses one flat CPU block id space that is sent to
    every worker.  Pipeline ranks can own different physical layer counts, so
    worker-side tensor bytes per block are not necessarily identical to the
    scheduler's KVCacheConfig estimate.  The logical id space must still be
    computed identically on both sides; otherwise the scheduler can emit a CPU
    block id that some PP stage never allocated.
    """
    assert len(kv_cache_config.kv_cache_tensors) > 0
    gpu_total_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    assert gpu_total_bytes > 0
    num_gpu_blocks = kv_cache_config.num_blocks
    return max(1, num_gpu_blocks * int(cpu_capacity_bytes) // gpu_total_bytes)
