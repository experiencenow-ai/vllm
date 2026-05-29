# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.hma_block_ids import (
    build_slot_mapping_for_block_ids,
    build_slot_mappings_for_block_id_groups,
    choose_lmcache_kv_cache_group_id,
    extend_block_id_groups,
    normalize_block_id_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


def test_normalize_block_id_groups_preserves_hma_groups() -> None:
    assert normalize_block_id_groups([1, 2, 3]) == ([1, 2, 3],)
    assert normalize_block_id_groups([[1], [2, 3]]) == ([1], [2, 3])
    assert normalize_block_id_groups(([4], [5, 6])) == ([4], [5, 6])


def test_extend_block_id_groups_preserves_hma_groups() -> None:
    assert extend_block_id_groups(([1], [10]), ([2, 3], [11])) == (
        [1, 2, 3],
        [10, 11],
    )
    assert extend_block_id_groups(([1], [10]), [2, 3]) == (
        [1, 2, 3],
        [10],
    )


def test_slot_mapping_uses_selected_block_group() -> None:
    mappings = build_slot_mappings_for_block_id_groups(
        ([1, 2], [20, 21]), block_size=4, token_count=6
    )
    assert torch.equal(mappings[0], torch.tensor([4, 5, 6, 7, 8, 9]))
    assert torch.equal(mappings[1], torch.tensor([80, 81, 82, 83, 84, 85]))
    assert torch.equal(
        build_slot_mapping_for_block_ids([7], block_size=4, token_count=3),
        torch.tensor([28, 29, 30]),
    )


def test_choose_lmcache_group_prefers_attention_over_mamba() -> None:
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((1, 2),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    attention_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.float16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=["layers.0"])],
        kv_cache_groups=[
            KVCacheGroupSpec(["layers.0"], mamba_spec),
            KVCacheGroupSpec(["layers.1"], attention_spec),
        ],
    )

    assert choose_lmcache_kv_cache_group_id(kv_cache_config, "auto") == 1
    assert choose_lmcache_kv_cache_group_id(kv_cache_config, 0) == 0
