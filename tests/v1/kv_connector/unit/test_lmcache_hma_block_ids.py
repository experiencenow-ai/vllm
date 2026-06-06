# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.hma_block_ids import (
    build_slot_mapping_for_block_ids,
    build_slot_mappings_for_block_id_groups,
    choose_lmcache_kv_cache_group_id,
    extend_block_id_groups,
    get_lmcache_kv_cache_group_layer_names,
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
    assert get_lmcache_kv_cache_group_layer_names(kv_cache_config, 1) == (
        "layers.1",
    )


def test_lmcache_grouped_engine_uses_group_chunk_size(monkeypatch) -> None:
    pytest.importorskip("lmcache")
    from lmcache.v1.config import LMCacheEngineConfig

    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (
        vllm_v1_adapter,
    )

    captured = {}

    class FakePagedMemGPUConnector:
        def __init__(
            self,
            hidden_dim_size,
            num_layer,
            *,
            use_gpu,
            chunk_size,
            dtype,
            device,
            use_mla,
        ):
            captured["hidden_dim_size"] = hidden_dim_size
            captured["num_layer"] = num_layer
            captured["chunk_size"] = chunk_size
            captured["use_gpu"] = use_gpu
            captured["dtype"] = dtype
            captured["device"] = device
            captured["use_mla"] = use_mla

    def fake_get_or_create(
        engine_name,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        broadcast,
        broadcast_object,
    ):
        captured["engine_name"] = engine_name
        captured["metadata"] = metadata
        captured["connector"] = vllm_gpu_connector
        return SimpleNamespace(metadata=None)

    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=32,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=["layers.0"])],
        kv_cache_groups=[
            KVCacheGroupSpec(["layers.0", "layers.1", "layers.2"], attention_spec),
        ],
    )
    lmcache_config = LMCacheEngineConfig(chunk_size=384)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            model="test-model",
            served_model_name=None,
        ),
        parallel_config=SimpleNamespace(rank=0, world_size=1),
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
        kv_transfer_config=SimpleNamespace(
            engine_id=None,
            kv_connector_extra_config={},
        ),
        speculative_config=None,
    )
    fake_tp_group = SimpleNamespace(
        broadcast=MagicMock(),
        broadcast_object=MagicMock(),
    )

    monkeypatch.setattr(vllm_v1_adapter.LMCacheEngineBuilder, "get", lambda _: None)
    monkeypatch.setattr(
        vllm_v1_adapter.LMCacheEngineBuilder,
        "get_or_create",
        fake_get_or_create,
    )
    monkeypatch.setattr(
        vllm_v1_adapter,
        "VLLMPagedMemGPUConnectorV2",
        FakePagedMemGPUConnector,
    )
    monkeypatch.setattr(vllm_v1_adapter, "get_tp_group", lambda: fake_tp_group)
    monkeypatch.setattr(vllm_v1_adapter.torch.accelerator, "device_count", lambda: 1)
    monkeypatch.setattr(
        vllm_v1_adapter.torch.accelerator,
        "set_device_index",
        lambda _: None,
    )

    vllm_v1_adapter._init_lmcache_engine(
        lmcache_config,
        vllm_config,
        kv_cache_config=kv_cache_config,
    )

    assert captured["chunk_size"] == 384
    assert captured["hidden_dim_size"] == 128
    assert captured["num_layer"] == 3
    assert captured["metadata"].kv_shape == (3, 2, 384, 4, 32)
