# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

import torch

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

BlockIdGroups = tuple[list[int], ...]


_ATTENTION_PREFERENCE = (
    KVCacheSpecKind.FULL_ATTENTION,
    KVCacheSpecKind.SINK_FULL_ATTENTION,
    KVCacheSpecKind.MLA_ATTENTION,
    KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION,
    KVCacheSpecKind.SLIDING_WINDOW_MLA,
    KVCacheSpecKind.SLIDING_WINDOW,
    KVCacheSpecKind.CROSS_ATTENTION,
)


def _is_integer_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, int) for item in value
    )


def normalize_block_id_groups(block_ids: object) -> BlockIdGroups:
    if block_ids is None:
        return ([],)

    if isinstance(block_ids, tuple):
        if len(block_ids) == 0:
            return ([],)
        if _is_integer_sequence(block_ids):
            return (list(block_ids),)
        return tuple(list(group) for group in block_ids)

    if isinstance(block_ids, list):
        if len(block_ids) == 0:
            return ([],)
        if _is_integer_sequence(block_ids):
            return (block_ids.copy(),)
        return tuple(list(group) for group in block_ids)

    raise ValueError(
        f"Unsupported block_ids type {type(block_ids)}: expected None, "
        "list[int], list[list[int]], or tuple[list[int], ...]."
    )


def extend_block_id_groups(
    allocated_block_id_groups: BlockIdGroups,
    new_block_ids: object,
) -> BlockIdGroups:
    new_block_id_groups = normalize_block_id_groups(new_block_ids)
    group_count = max(len(allocated_block_id_groups), len(new_block_id_groups))
    extended_groups: list[list[int]] = []
    for group_index in range(group_count):
        existing_group = (
            allocated_block_id_groups[group_index].copy()
            if group_index < len(allocated_block_id_groups)
            else []
        )
        if group_index < len(new_block_id_groups):
            existing_group.extend(new_block_id_groups[group_index])
        extended_groups.append(existing_group)
    return tuple(extended_groups)


def build_slot_mapping_for_block_ids(
    block_ids: Sequence[int],
    block_size: int,
    token_count: int,
) -> torch.Tensor:
    if token_count == 0:
        return torch.empty(0, dtype=torch.long)
    if len(block_ids) == 0:
        return torch.empty(0, dtype=torch.long)

    block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
    block_offsets = torch.arange(0, block_size, dtype=torch.long)
    slot_mapping = (
        block_offsets.reshape((1, block_size))
        + block_ids_tensor.reshape((len(block_ids), 1)) * block_size
    )
    return slot_mapping.flatten()[:token_count]


def build_slot_mappings_for_block_id_groups(
    allocated_block_id_groups: BlockIdGroups,
    block_size: int,
    token_count: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        build_slot_mapping_for_block_ids(block_ids, block_size, token_count)
        for block_ids in allocated_block_id_groups
    )


def _parse_explicit_group_id(requested_group_id: object) -> int | None:
    if requested_group_id is None:
        return None
    if isinstance(requested_group_id, str):
        stripped = requested_group_id.strip().lower()
        if stripped in ("", "auto"):
            return None
        return int(stripped)
    return int(requested_group_id)


def choose_lmcache_kv_cache_group_id(
    kv_cache_config: KVCacheConfig | None,
    requested_group_id: object = None,
) -> int:
    explicit_group_id = _parse_explicit_group_id(requested_group_id)

    if kv_cache_config is None:
        if explicit_group_id is None:
            return 0
        if explicit_group_id < 0:
            raise ValueError("LMCache KV cache group id must be non-negative.")
        return explicit_group_id

    groups = kv_cache_config.kv_cache_groups
    if len(groups) == 0:
        if explicit_group_id is None:
            return 0
        raise ValueError(
            "LMCache KV cache group id was set, but the KV cache config has "
            "no groups."
        )

    if explicit_group_id is not None:
        if explicit_group_id < 0 or explicit_group_id >= len(groups):
            raise ValueError(
                f"LMCache KV cache group id {explicit_group_id} is outside the "
                f"available KV cache groups [0, {len(groups) - 1}]."
            )
        return explicit_group_id

    indexed_groups: list[tuple[int, Any]] = list(enumerate(groups))
    non_empty_groups = [item for item in indexed_groups if item[1].layer_names]
    if len(non_empty_groups) > 0:
        indexed_groups = non_empty_groups

    group_kinds = [
        (group_index, get_kv_cache_spec_kind(group.kv_cache_spec))
        for group_index, group in indexed_groups
    ]
    for preferred_kind in _ATTENTION_PREFERENCE:
        for group_index, group_kind in group_kinds:
            if group_kind == preferred_kind:
                return group_index

    return indexed_groups[0][0]
