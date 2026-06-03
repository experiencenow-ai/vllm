import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops.attention import (
    _pad_positions_to_num_rows,
    _pad_positions_to_q_kv_rows,
)


def test_deepseek_v4_positions_padding_preserves_exact_rows():
    positions = torch.arange(4, dtype=torch.int64)

    padded = _pad_positions_to_num_rows(positions, 4)

    assert padded is positions


def test_deepseek_v4_positions_padding_extends_graph_rows():
    positions = torch.tensor([3, 4, 5], dtype=torch.int64)

    padded = _pad_positions_to_num_rows(positions, 6)

    assert padded.dtype == positions.dtype
    assert padded.tolist() == [3, 4, 5, 0, 0, 0]


def test_deepseek_v4_positions_padding_rejects_too_many_rows():
    positions = torch.arange(5, dtype=torch.int64)

    with pytest.raises(ValueError, match="positions exceed"):
        _pad_positions_to_num_rows(positions, 4)


def test_deepseek_v4_positions_padding_rejects_non_1d_padding():
    positions = torch.arange(6, dtype=torch.int64).view(2, 3)

    with pytest.raises(ValueError, match="must be 1-D"):
        _pad_positions_to_num_rows(positions, 4)


def test_deepseek_v4_fused_insert_padding_matches_q_kv_rows():
    positions = torch.tensor([8], dtype=torch.int64)
    q = torch.empty((4, 2), dtype=torch.float16)
    kv = torch.empty((4, 3), dtype=torch.float16)

    padded = _pad_positions_to_q_kv_rows(positions, q, kv)

    assert padded.tolist() == [8, 0, 0, 0]


def test_deepseek_v4_fused_insert_slices_graph_positions_to_q_kv_rows():
    positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    q = torch.empty((1, 2), dtype=torch.float16)
    kv = torch.empty((1, 3), dtype=torch.float16)

    padded = _pad_positions_to_q_kv_rows(positions, q, kv)

    assert padded.tolist() == [4]


def test_deepseek_v4_fused_insert_rejects_q_kv_row_mismatch():
    positions = torch.tensor([8], dtype=torch.int64)
    q = torch.empty((4, 2), dtype=torch.float16)
    kv = torch.empty((3, 3), dtype=torch.float16)

    with pytest.raises(ValueError, match="q and kv row counts"):
        _pad_positions_to_q_kv_rows(positions, q, kv)
