import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops.attention import (
    _pad_positions_to_num_rows,
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
