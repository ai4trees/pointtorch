"""Tests for the index raveling operations in pointtorch.operations.torch"""

import numpy as np
import pytest
import torch

from pointtorch.operations.torch import ravel_index, ravel_multi_index, unravel_flat_index


class TestRavelIndex:
    """Tests for the index raveling operations in pointtorch.operations.torch"""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_ravel_index_invalid_dim(self, device: torch.device):
        with pytest.raises(ValueError):
            ravel_index(torch.zeros((5, 3), dtype=torch.long, device=device), torch.zeros((10, 2), device=device), 2)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_ravel_index_valid(self, dim: int, device: torch.device):
        input_tensor = torch.randn((10, 5, 8), device=device)

        index_shape = list(input_tensor.shape)
        index_shape[dim] = 2
        index = torch.randint(0, input_tensor.size(dim), index_shape, device=device)

        raveled_index = ravel_index(index, input_tensor, dim)

        expected_indexing_result = torch.gather(input_tensor, dim, index).flatten()
        indexing_result = input_tensor.flatten()[raveled_index]

        np.testing.assert_array_equal(expected_indexing_result.cpu().numpy(), indexing_result.cpu().numpy())

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_ravel_multi_index(self, device: torch.device):
        index_shape = torch.Size((10, 20, 30))
        index_size = 15
        multi_index = [torch.randint(0, shape, (index_size,), device=device) for shape in index_shape]

        raveled_index = ravel_multi_index(torch.column_stack(multi_index), index_shape)
        raveled_index_np = np.ravel_multi_index([index.cpu().numpy() for index in multi_index], index_shape)

        unraveled_index = unravel_flat_index(raveled_index, index_shape)

        np.testing.assert_array_equal(raveled_index_np, raveled_index.cpu().numpy())
        np.testing.assert_array_equal(torch.column_stack(multi_index).cpu().numpy(), unraveled_index.cpu().numpy())
