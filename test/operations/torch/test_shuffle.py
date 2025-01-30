"""Tests for `pointtorch.operations.torch.shuffle`."""

import numpy as np
import pytest
import torch

from pointtorch.operations.torch import shuffle


class TestShuffle:
    """Tests for `pointtorch.operations.torch.shuffle`."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_shuffle_in_batch_items(self, device: str):
        points = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], device=device)
        point_cloud_sizes = torch.tensor([4, 2], dtype=torch.long, device=device)
        shuffled_points, _ = shuffle(points, point_cloud_sizes)

        np.testing.assert_array_equal(points.cpu().numpy(), shuffled_points.cpu().numpy())

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_shuffle_index_mapping(self, device: str):
        points = torch.tensor([[1, 2, 3], [1, 1, 1], [1, 3, 1], [3, 1, 2], [2, 2, 2], [1, 2, 3]], device=device)
        point_cloud_sizes = torch.tensor([4, 2], dtype=torch.long, device=device)
        shuffled_points, index_mapping = shuffle(points, point_cloud_sizes)

        np.testing.assert_array_equal(points.cpu().numpy(), shuffled_points[index_mapping].cpu().numpy())

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_shuffling(self, device: str):
        seed = 0
        generator = torch.Generator(device)
        generator.manual_seed(seed)

        points = torch.tensor([[1, 2, 3], [1, 1, 1], [1, 3, 1], [3, 1, 2], [2, 2, 2], [1, 2, 3]], device=device)
        point_cloud_sizes = torch.tensor([4, 2], dtype=torch.long, device=device)
        shuffled_points_1, _ = shuffle(points, point_cloud_sizes, generator=generator)
        shuffled_points_2, _ = shuffle(points, point_cloud_sizes, generator=generator)

        assert (shuffled_points_1 != shuffled_points_2).any()
