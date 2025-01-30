"""Tests for `pointtorch.operations.torch.random_sampling`."""

from hypothesis import given, settings, strategies as st
import numpy as np
import torch

from pointtorch.operations.torch import random_sampling


class TestRandomSampling:  # pylint: disable=too-few-public-methods
    """Tests for `pointtorch.operations.torch.random_sampling`."""

    @settings(deadline=None)
    @given(
        device=st.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        )
    )
    def test_random_sampling(self, device: torch.device):

        points = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], device=device)
        batch_indices = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([4, 2], dtype=torch.long, device=device)
        new_point_cloud_sizes = torch.tensor([2, 1], dtype=torch.long, device=device)
        result = random_sampling(points, batch_indices, point_cloud_sizes, new_point_cloud_sizes).cpu().numpy()
        expected_result = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2]])

        np.testing.assert_array_equal(expected_result, result)
