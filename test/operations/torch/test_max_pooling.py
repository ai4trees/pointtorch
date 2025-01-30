"""Tests for the pointtorch.operations.max_pooling function."""

import hypothesis
import numpy
import torch

from pointtorch.operations.torch import max_pooling


class TestMaxPooling:  # pylint: disable=too-few-public-methods
    """Tests for the pointtorch.operations.max_pooling function."""

    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        ),
        expand=hypothesis.strategies.booleans(),
    )
    def test_maxpooling(self, device: torch.device, expand: bool):
        x = torch.stack(
            [torch.arange(9, dtype=torch.float, device=device), torch.arange(10, 19, dtype=torch.float, device=device)],
            dim=-1,
        )

        point_cloud_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=device)

        if expand:
            expected_result = torch.tensor(
                [[4, 14], [4, 14], [4, 14], [4, 14], [4, 14], [8, 18], [8, 18], [8, 18], [8, 18]], dtype=torch.float
            )
        else:
            expected_result = torch.tensor([[4, 14], [8, 18]], dtype=torch.float)

        max_pooling_result = max_pooling(x, point_cloud_indices, expand=expand)

        numpy.testing.assert_array_equal(expected_result.numpy(), max_pooling_result.cpu().numpy())
