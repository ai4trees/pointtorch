""" Tests for the pointtorch.operations.pack_batch function. """

import hypothesis
import numpy
import torch

from pointtorch.operations.torch import pack_batch


class TestPackBatch:
    """Tests for the pointtorch.operations.pack_batch function."""

    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        )
    )
    def test_same_size_point_clouds(self, device: torch.device):
        batch_size = 16
        point_cloud_size = 1024
        points = torch.randn(batch_size * point_cloud_size, 3, device=device, dtype=torch.float)
        point_cloud_sizes = torch.full((batch_size,), fill_value=point_cloud_size, device=device, dtype=torch.long)

        batch, mask = pack_batch(points, point_cloud_sizes)

        assert torch.Size((batch_size, point_cloud_size, 3)) == batch.size()
        numpy.testing.assert_allclose(
            points.reshape(batch_size, point_cloud_size, -1).cpu().numpy(), batch.cpu().numpy()
        )
        assert mask.all().item()

    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        )
    )
    def test_variable_size_point_clouds(self, device: torch.device):
        batch_size = 2
        point_cloud_sizes = torch.tensor([1024, 512], device=device, dtype=torch.long)
        max_point_cloud_size = 1024
        points = torch.randn(int(point_cloud_sizes.sum().item()), 3, device=device, dtype=torch.float)

        batch, mask = pack_batch(points, point_cloud_sizes)

        assert torch.Size((batch_size, max_point_cloud_size, 3)) == batch.size()

        numpy.testing.assert_allclose(points.cpu().numpy(), batch[mask].cpu().numpy())
        assert point_cloud_sizes.sum().item() == mask.sum().item()
