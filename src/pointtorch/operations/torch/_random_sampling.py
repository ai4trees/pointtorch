"""Random sampling of points from a point cloud."""

__all__ = ["random_sampling"]

import torch


def random_sampling(
    points: torch.Tensor,
    batch_indices: torch.Tensor,
    point_cloud_sizes: torch.Tensor,
    new_point_cloud_sizes: torch.Tensor,
) -> torch.Tensor:
    r"""
    Randomly samples a given number of points from each point cloud in a batch. From each point cloud in a batch, as
    many points as specified by :attr:`new_point_cloud_sizes` are selected. The points are expected to be shuffled
    beforehand, so the first `new_point_cloud_sizes[i]` points are selected from the i-th point cloud and no shuffling
    is performed by this method.

    Args:
        points: Batch of point clouds from which to sample.
        batch_indices: Indices indicating to which input point cloud each point in the batch belongs.
        point_cloud_sizes: Number of points contained in each input point cloud.
        new_point_cloud_sizes: Number of points to be sampled from each input point cloud.


    Returns:
        Sampling results.

    Shape:
        - :attr:`points`: :math:`(N, ...)`
        - :attr:`batch_indices`: :math:`(N)`
        - :attr:`point_cloud_sizes`: :math:`(B)`
        - :attr:`new_point_cloud_sizes`: :math:`(B)`
        - Output: :math:`(N', ...)`

          | where
          |
          | :math:`N = \text{ number of points before sampling}`
          | :math:`N' = \text{ number of points after sampling}`
          | :math:`B = \text{ batch size}`
    """

    point_indices = torch.arange(len(points), dtype=torch.long, device=points.device)

    point_cloud_start_indices = torch.zeros(len(point_cloud_sizes), dtype=torch.long, device=points.device)
    point_cloud_start_indices[1:] = point_cloud_sizes.cumsum(dim=0)[:-1]

    point_cloud_end_indices = point_cloud_start_indices + new_point_cloud_sizes

    point_cloud_start_indices = torch.gather(point_cloud_start_indices, dim=0, index=batch_indices)
    point_cloud_end_indices = torch.gather(point_cloud_end_indices, dim=0, index=batch_indices)

    sampling_mask = torch.logical_and(
        point_indices >= point_cloud_start_indices, point_indices < point_cloud_end_indices
    )

    return points[sampling_mask]
