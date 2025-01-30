"""Shuffling of points in a point cloud."""

__all__ = ["shuffle"]

from typing import Optional, Tuple

import torch


def shuffle(
    points: torch.Tensor, point_cloud_sizes: torch.Tensor, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Shuffles points within a batch of point clouds. Each point cloud in the batch can contain a different number of
    points.

    Args:
        points: Batch of point clouds to shuffle.
        point_cloud_sizes: Number of points contained in each input point cloud.
        generator: Random generator to be used for shuffling. Defaults to `None`.

    Returns:
        Tuple of two tensors. The first is the shuffled tensor. The second contains the index of each point after \
        shuffling.

    Shape:
        - :attr:`points`: :math:`(N, ...)`
        - :attr:`point_cloud_sizes`: :math:`(B)`
        - Output: Tuple of two tensors. The first has the same shape as :attr:`points`. The second has shape `(N)`.

          | where
          |
          | :math:`N = \text{ number of points}`
          | :math:`B = \text{ batch size}`
    """

    max_point_cloud_size = int(point_cloud_sizes.max().item())
    shuffled_indices = torch.randperm(max_point_cloud_size, dtype=torch.long, device=points.device, generator=generator)
    shuffled_indices = shuffled_indices.unsqueeze(0).repeat(len(point_cloud_sizes), 1)
    invalid_mask = shuffled_indices >= point_cloud_sizes.unsqueeze(-1)
    shuffled_indices[1:] += point_cloud_sizes.cumsum(dim=0)[:-1].unsqueeze(-1)

    # mask invalid indices resulting from different point cloud sizes
    shuffled_indices[invalid_mask] = -1
    shuffled_indices = shuffled_indices.reshape(-1)
    shuffled_indices = shuffled_indices[shuffled_indices != -1]

    index_mapping = torch.argsort(shuffled_indices)

    return points[shuffled_indices], index_mapping
