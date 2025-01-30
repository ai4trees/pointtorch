"""Max pooling over all point features within a batch item."""

__all__ = ["max_pooling"]

import torch
from torch_scatter import scatter_max


def max_pooling(x: torch.Tensor, point_cloud_indices: torch.Tensor, expand: bool = False) -> torch.Tensor:
    r"""
    Max pooling over all point features within a batch item.

    Args:
        x: Point features to be pooled.
        point_cloud_indices: Indices indicating to which point cloud in the batch each point belongs.
        expand: Whether the max-pooling result should be expanded to the size of the input point clouds by repeating
            the maximum value for each point. Defaults to `False`.

    Returns:
        Max-pooled features.

    Shape:
        - | :attr:`x`: :math:`(N, D)`
        - | :attr:`point_cloud_indices`: :math:`(N)`
        - | Output: :math:`(B, D)` if :attr:`expand` is `False`, otherwise :math:`(N, D)`.
          |
          | where
          |
          | :math:`N = \text{ number of points}`
          | :math:`B = \text{ batch size}`
          | :math:`D = \text{ number of feature channels}`
    """

    max_features, _ = scatter_max(x, point_cloud_indices, dim=0)

    if expand:
        return torch.gather(max_features, dim=0, index=point_cloud_indices.unsqueeze(-1).expand(x.size()))

    return max_features
