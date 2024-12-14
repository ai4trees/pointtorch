""" Voxel-based downsampling of a point cloud. """

__all__ = ["voxel_downsampling"]

from typing import Literal, Optional, Tuple

import torch
from torch_scatter.scatter import scatter, scatter_min

from ._make_labels_consecutive import make_labels_consecutive
from ._ravel_index import ravel_multi_index, unravel_flat_index


def voxel_downsampling(  # pylint: disable=too-many-locals
    coords: torch.Tensor,
    batch_indices: torch.Tensor,
    point_cloud_sizes: torch.Tensor,
    voxel_size: float,
    *,
    features: Optional[torch.Tensor] = None,
    feature_aggregation: Literal["max", "mean", "min", "nearest_neighbor"] = "mean",
    point_aggregation: Literal["mean", "nearest_neighbor"] = "mean",
    preserve_order: bool = True,
    start: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Voxel-based downsampling of a batch of point clouds.

    Args:
        coords: Coordinates of the points to be downsampled.
        batch_indices: Indices indicating to which input point cloud each point in the batch belongs.
        point_cloud_sizes: Number of points contained in each input point cloud.
        voxel_size: The size of the voxels used for downsampling.
        features: The features of the points to be downsampled. Defaults to `None`.
        feature_aggregation: Method to be used to aggregate features of points within the same voxel: `"max"` | \
            `"mean"` | `"min"` | `"nearest_neighbor"`. `"nearest_neighbor"` means that the features of the point \
            closest to the voxel center are selected.
        point_aggregation: Method to be used to aggregate the point coordinates within the same voxel: `"mean"` |
            `"nearest_neighbor"`. `"nearest_neighbor"` means that the coordinates of the point closest to the voxel
            center are selected.
        preserve_order: If this is set to `True` and `point_aggregation` is set to `"nearest_neighbor"`, the point order
            is preserved during downsampling. This means that for any two points included in the downsampled point
            cloud, the point that is first in the original point cloud is also first in the downsampled point cloud.
            Defaults to `True`.
        start: Coordinates of a point at which the voxel grid is to be aligned, i.e., the grid is placed so that it
            starts at a corner point of a voxel. Defaults to `None`, which means that the grid is aligned at the
            coordinate origin.

    Returns:
        Tuple of five tensors. The first contains the coordinates of the downsampled points. The second tensor contains
        the features of the downsampled points and is `None` if no input features are provided. The third contains
        indices indicating to which point cloud each downsampled point belongs. The fourth contains the size of each
        downsampled point cloud. The fifth contains indices indicating in which voxel each point from the original point
        clouds is located.

    Raises:
        ValueError: If `start` is not `None` and has an invalid shape.

    Shape:
        - :attr:`coords`: :math:`(N, 3)`
        - :attr:`batch_indices`: :math:`(N)`
        - :attr:`point_cloud_sizes`: :math:`(B)`
        - :attr:`features`: :math:`(N, D)`
        - :attr:`start`: :math:`(B, 3)`
        - Output: Tuple of five tensors. The first has shape :math:`(N', 3)`. The second has shape :math:`(N', D)`. \
          The third has shape :math:`(N')`. The fourth has shape :math:`(B)`. The fifth has shape :math:`(N)`.

          | where
          |
          | :math:`N = \text{ number of points before downsampling}`
          | :math:`N' = \text{ number of points after downsampling}`
          | :math:`B = \text{ batch size}`
          | :math:`D = \text{ number of feature channels}`
    """

    if start is None:
        start_coords = torch.zeros((len(point_cloud_sizes), 3), dtype=torch.float, device=coords.device)
    else:
        if start.ndim != 2 or len(start) != len(point_cloud_sizes) or start.size(1) != coords.size(1):
            raise ValueError(f"The shape of the 'start' tensor is invalid: {start.shape}. ")

        start_coords = start

    shifted_coords = coords - start_coords[batch_indices]
    voxel_indices = torch.floor_divide(shifted_coords, voxel_size).long()  # (N, 3)

    # add batch index as additional coordinate dimension so that points from different batch items are put into
    # different voxels
    voxel_indices = torch.column_stack([batch_indices.unsqueeze(-1), voxel_indices])  # (N, 4)
    shift = voxel_indices.amin(dim=0, keepdim=True)  # (1, 4)
    voxel_indices = voxel_indices - shift
    shifted_coords = shifted_coords - shift[:, 1:].float() * voxel_size

    dimensions = voxel_indices.amax(0) + 1  # (4)
    flattened_indices = ravel_multi_index(voxel_indices, dimensions)  # (N)

    unqiue_cluster_indices, cluster = torch.unique(flattened_indices, sorted=True, return_inverse=True)

    cluster_centers = unravel_flat_index(unqiue_cluster_indices, dimensions)
    batch_indices = cluster_centers[:, 0]
    cluster_centers = cluster_centers[:, 1:].float() * voxel_size + 0.5 * voxel_size

    scatter_indices: torch.Tensor = make_labels_consecutive(  # type: ignore[assignment]
        flattened_indices - flattened_indices.min()
    )

    if point_aggregation == "nearest_neighbor" or features is not None and feature_aggregation == "nearest_neighbor":
        point_indices = torch.arange(len(shifted_coords), device=coords.device, dtype=torch.long)

        dists_to_cluster_centers = torch.linalg.norm(  # pylint: disable=not-callable
            shifted_coords - cluster_centers[cluster], dim=-1
        )

        _, argmin_indices = scatter_min(dists_to_cluster_centers, scatter_indices)

        selected_indices = point_indices[argmin_indices]
        if preserve_order and point_aggregation == "nearest_neighbor":
            selected_indices, sorting_indices = selected_indices.sort()

    if point_aggregation == "nearest_neighbor":
        coords = coords[selected_indices]
    else:
        coords = scatter(coords, scatter_indices, dim=0, reduce="mean")

    if features is not None:
        if feature_aggregation == "nearest_neighbor":
            features = features[selected_indices]
        else:
            features = scatter(features, scatter_indices, dim=0, reduce=feature_aggregation)
            if preserve_order and point_aggregation == "nearest_neighbor":
                features = features[sorting_indices]  # pylint: disable=used-before-assignment

    point_cloud_sizes = scatter(torch.ones_like(batch_indices), batch_indices, reduce="sum")

    return (coords, features, batch_indices, point_cloud_sizes, cluster)
