"""Voxel-based downsampling of a point cloud."""

__all__ = ["voxel_downsampling"]

from typing import Literal, Optional, Tuple

import numpy as np

from pointtorch.type_aliases import FloatArray, LongArray


def voxel_downsampling(  # pylint: disable=too-many-locals
    points: FloatArray,
    voxel_size: float,
    point_aggregation: Literal["nearest_neighbor", "random"] = "random",
    preserve_order: bool = True,
    start: Optional[FloatArray] = None,
) -> Tuple[FloatArray, LongArray, LongArray]:
    r"""
    Voxel-based downsampling of a point cloud.

    Args:
        points: The point cloud to downsample.
        voxel_size: The size of the voxels used for downsampling. If :code:`voxel_size` is set to zero or less, no
            downsampling is applied.
        point_aggregation: Method to be used to aggregate the points within the same voxel. Defaults to
            `nearest_neighbor`. `"nearest_neighbor"`: The point closest to the voxel center is selected. `"random"`:
            One point is randomly sampled from the voxel.
        preserve_order: If set to `True`, the point order is preserved during downsampling. This means that for any two
            points included in the downsampled point cloud, the point that is first in the original point cloud is
            also first in the downsampled point cloud. Defaults to `True`.
        start: Coordinates of a point at which the voxel grid is to be aligned, i.e., the grid is placed so that
            :code:`start` is at a corner point of a voxel. Defaults to `None`, which means that the grid is aligned at
            the coordinate origin.

    Returns:
        Tuple of three arrays. The first contains the points remaining after downsampling. The second contains the \
        indices of the points remaining after downsampling within the original point cloud. The third contains the
        indices of the voxel to which each point in the input point cloud belongs.

    Raises:
        ValueError: If `start` is not `None` and has an invalid shape.
        ValueError: If `point_aggregation` is invalid.

    Shape:
        - :code:`points`: :math:`(N, 3 + D)`.
        - :code:`start`: :math:`(3)`
        - Output: Tuple of three arrays. The first has shape :math:`(N', 3 + D)`, the second :math:`(N')`, and the third
          :math:`(N)`

          | where
          |
          | :math:`N = \text{ number of points before downsampling}`
          | :math:`N' = \text{ number of points after downsampling}`
          | :math:`D = \text{ number of feature channels excluding coordinate channels }`
    """

    if voxel_size <= 0:
        return points, np.arange(len(points), dtype=np.int64), np.arange(len(points), dtype=np.int64)

    if len(points) == 0:
        return points, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    if start is None:
        start_coords = np.array([0.0, 0.0, 0.0])
    else:
        if start.shape != (3,):
            raise ValueError(f"The shape of the 'start' array is invalid: {start.shape}. ")
        start_coords = start

    shifted_points = points[:, :3] - start_coords
    voxel_indices = np.floor_divide(shifted_points, voxel_size).astype(np.int64)
    shift = voxel_indices.min(axis=0)
    voxel_indices = voxel_indices - shift
    dimensions = voxel_indices.max(axis=0) + 1
    flattened_indices = np.ravel_multi_index(tuple(voxel_indices.T), dimensions)
    _, selected_indices, inverse_indices = np.unique(flattened_indices, return_index=True, return_inverse=True)

    if point_aggregation == "nearest_neighbor":
        shifted_points = shifted_points - shift.astype(np.float64) * voxel_size
        voxel_centers = voxel_indices.astype(np.float64) * voxel_size + 0.5 * voxel_size
        squared_dists_to_voxel_center = np.square(shifted_points - voxel_centers).sum(axis=-1)

        sorting_indices = np.lexsort((squared_dists_to_voxel_center, inverse_indices))
        sorted_inverse_indices = inverse_indices[sorting_indices]
        first_in_voxel = np.empty(len(sorting_indices), dtype=bool)
        first_in_voxel[0] = True
        first_in_voxel[1:] = sorted_inverse_indices[1:] != sorted_inverse_indices[:-1]
        selected_indices = sorting_indices[first_in_voxel]
    elif point_aggregation != "random":
        raise ValueError(f"Invalid point aggregation method: {point_aggregation}.")

    if preserve_order:
        ordered_indices = selected_indices.argsort()
        selected_indices = selected_indices[ordered_indices]
        inverse_indices = np.argsort(ordered_indices)[inverse_indices]

    return points[selected_indices], selected_indices, inverse_indices
