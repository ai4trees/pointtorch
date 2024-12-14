""" Voxel-based downsampling of a point cloud. """

__all__ = ["voxel_downsampling"]

from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch_scatter.scatter import scatter_min

from ._make_labels_consecutive import make_labels_consecutive


def voxel_downsampling(  # pylint: disable=too-many-locals
    points: npt.NDArray[np.float64],
    voxel_size: float,
    point_aggregation: Literal["nearest_neighbor", "random"] = "random",
    preserve_order: bool = True,
    start: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Voxel-based downsampling of a point cloud.

    Args:
        points: The point cloud to downsample.
        voxel_size: The size of the voxels used for downsampling. If :attr:`voxel_size` is set to zero or less, no
            downsampling is applied.
        point_aggregation: Method to be used to aggregate the points within the same voxel. Defaults to
            `nearest_neighbor`. `"nearest_neighbor"`: The point closest to the voxel center is selected. `"random"`:
            One point is randomly sampled from the voxel.
        preserve_order: If set to `True`, the point order is preserved during downsampling. This means that for any two
            points included in the downsampled point cloud, the point that is first in the original point cloud is
            also first in the downsampled point cloud. Defaults to `True`.
        start: Coordinates of a point at which the voxel grid is to be aligned, i.e., the grid is placed so that
            :attr:`start` is at a corner point of a voxel. Defaults to `None`, which means that the grid is aligned at
            the coordinate origin.

    Returns:
        Tuple of three arrays. The first contains the points remaining after downsampling. The second contains the \
        indices of the points remaining after downsampling within the original point cloud. The third contains the
        indices of the voxel to which each point in the input point cloud belongs.

    Raises:
        ValueError: If `start` is not `None` and has an invalid shape.

    Shape:
        - :attr:`points`: :math:`(N, 3 + D)`.
        - :attr:`start`: :math:`(3)`
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

    if start is None:
        start_coords = np.array([0.0, 0.0, 0.0])
    else:
        if start.shape != np.array([3]):
            raise ValueError(f"The shape of the 'start' array is invalid: {start.shape}. ")
        start_coords = start

    shifted_points = points[:, :3] - start_coords
    voxel_indices = np.floor_divide(shifted_points, voxel_size).astype(np.int64)

    if point_aggregation == "random":
        _, selected_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_index=True, return_inverse=True)
    else:
        shift = voxel_indices.min(axis=0)
        voxel_indices -= shift
        shifted_points -= shift.astype(float) * voxel_size

        filled_voxel_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        device = "cpu"

        # check if there is a GPU with sufficient memory to run the scatter_min step on GPU
        if torch.cuda.is_available():
            available_memory = torch.cuda.mem_get_info(device=torch.device("cuda:0"))[0]
            double_size = torch.empty((0,)).double().element_size()
            long_size = torch.empty((0,)).long().element_size()
            approx_required_memory = len(points) * 2 * (double_size + long_size)

            if available_memory > approx_required_memory:
                device = "cuda:0"

        voxel_centers = filled_voxel_indices.astype(float) * voxel_size + 0.5 * voxel_size

        dists_to_voxel_center = np.linalg.norm(shifted_points - voxel_centers[inverse_indices], axis=-1)

        dimensions = voxel_indices.max(axis=0) + 1
        scatter_indices: npt.NDArray[np.int64] = make_labels_consecutive(  # type: ignore[assignment]
            np.ravel_multi_index(tuple(voxel_indices[:, dim] for dim in range(voxel_indices.shape[1])), dimensions)
        )

        _, argmin_indices = scatter_min(
            torch.from_numpy(dists_to_voxel_center).to(device), torch.from_numpy(scatter_indices).long().to(device)
        )

        selected_indices = np.arange(len(points))[argmin_indices.cpu().numpy()]

    if preserve_order:
        ordered_indices = selected_indices.argsort()
        selected_indices = selected_indices[ordered_indices]
        inverse_indices = np.argsort(ordered_indices)[inverse_indices]

    return points[selected_indices], selected_indices, inverse_indices
