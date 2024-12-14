""" Different implementations of radius neighborhood search. """

__all__ = [
    "radius_search",
    "radius_search_cdist",
    "radius_search_open3d",
    "radius_search_pytorch3d",
    "radius_search_torch_cluster",
]

import math
from typing import Optional

import torch
import torch_cluster

from pointtorch.config import open3d_is_available, pytorch3d_is_available
from ._neighbor_search import neighbor_search_cdist
from ._pack_batch import pack_batch


def radius_search(  # pylint: disable=too-many-positional-arguments
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    batch_indices_support_points: torch.Tensor,
    batch_indices_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    radius: float,
    voxel_size: Optional[float] = None,
    k: Optional[int] = None,
    return_sorted: bool = False,
) -> torch.Tensor:
    r"""
    Retrieves the indices of all neighbor points within a radius. Decides between different implementations:

    - **Implementation from Open3D:** The Open3D implementation is based on a spatial hash table and therefore \
      achieves a high memory efficiency if all neighbors within the given radius are to be searched, i.e., if `k` is \
      `None`. None of the available implementations provides options to sort the returned neighbors by distance or to \
      select the k-nearest neighbors from the search radius if it contains more than `k` points. Therefore, if \
      `return_sorted` is `True`, all neighbors have to be searched and sorted by distance afterwards. For this reason, \
      the Open3D implementation is used when `k` is `None` or `return_sorted` is `True` and Open3D is installed. The \
      Open3D implementation is only used when the input size is smaller than 1048576. This is because the memory \
      allocation of the Open3D implementation potentially contains a bug and does not work for very large inputs.

    - **Implementation from PyTorch3D:** PyTorch3D implements a hybrid search that limits the maximum number of \
      neighbors to `k`. The GPU-based implementation launches one CUDA thread per query point and then loops \
      through all support points until at most `k` neighbors are found within the search radius. The \
      implementation requires the point clouds to be packed into a regular batch structure. The implementation is \
      quite efficient for small values of `k`. For a batch size of 1, it usually outperforms the torch-cluster \
      implementation. For larger batch sizes, the torch-cluster implementation uses slightly less memory because it \
      can handle variable-sized point clouds directly and does not require packing to a regular batch. Therefore, the \
      PyTorch3D implementation is used when `k` is not `None` and the batch size is 1 and PyTorch3D is installed. The
      torch-cluster implementation is used when `k` is not `None` and the batch size is greater than 1 or when \
      PyTorch3D is not installed.

    - **Implementation from torch-cluster:** The torch-cluster implementation is quite similar to the PyTorch3D \
      implementation but it can handle variable size point clouds directly.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        batch_indices_support_points: Indices indicating to which point cloud in the batch each support point belongs.
        batch_indices_query_points: Indices indicating to which point cloud in the batch each query point belongs.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        radius: Search radius in which to search for neighbors.
        voxel_size: Voxel size that was used to downsample the support point clouds before passing them to this method.
            If specified, this information can be used to calculate the maximum possible number of points within the
            search radius, which may be used to reduce the memory consumption of the neighbor search. Defaults to
            `None`.
        k: The maximum number of neighbors to search. If the radius neighborhood of a point contains more than `k`
            points, the returned neighbors are picked randomly if `return_sorted` is `False`. Otherwise, the `k`
            nearest neighbors are selected. Defaults to `None`, which means that all neighbors within the specified
            radius are returned.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `False`.

    Returns:
        The indices of the neighbors of each query point. If a query point has less than :math:`n_{max}` neighbors,
        where :math:`n_{max}` is the maximum number of neighbors a query point has, the invalid neighbor indices are set
        to :math:`N + 1` where :math:`N` is the number of support points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`batch_indices_support_points`: :math:`(N)`
        - :attr:`batch_indices_query_points`: :math:`(N')`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: :math:`(N', n_{max})` if :attr:`k` is None or :math:`n_{max} <` :attr:`k`, otherwise :math:`(N', k)`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    batch_size = len(point_cloud_sizes_query_points)
    input_size = int(point_cloud_sizes_query_points.sum().item())

    if open3d_is_available() and (k is None or return_sorted) and input_size <= 1048576:
        return radius_search_open3d(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            radius,
            k=k,
            return_sorted=return_sorted,
        )

    if pytorch3d_is_available() and k is not None and return_sorted is False and batch_size == 1:
        return radius_search_pytorch3d(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            radius,
            voxel_size=voxel_size,
            k=k,
            return_sorted=return_sorted,
        )

    return radius_search_torch_cluster(
        coords_support_points,
        coords_query_points,
        batch_indices_support_points,
        batch_indices_query_points,
        point_cloud_sizes_support_points,
        radius,
        voxel_size=voxel_size,
        k=k,
        return_sorted=return_sorted,
    )


def radius_search_cdist(  # pylint: disable=too-many-positional-arguments
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    radius: float,
    voxel_size: Optional[float] = None,
    k: Optional[int] = None,
    return_sorted: bool = False,
) -> torch.Tensor:
    r"""

    Computes the indices of all neighbor points within a radius.  This implementation packs the point clouds into a
    regular batch structure and pads missing points so that all neighbor indices can be computed in parallel using
    `PyTorch's cdist <https://pytorch.org/docs/stable/generated/torch.cdist.html>`_
    function. The memory consumption of this implementation is quadratic with regard to the number of points and
    can therefore only be used for small point cloud sizes.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        radius: Search radius in which to search for neighbors.
        voxel_size: Voxel size that was used to downsample the support point clouds before passing them to this method.
            If specified, this information can be used to calculate the maximum possible number of points within the
            search radius, which may be used to reduce the memory consumption of the neighbor search. Defaults to
            `None`.
        k: The maximum number of neighbors to search. If the radius neighborhood of a point contains more than `k`
            points, the returned neighbors are picked randomly if `return_sorted` is `False`. Otherwise, the `k`
            nearest neighbors are selected. Defaults to `None`, which means that all neighbors within the specified
            radius are returned.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `False`.

    Returns:
        The indices of the neighbors of each query point. If a query point has less than :math:`n_{max}` neighbors,
        where :math:`n_{max}` is the maximum number of neighbors a query point has, the invalid neighbor indices are set
        to :math:`N + 1` where :math:`N` is the number of support points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: :math:`(N', n_{max})` if :attr:`k` is None or :math:`n_{max} <` :attr:`k`, otherwise :math:`(N', k)`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`

    Raises:
        ValueError: If the input point clouds contain too many points to compute a pairwise distance matrix.
    """

    max_num_neighbors = int(point_cloud_sizes_support_points.amax().item())

    if voxel_size is not None:
        max_num_neighbors = min(max_num_neighbors, int(math.ceil(4 / 3 * math.pi * ((radius / voxel_size) + 1) ** 3)))

    if k is not None:
        max_num_neighbors = min(max_num_neighbors, k)

    neighbor_indices, _ = neighbor_search_cdist(
        coords_support_points,
        coords_query_points,
        point_cloud_sizes_support_points,
        point_cloud_sizes_query_points,
        radius=radius,
        k=max_num_neighbors,
        return_sorted=return_sorted,
    )
    return neighbor_indices


def radius_search_open3d(  # pylint: disable=too-many-locals, too-many-positional-arguments
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    radius: float,
    k: Optional[int] = None,
    return_sorted: bool = False,
) -> torch.Tensor:
    r"""
    Computes the indices of all neighbor points within a radius. This implementation is based on Open3D and uses a
    spatial hash table to implement the neighborhood search.
    Computes the indices of all neighbor points within a radius. This implementation is based on
    `Open3D's fixed_radius_search \
    <https://www.open3d.org/docs/release/python_api/open3d.ml.torch.layers.FixedRadiusSearch.html>`_ function
    and currently only supports CPU devices.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        radius: Search radius in which to search for neighbors.
        k: The maximum number of neighbors to search. If the radius neighborhood of a point contains more than `k`
            points, the returned neighbors are picked randomly if `return_sorted` is `False`. Otherwise, the `k`
            nearest neighbors are selected. Defaults to `None`, which means that all neighbors within the specified
            radius are returned.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `False`.

    Returns:
        The indices of the neighbors of each query point. If a query point has less than :math:`n_{max}` neighbors,
        where :math:`n_{max}` is the maximum number of neighbors a query point has, the invalid neighbor indices are set
        to :math:`N + 1` where :math:`N` is the number of support points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: :math:`(N', n_{max})` if :attr:`k` is None or :math:`n_{max} <` :attr:`k`, otherwise :math:`(N', k)`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    # the import is placed inside this method because the Open3D-ML extension might not be available on all
    # systems
    from open3d.ml.torch.ops import (  # pylint: disable=import-error,import-outside-toplevel
        build_spatial_hash_table,
        fixed_radius_search,
    )

    coords_support_points = coords_support_points.float()
    coords_query_points = coords_query_points.float()

    device = coords_query_points.device
    num_batches = len(point_cloud_sizes_support_points)
    num_query_points = len(coords_query_points)
    num_support_points = len(coords_support_points)
    invalid_neighbor_index = num_support_points

    splits_support_points = torch.zeros(num_batches + 1, device=device, dtype=torch.long)
    splits_support_points[1:] = torch.cumsum(point_cloud_sizes_support_points, dim=0)
    splits_query_points = torch.zeros(num_batches + 1, device=device, dtype=torch.long)
    splits_query_points[1:] = torch.cumsum(point_cloud_sizes_query_points, dim=0)

    hash_table = build_spatial_hash_table(
        coords_support_points, radius, points_row_splits=splits_support_points, hash_table_size_factor=1 / 32
    )

    neighbors_index, neighbors_splits, neighbors_distance = fixed_radius_search(
        coords_support_points,
        coords_query_points,
        radius,
        splits_support_points,
        splits_query_points,
        **hash_table._asdict(),
        return_distances=True,
        index_dtype=torch.long,
    )

    neighborhood_sizes = torch.diff(neighbors_splits)
    del neighbors_splits
    max_neighbors = int(neighborhood_sizes.amax().item())

    neighbor_indices = (
        torch.arange(max_neighbors, dtype=torch.long, device=device).unsqueeze(0).repeat(num_query_points, 1)
    )
    neighbor_indices[neighbor_indices >= neighborhood_sizes.unsqueeze(-1)] = invalid_neighbor_index
    neighbor_indices = neighbor_indices.flatten()
    neighbor_indices[neighbor_indices != invalid_neighbor_index] = neighbors_index

    neighbor_distances = torch.full(
        [num_query_points * max_neighbors], fill_value=torch.inf, dtype=neighbors_distance.dtype, device=device
    )
    neighbor_distances[neighbor_indices != invalid_neighbor_index] = neighbors_distance

    neighbor_indices = neighbor_indices.reshape(num_query_points, max_neighbors)
    neighbor_distances = neighbor_distances.reshape(num_query_points, max_neighbors)

    if return_sorted:
        if k is not None and k < max_neighbors:
            max_num_neighbors = k
        else:
            max_num_neighbors = max_neighbors
        _, sorted_indices = torch.topk(neighbor_distances, k=max_num_neighbors, dim=-1, sorted=True, largest=False)
        neighbor_indices = torch.gather(neighbor_indices, -1, sorted_indices)

    elif k is not None and k < max_neighbors:
        neighbor_indices = neighbor_indices[:, :k]

    return neighbor_indices


def radius_search_pytorch3d(  # pylint: disable=too-many-locals, too-many-positional-arguments
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    radius: float,
    voxel_size: Optional[float] = None,
    k: Optional[int] = None,
    return_sorted: bool = False,
) -> torch.Tensor:
    r"""
    Computes the indices of all neighbors within a radius. This implementation is based on
    `PyTorch3D's ball_query <https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.ball_query>`_
    function.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        radius: Search radius in which to search for neighbors.
        voxel_size: Voxel size that was used to downsample the support point clouds before passing them to this method.
            If specified, this information can be used to calculate the maximum possible number of points within the
            search radius, which may be used to reduce the memory consumption of the neighbor search. Defaults to
            `None`.
        k: The maximum number of neighbors to search. If the radius neighborhood of a point contains more than `k`
            points, the returned neighbors are picked randomly if `return_sorted` is `False`. Otherwise, the `k`
            nearest neighbors are selected. Defaults to `None`, which means that all neighbors within the specified
            radius are returned.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `False`.

    Returns:
        The indices of the neighbors of each query point. If a query point has less than :math:`n_{max}` neighbors,
        where :math:`n_{max}` is the maximum number of neighbors a query point has, the invalid neighbor indices are set
        to :math:`N + 1` where :math:`N` is the number of support points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`batch_indices_support_points`: :math:`(N)`
        - :attr:`batch_indices_query_points`: :math:`(N')`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: :math:`(N', n_{max})` if :attr:`k` is None or :math:`n_{max} <` :attr:`k`, otherwise :math:`(N', k)`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    # the import is placed inside this method because the Pytorch3D package is an optional dependency and might not be
    # available on all systems
    from pytorch3d.ops import ball_query  # pylint: disable=import-error,import-outside-toplevel

    num_support_points = len(coords_support_points)
    invalid_neighbor_index = num_support_points

    coords_support_points = coords_support_points.float()
    coords_query_points = coords_query_points.float()

    coords_query_points_batch, mask_query_points = pack_batch(
        coords_query_points, point_cloud_sizes_query_points, fill_value=-torch.inf
    )
    coords_support_points_batch, _ = pack_batch(
        coords_support_points, point_cloud_sizes_support_points, fill_value=torch.inf
    )

    max_num_neighbors = int(point_cloud_sizes_support_points.amax().item())

    if voxel_size is not None:
        max_num_neighbors = min(max_num_neighbors, int(math.ceil(4 / 3 * math.pi * ((radius / voxel_size) + 1) ** 3)))

    if k is not None and not return_sorted:
        max_num_neighbors = min(max_num_neighbors, k)

    # if the maximum possible number of neighbors cannot be determined from the function parameters, we simply set the
    # maximum number of neighbors to 200 and double it until all points have fewer neighbors than the set maximum value
    if k is not None:
        current_k = min(k, max_num_neighbors)
    else:
        current_k = min(200, max_num_neighbors)

    while True:
        neighbor_distances, neighbor_indices, _ = ball_query(
            coords_query_points_batch,
            coords_support_points_batch,
            point_cloud_sizes_query_points,
            point_cloud_sizes_support_points,
            radius=radius,
            K=current_k,
            return_nn=False,
        )

        max_neighbors = int((neighbor_indices >= 0).sum(dim=-1).max().item())

        if max_neighbors < current_k or current_k >= max_num_neighbors:
            break
        current_k *= 2

    neighbor_indices = neighbor_indices[:, :, :max_neighbors]
    neighbor_distances = neighbor_distances[:, :, :max_neighbors]

    batch_start_index = torch.cumsum(point_cloud_sizes_support_points, dim=0) - point_cloud_sizes_support_points
    batch_start_index = batch_start_index.unsqueeze(-1).unsqueeze(-1)  # convert to shape (B, 1, 1)

    invalid_neighbor_mask = neighbor_indices < 0
    neighbor_indices += batch_start_index
    neighbor_indices[invalid_neighbor_mask] = invalid_neighbor_index
    neighbor_distances[invalid_neighbor_mask] = torch.inf

    neighbor_indices = neighbor_indices[mask_query_points]
    neighbor_distances = neighbor_distances[mask_query_points]

    if return_sorted:
        if k is not None and k < max_neighbors:
            max_num_neighbors = k
        else:
            max_num_neighbors = max_neighbors

        _, sorted_indices = torch.topk(neighbor_distances, k=max_num_neighbors, dim=-1, sorted=True, largest=False)
        neighbor_indices = torch.gather(neighbor_indices, -1, sorted_indices)

    return neighbor_indices


def radius_search_torch_cluster(  # pylint: disable=too-many-locals, too-many-positional-arguments
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    batch_indices_support_points: torch.Tensor,
    batch_indices_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    radius: float,
    voxel_size: Optional[float] = None,
    k: Optional[int] = None,
    return_sorted: bool = False,
) -> torch.Tensor:
    r"""
    Computes the indices of all neighbor points within a radius. This implementation is based on the
    `radius method from torch-cluster <https://github.com/rusty1s/pytorch_cluster>`_.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        batch_indices_support_points: Indices indicating to which point cloud in the batch each support point belongs.
        batch_indices_query_points: Indices indicating to which point cloud in the batch each query point belongs.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        radius: Search radius in which to search for neighbors.
        voxel_size: Voxel size that was used to downsample the support point clouds before passing them to this method.
            If specified, this information can be used to calculate the maximum possible number of points within the
            search radius, which may be used to reduce the memory consumption of the neighbor search. Defaults to
            `None`.
        k: The maximum number of neighbors to search. If the radius neighborhood of a point contains more than `k`
            points, the returned neighbors are picked randomly if `return_sorted` is `False`. Otherwise, the `k`
            nearest neighbors are selected. Defaults to `None`, which means that all neighbors within the specified
            radius are returned.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `False`.

    Returns:
        The indices of the neighbors of each query point. If a query point has less than :math:`n_{max}` neighbors,
        where :math:`n_{max}` is the maximum number of neighbors a query point has, the invalid neighbor indices are set
        to :math:`N + 1` where :math:`N` is the number of support points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`batch_indices_support_points`: :math:`(N)`
        - :attr:`batch_indices_query_points`: :math:`(N')`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - Output: :math:`(N', n_{max})` if :attr:`k` is None or :math:`n_{max} <` :attr:`k`, otherwise :math:`(N', k)`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    device = coords_query_points.device
    num_query_points = len(coords_query_points)
    num_support_points = len(coords_support_points)
    invalid_neighbor_index = num_support_points

    max_num_neighbors = int(point_cloud_sizes_support_points.amax().item())

    if voxel_size is not None:
        max_num_neighbors = min(max_num_neighbors, int(math.ceil(4 / 3 * math.pi * ((radius / voxel_size) + 1) ** 3)))

    if k is not None and not return_sorted:
        max_num_neighbors = min(max_num_neighbors, k)

    # if the maximum possible number of neighbors cannot be determined from the function parameters, we simply set the
    # maximum number of neighbors to 200 and double it until all points have fewer neighbors than the set maximum value
    if k is not None:
        current_k = min(k, max_num_neighbors)
    else:
        current_k = min(200, max_num_neighbors)

    while True:
        num_elements_neighbor_matrix = num_query_points * current_k

        if num_elements_neighbor_matrix > torch.iinfo(torch.int).max:
            error_str = (
                f"The size of the neighbor index matrix would exceed the maximum supported size. Try reducing "
                f"the number of query or support points, k, or the search radius"
                f'{"or set `return_sorted` to False" if return_sorted else ""}.'
            )
            raise ValueError(error_str)

        neighbor_graph_edge_indices = torch_cluster.radius(
            coords_support_points,
            coords_query_points,
            radius,
            batch_indices_support_points,
            batch_indices_query_points,
            max_num_neighbors=current_k,
        )

        if neighbor_graph_edge_indices.size(1) == 0:
            return torch.empty((num_query_points, 0), dtype=torch.long, device=device)

        query_indices = neighbor_graph_edge_indices[0]
        support_indices = neighbor_graph_edge_indices[1]

        unique_indices, unique_counts = torch.unique(query_indices, return_counts=True)
        neighbor_counts = torch.zeros(num_query_points, device=device, dtype=torch.long)
        neighbor_counts.scatter_(0, unique_indices, unique_counts)
        max_neighbors = int(neighbor_counts.amax().item())

        if max_neighbors < current_k or current_k >= max_num_neighbors:
            break
        current_k *= 2

    neighbor_indices = torch.full(
        [num_query_points * max_neighbors], fill_value=invalid_neighbor_index, device=device, dtype=torch.long
    )
    mask = torch.arange(max_neighbors, device=device).unsqueeze(0).repeat(num_query_points, 1)
    mask = (mask < neighbor_counts.unsqueeze(-1)).view(-1)

    neighbor_indices[mask] = support_indices
    neighbor_indices = neighbor_indices.view(num_query_points, max_neighbors)

    if return_sorted:
        query_coords = coords_query_points[query_indices]
        support_coords = coords_support_points[support_indices]
        neighbor_distances = torch.linalg.norm(query_coords - support_coords, dim=-1)  # pylint: disable=not-callable

        neighbor_distances = torch.full(
            [num_query_points * max_neighbors], fill_value=torch.inf, device=device, dtype=query_coords.dtype
        )
        neighbor_distances[mask] = torch.linalg.norm(  # pylint: disable=not-callable
            query_coords - support_coords, dim=-1
        )
        neighbor_distances = neighbor_distances.view(num_query_points, max_neighbors)

        if k is not None and k < max_neighbors:
            max_num_neighbors = k
        else:
            max_num_neighbors = max_neighbors

        _, sorted_indices = torch.topk(neighbor_distances, k=max_num_neighbors, dim=-1, sorted=True, largest=False)
        neighbor_indices = torch.gather(neighbor_indices, -1, sorted_indices)

    return neighbor_indices
