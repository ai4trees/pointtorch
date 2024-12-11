""" Neighborhood search implementation based on computing a pairwise distance matrix. """

__all__ = ["neighbor_search_cdist"]

from typing import Optional, Tuple

import torch

from ._pack_batch import pack_batch


def neighbor_search_cdist(  # pylint: disable=too-many-locals
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    *,
    radius: Optional[float] = None,
    k: Optional[int] = None,
    return_sorted: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Retrieves the neighbor points within a given search radius or the :attr:`k` nearest neighbors. The implementation of
    this function is based on PyTorch's :attr:`cdist` function. The memory consumption of this implementation is
    quadratic in the number of points, so it can only be used for small point cloud sizes. Internally,
    this function packs the point clouds into a regular batch structure by padding all point clouds to the same size, so
    that all neighbor points can be computed in parallel using PyTorch's `cdist` function.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        radius: Search radius in which to search for neighbors. Defaults to `None`, which means that an infinite search
            radius is used. Either :attr:`radius` or :attr:`k` must not be `None`.
        k: The maximum number of neighbors to search. If :attr:`radius` is not `None` and the radius neighborhood of a
            point contains more than :attr:`k` points, the :attr:`k` nearest neighbors are selected. Defaults to `None`,
            which means that all neighbors within the specified radius are searched. Either :attr:`radius` or :attr:`k`
            must not be `None`.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `False`.

    Returns:
        Tuple of two tensors. The first tensor contains the indices of the neighbors of each query point. The
        second tensor contains the distances between the neighbors and the query points. If a query point has less
        than :math:`n_{max}` neighbors, where :math:`n_{max}` is the maximum number of neighbors a query point has, the
        invalid neighbor indices in the first tensor are set to :math:`N + 1` where :math:`N` is the number of support
        points and the invalid distances in the second tensor are set to `torch.inf`.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: Tuple of two tensors, both with shape :math:`(N', n_{max})` if :attr:`k` is None or
          :math:`n_{max} <` :attr:`k`, otherwise :math:`(N', k)`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`

    Raises:
        ValueError: If both :attr:`radius` and :attr:`k` are set to `None` or the input point clouds contain too many
            points to compute a pairwise distance matrix.
    """
    if radius is None and k is None:
        raise ValueError("Either radius or k must be specified.")

    if radius is None:
        radius = float("inf")

    num_elements_dist_matrix = int((point_cloud_sizes_query_points * point_cloud_sizes_support_points).sum().item())

    if num_elements_dist_matrix > torch.iinfo(torch.int).max:
        raise ValueError(
            "The size of the distance matrix would exceed  the maximum supported size. Try reducing the "
            "number of query or support points."
        )

    invalid_neighbor_index = len(coords_support_points)

    coords_query_points_batch, mask_query_points = pack_batch(
        coords_query_points, point_cloud_sizes_query_points, fill_value=-torch.inf
    )
    coords_support_points_batch, _ = pack_batch(
        coords_support_points, point_cloud_sizes_support_points, fill_value=torch.inf
    )

    dists = torch.cdist(coords_query_points_batch, coords_support_points_batch)
    dists[dists.isnan()] = torch.inf
    is_neighbor = torch.logical_and(dists <= radius, torch.isfinite(dists))

    batch_start_index = torch.cumsum(point_cloud_sizes_support_points, dim=0) - point_cloud_sizes_support_points
    batch_start_index = batch_start_index.unsqueeze(-1).unsqueeze(-1)  # convert to shape (B, 1, 1)

    max_neighbors = int(is_neighbor.count_nonzero(dim=-1).max().item())  # maximum number of neighbors a point has
    if k is not None:
        max_neighbors = min(k, max_neighbors)

    if max_neighbors == 0:
        num_query_points = len(coords_query_points)
        neighbor_indices = torch.empty((num_query_points, 0), device=coords_query_points.device, dtype=torch.long)
        neighbor_distances = torch.empty((num_query_points, 0), device=coords_query_points.device, dtype=torch.float)
        return neighbor_indices, neighbor_distances

    neighbor_distances, neighbor_indices = torch.topk(dists, max_neighbors, dim=-1, sorted=return_sorted, largest=False)

    neighbor_indices += batch_start_index
    if radius == float("inf"):
        invalid_neighbor_mask = torch.isinf(neighbor_distances)
    else:
        invalid_neighbor_mask = neighbor_distances > radius

    neighbor_indices[invalid_neighbor_mask] = invalid_neighbor_index
    neighbor_distances[invalid_neighbor_mask] = torch.inf
    neighbor_indices = neighbor_indices[mask_query_points].view(-1, max_neighbors)
    neighbor_distances = neighbor_distances[mask_query_points].view(-1, max_neighbors)

    return neighbor_indices, neighbor_distances
