from collections import namedtuple
from typing import Literal, NamedTuple, Tuple

import torch

_knn_result = namedtuple("_knn_result", "neighbors_index neighbors_row_splits neighbors_distance")

def fixed_radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    points_row_splits: torch.Tensor,
    queries_row_splits: torch.Tensor,
    hash_table_splits: torch.Tensor,
    hash_table_index: torch.Tensor,
    hash_table_cell_splits: torch.Tensor,
    index_dtype: torch.dtype = ...,
    metric: str = ...,
    ignore_query_point: bool = ...,
    return_distances: bool = ...,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def build_spatial_hash_table(
    points: torch.Tensor,
    radius: float,
    points_row_splits: torch.Tensor,
    hash_table_size_factor: float,
    max_hash_table_size: int = ...,
) -> NamedTuple: ...
def knn_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    points_row_splits: torch.Tensor,
    queries_row_splits: torch.Tensor,
    index_dtype: int = ...,
    metric: Literal["L1", "L2"] = ...,
    ignore_query_point: bool = ...,
    return_distances: bool = ...,
) -> _knn_result: ...
