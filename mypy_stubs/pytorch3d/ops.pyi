from collections import namedtuple
from typing import Union

import torch

_KNN = namedtuple("_KNN", "dists idx knn")

def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = ...,
    lengths2: Union[torch.Tensor, None] = ...,
    K: int = ...,
    radius: float = ...,
    return_nn: bool = ...,
): ...
def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = ...,
    lengths2: Union[torch.Tensor, None] = ...,
    norm: int = ...,
    K: int = ...,
    version: int = ...,
    return_nn: bool = ...,
    return_sorted: bool = ...,
) -> _KNN: ...
