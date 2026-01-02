from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

class KDTree:
    def __init__(
        self,
        data: npt.ArrayLike,
        leafsize: int = ...,
        compact_nodes: bool = ...,
        copy_data: bool = ...,
        balanced_tree: bool = ...,
        boxsize: Optional[Union[npt.ArrayLike, float]] = ...,
    ): ...
    def query(
        self,
        x: npt.ArrayLike,
        k: Optional[int] = ...,
        eps: float = ...,
        p: int = ...,
        distance_upper_bound: float = ...,
        workers: int = 1,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.int64]]: ...
    def query_ball_point(
        self,
        x: npt.ArrayLike,
        r: float,
        p: float = ...,
        eps: int = ...,
        workers: int = ...,
        return_sorted: bool = ...,
        return_length: bool = ...,
    ): ...
