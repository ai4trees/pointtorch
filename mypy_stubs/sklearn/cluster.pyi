from typing import Optional

import numpy

class MeanShift:
    def __init__(
        self,
        *,
        bandwidth: Optional[float] = ...,
        seeds: Optional[numpy.ndarray] = ...,
        bin_seeding: bool = ...,
        min_bin_freq: int = ...,
        cluster_all: bool = ...,
        n_jobs: Optional[int] = ...,
        max_iter: int = ...,
    ):
        self.cluster_centers_: numpy.ndarray
        self.labels_: numpy.ndarray
        self.n_iter_: int
        ...

    def fit(self, X: numpy.ndarray, y: None = ...) -> MeanShift: ...
