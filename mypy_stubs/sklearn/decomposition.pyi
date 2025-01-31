from typing import Literal, Optional, Union

import numpy

class PCA:
    def __init__(
        self,
        n_components: Optional[Union[int, float, Literal["mle"]]] = ...,
        copy: bool = ...,
        whiten: bool = ...,
        svd_solver: Literal["auto", "full", "arpack", "randomized"] = ...,
        tol: float = ...,
        iterated_power: Union[int, Literal["auto"]] = ...,
        n_oversamples: int = ...,
        power_iteration_normalizer: Literal["auto", "QR", "LU", "none"] = ...,
        random_state: Optional[Union[int, numpy.random.RandomState]] = ...,
    ):
        self.components_: numpy.ndarray

    def fit(self, x: numpy.ndarray): ...
