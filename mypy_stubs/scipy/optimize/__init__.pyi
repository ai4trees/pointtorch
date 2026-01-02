from typing import Tuple

import numpy as np
import numpy.typing as npt

def linear_sum_assignment(
    cost_matrix: npt.ArrayLike, maximize: bool = ...
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
