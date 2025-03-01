from typing import Optional, Sequence, Union

import numpy
from numpy.random import RandomState

def train_test_split(
    *arrays: Sequence[numpy.ndarray],
    test_size: Optional[Union[float, int]] = ...,
    train_size: Optional[Union[float, int]] = ...,
    random_state: Optional[Union[int, RandomState]] = ...,
    shuffle: bool = ...,
    stratify: Optional[numpy.ndarray] = ...,
) -> list[numpy.ndarray]: ...
