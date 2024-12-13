from typing import Optional, Tuple

import torch

def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = ...,
    out: Optional[torch.Tensor] = ...,
    dim_size: Optional[int] = ...,
    reduce: str = ...,
) -> torch.Tensor: ...
