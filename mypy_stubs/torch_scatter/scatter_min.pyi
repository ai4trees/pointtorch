from typing import Optional, Tuple

import torch

def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = ...,
    out: Optional[torch.Tensor] = ...,
    dim_size: Optional[int] = ...,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
