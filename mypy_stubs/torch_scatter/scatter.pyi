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
def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = ...,
    out: Optional[torch.Tensor] = ...,
    dim_size: Optional[int] = ...,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = ...,
    out: Optional[torch.Tensor] = ...,
    dim_size: Optional[int] = ...,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = ...,
    out: Optional[torch.Tensor] = ...,
    dim_size: Optional[int] = ...,
) -> torch.Tensor: ...
