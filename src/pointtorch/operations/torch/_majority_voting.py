"""Majority voting for sets of labels."""

__all__ = ["majority_voting"]

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add


def majority_voting(labels: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
    """
    Selects the most common label among a set of labels belonging to the same batch item.

    Args:
        labels: Label for each point.
        batch_indices: Indices indicating to which input point cloud each point in the batch belongs.

    Returns:
        Most common label for each batch item.

    Shape:
        - :code:`labels`: :math:`(N)`
        - :code:`batch_indices`: :math:`(N)`
        - Output: :math:`(B)`

          | where
          |
          | :math:`B` = batch size
          | :math:`N` = number of points
    """
    min_label = labels.min()

    one_hot_labels = F.one_hot(labels - min_label)

    label_counts = scatter_add(one_hot_labels, batch_indices, dim=0)

    majority_labels = torch.argmax(label_counts, dim=-1) + min_label

    return majority_labels
