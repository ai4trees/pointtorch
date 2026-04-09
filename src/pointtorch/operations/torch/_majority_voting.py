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
    labels = labels - min_label
    num_classes = int(labels.max().item()) + 1
    num_batch_items = int(batch_indices.max().item()) + 1

    label_counts = torch.zeros((num_batch_items, num_classes), device=labels.device, dtype=torch.long)
    label_counts.index_put_((batch_indices, labels), torch.ones_like(labels, dtype=torch.long), accumulate=True)

    majority_labels = label_counts.argmax(dim=-1) + min_label
    return majority_labels
