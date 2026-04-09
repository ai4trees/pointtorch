"""Majority voting for sets of labels."""

__all__ = ["majority_voting"]

import torch
from torch_scatter import scatter_max


def majority_voting(labels: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
    """
    Selects the most common label among a set of labels belonging to the same batch item.

    Args:
        labels: Label for each point.
        batch_indices: Indices indicating to which input point cloud each point in the batch belongs.

    Returns:
        Most common label for each batch item. If several labels occur with the exactly the same frequencey within a
        batch item, any of them may be returned.

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
    num_batch_items = int(batch_indices.max().item()) + 1
    label_range = int(labels.max().item()) + 1

    pair_ids = batch_indices * label_range + labels
    unique_pair_ids, counts = torch.unique(pair_ids, sorted=True, return_counts=True)

    pair_batch_indices = unique_pair_ids // label_range
    pair_labels = unique_pair_ids % label_range

    _, majority_pair_indices = scatter_max(counts, pair_batch_indices, dim=0, dim_size=num_batch_items)
    return pair_labels[majority_pair_indices] + min_label
