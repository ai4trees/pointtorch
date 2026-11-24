"""Majority voting for sets of labels."""

__all__ = ["majority_voting"]

import torch
from torch_scatter import scatter_max


def majority_voting(labels: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
    """
    Selects the most common label among a set of labels belonging to the same batch item. If several labels occur with
    exactly the same frequency within a batch item, the smallest of them is selected, so that the result is
    deterministic.

    Args:
        labels: Label for each point.
        batch_indices: Indices indicating to which input point cloud each point in the batch belongs.

    Returns:
        Most common label for each batch item. If several labels occur with exactly the same frequency within a batch
        item, the smallest of them is returned.

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

    # on CUDA devices, the index of the maximum returned by scatter_max is not deterministic if several pairs of a batch
    # item have the maximum count; therefore, the pairs are ranked by a key that is unique within each batch item and
    # orders them by count and, for equal counts, prefers the smaller label; the maximum value of the key does not
    # depend on the order in which the pairs are processed
    pair_keys = counts * label_range + (label_range - 1 - pair_labels)
    max_pair_keys, _ = scatter_max(pair_keys, pair_batch_indices, dim=0, dim_size=num_batch_items)
    return label_range - 1 - max_pair_keys % label_range + min_label
