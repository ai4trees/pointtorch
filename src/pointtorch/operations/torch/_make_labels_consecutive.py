""" Transformation of input labels into consecutive integer labels. """

__all__ = ["make_labels_consecutive"]

import torch


def make_labels_consecutive(labels: torch.Tensor, start_id: int = 0) -> torch.Tensor:
    """
    Transforms the input labels into consecutive integer labels starting from a given :code:`start_id`.

    Args:
        labels: An array of original labels.
        start_id: The starting ID for the consecutive labels. Defaults to zero.

    Returns:
        numpy.ndarray: An array with the transformed consecutive labels.
    """

    unique_labels = torch.unique(labels)
    unique_labels = torch.sort(unique_labels)[0]
    key = torch.arange(0, len(unique_labels), device=labels.device)
    index = torch.bucketize(labels, unique_labels, right=False)
    labels = key[index]
    labels = labels + start_id
    return labels
