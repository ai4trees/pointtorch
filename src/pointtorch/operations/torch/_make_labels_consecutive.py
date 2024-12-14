""" Transformation of input labels into consecutive integer labels. """

__all__ = ["make_labels_consecutive"]

from typing import Optional, Tuple, Union

import torch


def make_labels_consecutive(
    labels: torch.Tensor,
    start_id: int = 0,
    ignore_id: Optional[int] = None,
    inplace: bool = False,
    return_unique_labels: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Transforms the input labels into consecutive integer labels starting from a given :code:`start_id`.

    Args:
        labels: An array of original labels.
        start_id: The starting ID for the consecutive labels. Defaults to zero.
        ignore_id: A label ID that should not be changed when transforming the labels.
        inplace: Whether the transformation should be applied inplace to the :code:`labels` array. Defaults to
            :code:`False`.
        return_unique_labels: Whether the unique labels after applying the transformation (excluding :code:`ignore_id`)
            should be returned. Defaults to :code:`False`.

    Returns:
        A tensor with the transformed consecutive labels. If :code:`return_unique_labels` is set to :code:`True`, a
        tuple of two tensors is returned, where the second tensor contains the unique labels after the transformation.
    """

    if len(labels) == 0:
        if return_unique_labels:
            return labels, torch.empty_like(labels)
        return labels

    if not inplace:
        labels = labels.clone()

    if ignore_id is not None:
        mask = labels != ignore_id
        labels_to_remap = labels[mask]
    else:
        labels_to_remap = labels

    unique_labels = torch.unique(labels_to_remap)
    unique_labels = torch.sort(unique_labels)[0]
    key = torch.arange(0, len(unique_labels), device=labels.device)
    index = torch.bucketize(labels_to_remap, unique_labels, right=False)
    labels_to_remap = key[index]
    labels_to_remap += start_id

    if ignore_id is not None:
        labels[mask] = labels_to_remap
    else:
        labels[:] = labels_to_remap

    if return_unique_labels:
        return labels, key + start_id

    return labels
