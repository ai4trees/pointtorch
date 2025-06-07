"""Transformation of input labels into consecutive integer labels."""

__all__ = ["make_labels_consecutive"]

from typing import Optional, Tuple, Union

import numpy as np

from pointtorch.type_aliases import LongArray


def make_labels_consecutive(
    labels: LongArray,
    start_id: Optional[int] = None,
    ignore_id: Optional[int] = None,
    inplace: bool = False,
    return_unique_labels: bool = False,
) -> Union[LongArray, Tuple[LongArray, LongArray]]:
    """
    Transforms the input labels into consecutive integer labels starting from a given :code:`start_id`.

    Args:
        labels: An array of original labels.
        start_id: The starting ID for the consecutive labels. If set to :code:`None`, the starting ID is set to
            :code:`ignore_id + 1` if :code:`ignore_id` is not :code:`None` and zero otherwise. Defaults to :code:`None`.
        ignore_id: A label ID that should not be changed when transforming the labels.
        inplace: Whether the transformation should be applied inplace to the :code:`labels` array. Defaults to
            :code:`False`.
        return_unique_labels: Whether the unique labels after applying the transformation (excluding :code:`ignore_id`)
            should be returned. Defaults to :code:`False`.

    Returns:
        An array with the transformed consecutive labels. If :code:`return_unique_labels` is set to :code:`True`, a
        tuple of two arrays is returned, where the second array contains the unique labels after the transformation.
    """

    if len(labels) == 0:
        if return_unique_labels:
            return labels, np.empty_like(labels)
        return labels

    if start_id is None:
        if ignore_id is not None:
            start_id = ignore_id + 1
        else:
            start_id = 0

    if not inplace:
        labels = labels.copy()

    if ignore_id is not None:
        mask = labels != ignore_id
        labels_to_remap = labels[mask]
    else:
        labels_to_remap = labels

    unique_labels = np.unique(labels_to_remap)
    key = np.arange(0, len(unique_labels), dtype=labels.dtype)
    idx = np.digitize(labels_to_remap, unique_labels, right=True)
    labels_to_remap[:] = key[idx]
    labels_to_remap += start_id

    if ignore_id is not None:
        labels[mask] = labels_to_remap
    else:
        labels[:] = labels_to_remap

    if return_unique_labels:
        return labels, key + start_id

    return labels
