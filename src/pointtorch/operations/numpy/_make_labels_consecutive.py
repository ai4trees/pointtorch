""" Transformation of input labels into consecutive integer labels. """

__all__ = ["make_labels_consecutive"]

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


def make_labels_consecutive(
    labels: npt.NDArray[np.int64],
    start_id: int = 0,
    ignore_id: Optional[int] = None,
    inplace: bool = False,
    return_unique_labels: bool = False,
) -> Union[npt.NDArray[np.int64], Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
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
        An array with the transformed consecutive labels. If :code:`return_unique_labels` is set to :code:`True`, a
        tuple of two arrays is returned, where the second array contains the unique labels after the transformation.
    """

    if len(labels) == 0:
        if return_unique_labels:
            return labels, np.empty_like(labels)
        return labels

    if not inplace:
        labels = labels.copy()

    if ignore_id is not None:
        mask = labels != ignore_id
        labels_to_remap = labels[mask]
    else:
        labels_to_remap = labels

    unique_labels = np.unique(labels_to_remap)
    unique_labels = np.sort(unique_labels)
    key = np.arange(0, len(unique_labels))
    index = np.digitize(labels_to_remap, unique_labels, right=True)
    labels_to_remap[:] = key[index]
    labels_to_remap += start_id

    if ignore_id is not None:
        labels[mask] = labels_to_remap
    else:
        labels[:] = labels_to_remap

    if return_unique_labels:
        return labels, key + start_id

    return labels
