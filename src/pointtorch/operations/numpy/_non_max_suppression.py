"""Non-maximum suppression."""

__all__ = ["non_max_suppression", "compute_pairwise_ious"]

from typing import Type

import numpy as np

from pointtorch.type_aliases import FloatArray, LongArray


def compute_pairwise_ious(
    instances: LongArray, instance_sizes: LongArray, eps: float = 1e-8, dtype: Type = np.float64
) -> FloatArray:
    r"""
    Computes pairwise intersection over union (IoU) between instances.

    Args:
        instances: List of instances where each instance is represented by a set of point indices that are stored
            consecutively.
        instance_sizes: Number of points belonging to each instance.
        dtype: Data type of the returned array. Defaults to :code:`np.float64`.

    Returns:
        Pairwise IoU scores.

    Shape:
        - :attr:`instances`: :math:`(N_1 + ... + N_I)`
        - :attr:`instance_sizes`: :math:`(I)`
        - Output: :math:`(I, I)`

          | where
          |
          | :math:`I = \text{ number of instances}`
          | :math:`N_i = \text{ number of points belonging to the i-th instance}`
    """

    if len(instance_sizes) == 0 or instance_sizes.sum() == 0:
        return np.empty((0, 0), dtype=dtype)

    split_instances = np.split(instances, np.cumsum(instance_sizes)[:-1])
    ious = np.zeros((len(instance_sizes), len(instance_sizes)), dtype=dtype)

    for idx_1, instance_proposal in enumerate(split_instances):
        for idx_2, second_instance_proposal in enumerate(split_instances):
            intersection = len(np.intersect1d(instance_proposal, second_instance_proposal))
            union = len(instance_proposal) + len(second_instance_proposal) - intersection
            ious[idx_1, idx_2] = intersection / (union + eps)

    return ious


def non_max_suppression(ious: FloatArray, scores: FloatArray, iou_threshold: float) -> LongArray:
    r"""
    Non-maximum suppression operation for instance detection.

    Args:
        ious: Pairwise intersection over union of all instance proposals.
        scores: Confidence scores for each instance proposal.
        iou_threshold: Maximum IoU that two instances can have in order to be kept as separate instances.

    Returns:
        Indices of the instances remaining after non-maximum suppression.

    Shape:
        - :attr:`ious`: :math:`(I, I)`
        - :attr:`scores`: :math:`(I)`
        - Output: :math:`(I')`

          | where
          |
          | :math:`I = \text{ number of instance proposals}`
          | :math:`I' = \text{ number of instances remaining after non-maximum suppression}'`
    """

    # sort confidence scores in descending order
    sorted_indices = scores.argsort()[::-1]
    picked_instance_indices = []

    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        picked_instance_indices.append(i)
        iou = ious[i, sorted_indices[1:]]
        remove_indices = np.where(np.logical_and(iou > iou_threshold, iou > 0))[0]
        sorted_indices = np.delete(sorted_indices, 0)
        sorted_indices = np.delete(sorted_indices, remove_indices)

    return np.array(picked_instance_indices, dtype=np.int64)
