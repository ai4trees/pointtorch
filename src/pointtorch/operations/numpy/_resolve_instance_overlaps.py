"""Resolution of overlaps between segmented instances."""

__all__ = ["resolve_instance_overlaps"]

from typing import Tuple

import numpy as np

from pointtorch.type_aliases import FloatArray, LongArray


def resolve_instance_overlaps(
    instances: LongArray, instance_sizes: LongArray, scores: FloatArray
) -> Tuple[LongArray, LongArray, LongArray]:
    r"""
    Resolves overlaps between instances by assigning points that are included in multiple instances to the instance
    with the highest score.

    Args:
        instances: List of instances where each instance is represented by a set of point indices that are stored
            consecutively.
        instance_sizes: Number of points belonging to each instance.
        scores: Confidence score for each instance.

    Returns:
        Tuple of two arrays: The first represents the updated instance assignments in the same format as
            :code:`instances`. The second contains the indices indicating to which instance each index in the first
            array belongs. The third contains the number of points belonging to each updated instance.

    Shape:
        - :attr:`instances`: :math:`(N_1 + ... + N_I)`
        - :attr:`instance_sizes`: :math:`(I)`
        - Output: :math:`(N_1' + ... + N_I')`, :math:`(N_1' + ... + N_I')`, and :math:`(I')`

          | where
          |
          | :math:`I = \text{ number of instances before the filtering}`
          | :math:`I' = \text{ number of instances after the filtering}`
          | :math:`N_i = \text{ number of points belonging to the i-th instance before the filtering}`
          | :math:`N_i' = \text{ number of points belonging to the i-th instance after the filtering}`
    """

    sorted_indices = scores.argsort()[::-1]

    is_assigned = np.zeros(instances.max() + 1, dtype=bool)

    split_instances = np.split(instances, instance_sizes.cumsum()[:-1])

    new_instances = []
    new_instance_sizes = []

    for i in sorted_indices:
        instance = split_instances[i]
        instance = instance[~is_assigned[instance]]
        is_assigned[instance] = True

        if len(instance) > 0:
            new_instances.append(instance)
            new_instance_sizes.append(len(instance))

    new_instance_batch_indices = np.repeat(np.arange(len(new_instance_sizes), dtype=np.int64), new_instance_sizes)
    return np.concatenate(new_instances), new_instance_batch_indices, np.array(new_instance_sizes, dtype=np.int64)
