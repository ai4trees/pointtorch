"""Resolution of overlaps between segmented instances."""

__all__ = ["resolve_instance_overlaps"]

from typing import Tuple

import numpy as np

from pointtorch.type_aliases import FloatArray, LongArray


def resolve_instance_overlaps(
    instances: LongArray, instance_sizes: LongArray, scores: FloatArray
) -> Tuple[LongArray, LongArray, LongArray, LongArray]:
    r"""
    Resolves overlaps between instances by assigning points that are included in multiple instances to the instance
    with the highest score.

    Args:
        instances: List of instances where each instance is represented by a set of point indices that are stored
            consecutively.
        instance_sizes: Number of points belonging to each instance.
        scores: Confidence score for each instance.

    Returns: :Tuple with the following elements:
        - :code:`new_instances`: Updated instance assignments in the same format and order as :code:`instances`.
        - :code:`new_instance_batch_indices`: Indices indicating to which instance each index in :code:`new_instances`
          belongs.
        - :code:`new_instance_sizes`: Number of points belonging to each updated instance.
        - :code:`selected_indices`: Indices of the instances remaining after resolving overlaps (instances fully
          overlapping with other instances can be removed).

    Shape:
        - :attr:`instances`: :math:`(N_1 + ... + N_I)`
        - :attr:`instance_sizes`: :math:`(I)`
        - Output: :math:`(N_1' + ... + N_I')`, :math:`(N_1' + ... + N_I')`, and :math:`(I')`

          | where
          |
          | :math:`I` = number of instances before the filtering
          | :math:`I'` = number of instances after the filtering
          | :math:`N_i` = number of points belonging to the i-th instance before the filtering
          | :math:`N_i'` = number of points belonging to the i-th instance after the filtering
    """

    if len(instances) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    instance_ids_per_point = np.repeat(np.arange(len(instance_sizes)), instance_sizes)
    scores_per_point = scores[instance_ids_per_point]

    sorting_indices = np.lexsort((-scores_per_point, instances))
    point_indices_sorted = instances[sorting_indices]
    instance_ids_per_point = instance_ids_per_point[sorting_indices]

    unique_points_mask = np.ones(len(point_indices_sorted), dtype=bool)
    unique_points_mask[1:] = point_indices_sorted[1:] != point_indices_sorted[:-1]

    best_instance_id_per_point = instance_ids_per_point[unique_points_mask]

    sorting_indices = np.argsort(best_instance_id_per_point)

    new_instances = point_indices_sorted[unique_points_mask][sorting_indices]

    new_instance_sizes = np.bincount(best_instance_id_per_point, minlength=len(instance_sizes))
    selected_indices = np.nonzero(new_instance_sizes > 0)[0]
    new_instance_sizes = new_instance_sizes[selected_indices]
    new_instance_batch_indices = np.repeat(np.arange(len(new_instance_sizes), dtype=np.int64), new_instance_sizes)

    return new_instances, new_instance_batch_indices, new_instance_sizes, selected_indices
