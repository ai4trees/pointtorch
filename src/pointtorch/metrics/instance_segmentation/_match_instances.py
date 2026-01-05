"""Matching of target and predicted instances."""

__all__ = ["match_instances", "match_instances_iou", "match_instances_point2tree", "match_instances_tree_learn"]

from typing import Dict, Literal, Optional, Tuple

from scipy.optimize import linear_sum_assignment
import torch
from torch_scatter import scatter_max, scatter_min


def match_instances(  # pylint: disable=too-many-locals, too-many-statements, too-many-return-statements, too-many-branches
    target: torch.Tensor,
    prediction: torch.Tensor,
    xyz: Optional[torch.Tensor] = None,
    invalid_instance_id: int = -1,
    method: Literal[
        "panoptic_segmentation",
        "point2tree",
        "for_instance",
        "for_ai_net",
        "for_ai_net_coverage",
        "tree_learn",
    ] = "panoptic_segmentation",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    This method implements the instance matching methods proposed in the following works:

    - :code:`panoptic_segmentation`: `Kirillov, Alexander, et al. "Panoptic segmentation." Proceedings of the IEEE/CVF \
      Conference on Computer Vision and Pattern Recognition. 2019. <https://doi.org/10.1109/CVPR.2019.00963>`__
      
      This method matches predicted and target instances if their IoU is striclty greater than 0.5, which results
      in an unambigous matching. This method is also used in `Wielgosz, Maciej, et al. "SegmentAnyTree: A Sensor and \
      Platform Agnostic Deep Learning Model for Tree Segmentation Using Laser Scanning Data." Remote Sensing of \
      Environment 313 (2024): 114367. <https://doi.org/10.1016/j.rse.2024.114367>`__

    - :code:`point2tree`: `Wielgosz, Maciej, et al. "Point2Tree (P2T)—Framework for Parameter Tuning of Semantic and \
      Instance Segmentation Used with Mobile Laser Scanning Data in Coniferous Forest." Remote Sensing 15.15 (2023): \
      3737. <https://doi.org/10.3390/rs15153737>`__

      This method processes the target instances sorted according to their height. Starting with the highest target
      instance, each target instance is matched with the predicted instance with which it has the highest IoU. Predicted
      instances that were already matched to a target instance before, are excluded from the matching.

    - :code:`for_instance`: `Puliti, Stefano, et al. "For-Instance: a UAV Laser Scanning Benchmark Dataset for \
      Semantic and Instance Segmentation of Individual Trees." arXiv preprint arXiv:2309.01279 (2023). \
      <https://arxiv.org/pdf/2309.01279>`__

      This method is based on the method proposed by Wielgosz et al. (2023) but additionally introduces the criterion
      that target and predicted instances must have an IoU of a least 0.5 to be matched.

    - :code:`for_ai_net`: `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density Airborne LiDAR \
      Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078. \
      <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method is similar to the method proposed by Kirillov et al., with the difference that target and predicted
      instances are also matched if their IoU is equal to 0.5.

    - :code:`for_ai_net_coverage`: `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density \
      Airborne LiDAR Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078. \
      <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method matches each target instance with the predicted instance with which it has the highest IoU. This
      means that predicted instances can be matched with multiple target instances. Such a matching approach is useful
      for the calculation of segmentation metrics (e.g., coverage) that should be independent from the instance
      detection rate.
  
    - :code:`tree_learn`: `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for Segmenting Individual Trees
      from Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888.
      <https://doi.org/10.1016/j.ecoinf.2024.102888>`__

      This method uses Hungarian matching to match predicted and target instances in such a way that the sum of the IoU
      scores of all matched instance pairs is maximized. Subsequently, matches with an IoU score less than o equal to
      0.5 are discarded.

    Args:
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        xyz: Coordinates of all points. Must be provided if method is set to :code:`"point2tree"` or
            :code:`"for_instance"`. Defaults to :code:`None`.
        method: Instance matching method to be used. Defaults to :code:`"panoptic_segmentation"`.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Defaults to `-1`.

    Returns: A tuple with the following elements:
        - :code:`matched_target_ids`: IDs of the matched target instance for each predicted instance. Predicted
          instances that are not matched to a target instance are assigned :code:`invalid_tree_id`.
        - :code:`matched_predicted_ids`: IDs of the matched predicted instance for each target instance. Target
          instances that are not matched to a predicted instance are assigned :code:`invalid_tree_id`.
        - :code:`metrics`: Dictionary with the keys :code:`"tp"`, :code:`"fp"`, :code:`"fn"`. The values are
          tensors whose length is equal to the number of target instances and that contain the number of true positive,
          false positive, and false negative points between the matched instances. For target instances not matched to
          any prediction, the true and false posiitves are set to zero and the false negatives to the number of target
          points.

    Raises:
        ValueError: If :code:`target` and :code:`prediction` don't have the same length.
        ValueError: If the unique instance IDs are not consecutive.
        ValueError: If the unique target and predicted instance IDs don't start with the same number.
        ValueError: If :code:`invalid_instance_id` is not smaller than the first valid instance ID.
        ValueError: If :code:`xyz` is :code:`None` and :code:`method` is :code:`"point2tree"` or :code:`"for_instance"`.
        ValueError: If :code:`method` is set to an invalid value.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - Output:
            - :code:`matched_target_ids`: :math:`(P)`
            - :code:`matched_predicted_ids`: :math:`(T)`
            - :code:`metrics`: Dictionary whose values are tensors of length :math:`(T)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of predicted instances
        | :math:`T` = number of target instances
    """

    device = target.device

    if target.numel() != prediction.numel():
        raise ValueError("Target and prediction must have the same length.")

    unique_target_ids, target_sizes = torch.unique(target, return_counts=True)
    valid_mask = unique_target_ids != invalid_instance_id
    unique_target_ids = unique_target_ids[valid_mask]
    target_sizes = target_sizes[valid_mask]

    unique_prediction_ids = torch.unique(prediction)
    unique_prediction_ids = unique_prediction_ids[unique_prediction_ids != invalid_instance_id]

    num_target_instances = len(unique_target_ids)
    num_predicted_instances = len(unique_prediction_ids)

    if num_predicted_instances == 0 or num_target_instances == 0:
        return _initialize_matching_results(
            num_target_instances, num_predicted_instances, target_sizes, invalid_instance_id, device
        )

    start_instance_id_target = unique_target_ids.min()
    start_instance_id_prediction = unique_prediction_ids.min()

    if start_instance_id_target != start_instance_id_prediction:
        raise ValueError("Start instance IDs for target and prediction must be identical.")

    if unique_target_ids.max() - start_instance_id_target != len(unique_target_ids) - 1:
        raise ValueError("The target instance IDs must be consecutive.")

    if unique_prediction_ids.max() - start_instance_id_prediction != len(unique_prediction_ids) - 1:
        raise ValueError("The predicted instance IDs must be consecutive.")

    if invalid_instance_id >= start_instance_id_target:
        raise ValueError("Invalid instance ID must be smaller than the first valid instance ID.")

    if xyz is None and method in ["point2tree", "for_instance"]:
        raise ValueError(f"xyz must be provided for method '{method}'.")

    if method == "panoptic_segmentation":
        return match_instances_iou(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id_target,
            invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=False,
        )
    if method == "for_ai_net":
        return match_instances_iou(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id_target,
            invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=True,
        )
    if method == "for_ai_net_coverage":
        return match_instances_iou(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id_target,
            invalid_instance_id,
            min_iou_treshold=None,
            accept_equal_iou=True,
        )
    if method == "point2tree":
        return match_instances_point2tree(
            xyz,  # type: ignore[arg-type]
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id_target,
            invalid_instance_id,
            min_iou_treshold=None,
            accept_equal_iou=True,
            sort_by_target_height=True,
        )
    if method == "for_instance":
        return match_instances_point2tree(
            xyz,  # type: ignore[arg-type]
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id_target,
            invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=True,
            sort_by_target_height=True,
        )
    if method == "tree_learn":
        return match_instances_tree_learn(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id_target,
            invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=False,
        )

    raise ValueError(f"Invalid matching method: {method}.")


def match_instances_iou(  # pylint: disable=too-many-statements, too-many-locals, too-many-positional-arguments
    target: torch.Tensor,
    unique_target_ids: torch.Tensor,
    prediction: torch.Tensor,
    unique_prediction_ids: torch.Tensor,
    start_instance_id: torch.Tensor,
    invalid_instance_id: int,
    min_iou_treshold: Optional[float] = 0.5,
    accept_equal_iou: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    This method matches each target instance with the predicted instance with which it has the highest Intersection over
    Union (IoU). If :code:`min_iou_treshold` is not :code:`None`, instances are only matched if their IoU is greater
    than this threshold.

    This method implements the instance matching methods proposed in the following works:

    - `Kirillov, Alexander, et al. "Panoptic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision \
      and Pattern Recognition. 2019. <https://doi.org/10.1109/CVPR.2019.00963>`__
      
      This method matches predicted and target instances if their IoU is striclty greater than 0.5, which results
      in an unambigous matching.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = 0.5
      - :code:`accept_equal_iou` = :code:`False`

    - `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density Airborne LiDAR Point Clouds with 3D
      Deep Learning." Remote Sensing of Environment 305 (2024): 114078. <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method is similar to the method proposed by Kirillov et al., with the difference that target and
      predicted instances are also matched if their IoU is equal to 0.5.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = 0.5
      - :code:`accept_equal_iou` = :code:`True`

    - Matching approach for computing the coverage metric in `Xiang, Binbin, et al. "Automated Forest Inventory: \
      Analysis of High-Density Airborne LiDAR Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 \
      (2024): 114078. <https://doi.org/10.1016/j.rse.2024.114078>`__.

      This method is similar to the method proposed by Kirillov et al., with the difference that target and predicted
      instances are also matched if their IoU is equal to 0.5.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = :code:`None`
      - :code:`accept_equal_iou` = :code:`True`

    Args:
        target: Ground truth instance ID for each point.
        unique_target_ids: Unique target instance IDs excluding :code:`invalid_instance_id`.
        prediction: Predicted instance ID for each point.
        unique_prediction_ids: Unique predicted instance IDs excluding :code:`invalid_instance_id`.
        start_instance_id: Smallest valid instance ID. All instance IDs are expected to be consecutive.
        invalid_instance_id: ID that is assigned to points not assigned to any instance.
        min_iou_treshold: IoU threshold for instance matching. If set to a value that is not :code:`None`,
            instances are only matched if their IoU is equal to (only if :code:`accept_equal_iou` is :code:`True`) or
            stricly greater than this threshold.
        accept_equal_iou: Whether matched pairs of instances should be accepted if their IoU is equal to
            :code:`min_iou_treshold`.

    Returns: A tuple with the following elements:
        - :code:`matched_target_ids`: IDs of the matched target instance for each predicted instance. Predicted
          instances that are not matched to a target instance are assigned :code:`invalid_tree_id`.
        - :code:`matched_predicted_ids`: IDs of the matched predicted instance for each target instance. Target
          instances that are not matched to a predicted instance are assigned :code:`invalid_tree_id`.
        - :code:`metrics`: Dictionary with the keys :code:`"tp"`, :code:`"fp"`, :code:`"fn"`. The values are
          tensors whose length is equal to the number of target instances and that contain the number of true positive,
          false positive, and false negative points between the matched instances. For target instances not matched to
          any prediction, the true and false posiitves are set to zero and the false negatives to the number of target
          points.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`unique_target_ids`: math:`(T)`
        - :code:`prediction`: :math:`(N)`
        - :code:`unique_prediction_ids`: math:`(P)`
        - Output:
            - :code:`matched_target_ids`: :math:`(P)`
            - :code:`matched_predicted_ids`: :math:`(T)`
            - :code:`metrics`: Dictionary whose values are tensors of length :math:`(T)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of predicted instances
        | :math:`T` = number of target instances
    """

    if min_iou_treshold is None:
        min_iou_treshold = -1

    device = target.device

    # remap instances IDs so that the first valid instance has ID 0 and the invalid instance ID is set to -1
    invalid_target_mask = target == invalid_instance_id
    target = target - start_instance_id
    target[invalid_target_mask] = -1

    invalid_prediction_mask = prediction == invalid_instance_id
    prediction = prediction - start_instance_id
    prediction[invalid_prediction_mask] = -1

    unique_target_ids = unique_target_ids - start_instance_id
    unique_prediction_ids = unique_prediction_ids - start_instance_id

    num_target_instances = len(unique_target_ids)
    num_predicted_instances = len(unique_prediction_ids)

    matching_candidates = _get_matching_candidates(target, prediction, num_predicted_instances, -1)

    target_sizes, predicted_sizes = _get_instance_sizes(
        target, prediction, num_target_instances, num_predicted_instances, -1
    )

    if matching_candidates is None:
        return _initialize_matching_results(
            num_target_instances, num_predicted_instances, target_sizes, invalid_instance_id, device
        )

    paired_target_ids, paired_predicted_ids, pair_counts = matching_candidates

    matched_target_ids, matched_predicted_ids, metrics = _initialize_matching_results(
        num_target_instances, num_predicted_instances, target_sizes, -1, device
    )

    target_ids_to_match, target_batch_indices = torch.unique_consecutive(paired_target_ids, return_inverse=True)

    tp, best_predicted_ids = scatter_max(pair_counts, target_batch_indices, dim=0)
    predicted_ids_to_match = paired_predicted_ids[best_predicted_ids]

    has_overlap = tp > 0
    tp = tp[has_overlap]
    target_ids_to_match = target_ids_to_match[has_overlap]
    predicted_ids_to_match = predicted_ids_to_match[has_overlap]

    target_sizes = target_sizes[target_ids_to_match]
    predicted_sizes = predicted_sizes[predicted_ids_to_match]

    fp = predicted_sizes - tp
    fn = target_sizes - tp

    iou = tp.to(torch.float) / (tp.to(torch.float) + fp + fn)

    if accept_equal_iou:
        matching_mask = iou >= min_iou_treshold
    else:
        matching_mask = iou > min_iou_treshold

    iou = iou[matching_mask]
    tp = tp[matching_mask]
    fp = fp[matching_mask]
    fn = fn[matching_mask]

    matched_target_indices = target_ids_to_match[matching_mask]
    matched_predicted_indices = predicted_ids_to_match[matching_mask]

    matched_predicted_ids[matched_target_indices] = matched_predicted_indices

    # in this case, each predicted instance is matched to only one target
    if (not accept_equal_iou and min_iou_treshold >= 0.5) or min_iou_treshold > 0.5:
        matched_target_ids[matched_predicted_indices] = matched_target_indices
    else:
        unique_matched_predicted_indices, batch_indices_prediction = torch.unique(
            matched_predicted_indices, return_inverse=True, sorted=True
        )
        _, best_target_indices = scatter_max(iou, batch_indices_prediction, dim=0)

        matched_target_ids[unique_matched_predicted_indices] = matched_target_indices[best_target_indices]

    metrics["tp"][matched_target_indices] = tp
    metrics["fp"][matched_target_indices] = fp
    metrics["fn"][matched_target_indices] = fn

    invalid_matches_mask = matched_target_ids == -1
    matched_target_ids = matched_target_ids + start_instance_id
    matched_target_ids[invalid_matches_mask] = invalid_instance_id

    invalid_matches_mask = matched_predicted_ids == -1
    matched_predicted_ids = matched_predicted_ids + start_instance_id
    matched_predicted_ids[invalid_matches_mask] = invalid_instance_id

    return matched_target_ids, matched_predicted_ids, metrics


def match_instances_point2tree(  # pylint: disable=too-many-statements, too-many-locals, too-many-positional-arguments
    xyz: torch.Tensor,
    target: torch.Tensor,
    unique_target_ids: torch.Tensor,
    prediction: torch.Tensor,
    unique_prediction_ids: torch.Tensor,
    start_instance_id: int,
    invalid_instance_id: int,
    min_iou_treshold: Optional[float] = 0.5,
    accept_equal_iou: bool = False,
    sort_by_target_height: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    This method sorts target instances either by their height (if :code:`sort_by_target_height` is :code:`True`) or by
    their maximum IoU with a predicted instance. The target instances are processed in the sorting order and each
    target instance is matched with the predicted instance with which it has the highest IoU. Predicted instances that
    were already matched to a target instance before, are excluded from the matching, so that predicted instances cannot
    be matched by multiple target instances.

    Then each target is matched with the predicted instance with which 

    This method implements the instance matching methods proposed in the following works:

    - `Wielgosz, Maciej, et al. "Point2Tree (P2T)—Framework for Parameter Tuning of Semantic and Instance Segmentation \
      Used with Mobile Laser Scanning Data in Coniferous Forest." Remote Sensing 15.15 (2023): 3737. \
      <https://doi.org/10.3390/rs15153737>`__

      This method processes the target instances sorted according to their height.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = :code:`None`
      - :code:`sort_by_target_height` = :code:`True`

    - `Puliti, Stefano, et al. "For-Instance: a UAV Laser Scanning Benchmark Dataset for Semantic and Instance
      Segmentation of Individual Trees." arXiv preprint arXiv:2309.01279 (2023). <https://arxiv.org/pdf/2309.01279>`__

      This method is based on the method proposed by Wielgosz et al. (2023) but additionally introduces the criterion
      that target and predicted instances must have an IoU of a least 0.5 to be matched.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = 0.5
      - :code:`accept_equal_iou` = :code:`True`
      - :code:`sort_by_target_height` = :code:`True`

    Args:
        xyz: Coordinates of all points.
        target: Ground truth instance ID for each point.
        unique_target_ids: Unique target instance IDs excluding :code:`invalid_instance_id`.
        prediction: Predicted instance ID for each point.
        unique_prediction_ids: Unique predicted instance IDs excluding :code:`invalid_instance_id`.
        start_instance_id: Smallest valid instance ID. All instance IDs are expected to be consecutive.
        invalid_instance_id: ID that is assigned to points not assigned to any instance.
        min_iou_treshold: IoU threshold for instance matching. If set to a value that is not :code:`None`,
            instances are only matched if their IoU is equal to (only if :code:`accept_equal_iou` is :code:`True`) or
            stricly greater than this threshold.
        accept_equal_iou: Whether matched pairs of instances should be accepted if their IoU is equal to
            :code:`min_iou_treshold`.
        sort_by_target_height: Whether the instance matching should process the target instances sorted according to
            their height. This corresponds to the matching approach proposed by Wielgosz et al. The processing order of
            the target instances is only relevant if the matching can be ambiguous, i.e. if matches with an IoU <= 0.5
            are accepted.

    Returns: A tuple with the following elements:
        - :code:`matched_target_ids`: IDs of the matched target instance for each predicted instance. Predicted
          instances that are not matched to a target instance are assigned :code:`invalid_tree_id`.
        - :code:`matched_predicted_ids`: IDs of the matched predicted instance for each target instance. Target
          instances that are not matched to a predicted instance are assigned :code:`invalid_tree_id`.
        - :code:`metrics`: Dictionary with the keys :code:`"tp"`, :code:`"fp"`, :code:`"fn"`. The values are
          tensors whose length is equal to the number of target instances and that contain the number of true positive,
          false positive, and false negative points between the matched instances. For target instances not matched to
          any prediction, the true and false posiitves are set to zero and the false negatives to the number of target
          points.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`unique_target_ids`: math:`(T)`
        - :code:`prediction`: :math:`(N)`
        - :code:`unique_prediction_ids`: math:`(P)`
        - Output:
            - :code:`matched_target_ids`: :math:`(P)`
            - :code:`matched_predicted_ids`: :math:`(T)`
            - :code:`metrics`: Dictionary whose values are tensors of length :math:`(T)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of predicted instances
        | :math:`T` = number of target instances
    """

    if min_iou_treshold is None:
        min_iou_treshold = -1

    device = target.device

    invalid_target_mask = target == invalid_instance_id
    target = target - start_instance_id
    target[invalid_target_mask] = -1

    invalid_prediction_mask = prediction == invalid_instance_id
    prediction = prediction - start_instance_id
    prediction[invalid_prediction_mask] = -1

    unique_target_ids = unique_target_ids - start_instance_id
    unique_prediction_ids = unique_prediction_ids - start_instance_id

    num_target_instances = len(unique_target_ids)
    num_predicted_instances = len(unique_prediction_ids)

    matching_candidates = _get_matching_candidates(target, prediction, num_predicted_instances, -1)

    target_sizes, predicted_sizes = _get_instance_sizes(
        target, prediction, num_target_instances, num_predicted_instances, -1
    )

    if matching_candidates is None:
        return _initialize_matching_results(
            num_target_instances, num_predicted_instances, target_sizes, invalid_instance_id, device
        )

    paired_target_ids, paired_predicted_ids, pair_counts = matching_candidates

    matched_target_ids, matched_predicted_ids, metrics = _initialize_matching_results(
        num_target_instances, num_predicted_instances, target_sizes, -1, device
    )

    # Find segment starts/ends for each target that appears
    target_ids_to_match, target_batch_indices, prediction_counts_per_target = torch.unique_consecutive(
        paired_target_ids, return_inverse=True, return_counts=True
    )
    starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.long), prediction_counts_per_target[:-1]]), dim=0
    )
    ends = starts + prediction_counts_per_target

    remap_target_indices = torch.full((num_target_instances,), -1, device=device, dtype=torch.long)
    remap_target_indices[target_ids_to_match] = torch.arange(len(target_ids_to_match), device=device, dtype=torch.long)

    valid_target_mask = target != -1
    valid_target = target[valid_target_mask]

    true_positives = pair_counts
    ious = true_positives.to(torch.float) / (
        target_sizes[paired_target_ids] + predicted_sizes[paired_predicted_ids] - true_positives.to(torch.float)
    )

    if sort_by_target_height:
        z = xyz[valid_target_mask, 2]
        instance_heights = (
            scatter_max(z, valid_target, dim=0, dim_size=num_target_instances)[0]
            - scatter_min(z, valid_target, dim=0, dim_size=num_target_instances)[0]
        )
        sorting_indices = torch.argsort(-instance_heights)
    else:
        max_iou_per_target = torch.zeros(num_target_instances, device=device, dtype=torch.float)
        max_iou_per_target[target_ids_to_match] = scatter_max(
            ious, target_batch_indices, dim=0, dim_size=len(target_ids_to_match)
        )[0]

        sorting_indices = torch.argsort(-max_iou_per_target)

    sorted_unique_target_ids = unique_target_ids[sorting_indices]

    restrict_by_availability = (accept_equal_iou and min_iou_treshold <= 0.5) or (min_iou_treshold < 0.5)
    accept = (lambda x: x >= min_iou_treshold) if accept_equal_iou else (lambda x: x > min_iou_treshold)

    for target_id in sorted_unique_target_ids.tolist():
        remap_idx = int(remap_target_indices[target_id].item())
        if remap_idx < 0:
            continue  # no overlapping predicted ids

        start = int(starts[remap_idx].item())
        end = int(ends[remap_idx].item())

        candidate_predictions = paired_predicted_ids[start:end]
        candidate_tp = true_positives[start:end]
        candidate_ious = ious[start:end]

        if restrict_by_availability:
            is_available = matched_target_ids[candidate_predictions] == -1
            if not torch.any(is_available):
                continue
            candidate_predictions = candidate_predictions[is_available]
            candidate_ious = candidate_ious[is_available]

        best_prediction_idx = torch.argmax(candidate_ious)
        predicted_id = int(candidate_predictions[best_prediction_idx].item())
        tp = candidate_tp[best_prediction_idx]
        iou = candidate_ious[best_prediction_idx]

        if accept(iou):
            matched_target_ids[predicted_id] = target_id
            matched_predicted_ids[target_id] = predicted_id
            metrics["tp"][target_id] = tp
            metrics["fp"][target_id] = predicted_sizes[predicted_id] - tp
            metrics["fn"][target_id] = target_sizes[target_id] - tp

    invalid_matches_mask = matched_target_ids == -1
    matched_target_ids = matched_target_ids + start_instance_id
    matched_target_ids[invalid_matches_mask] = invalid_instance_id

    invalid_matches_mask = matched_predicted_ids == -1
    matched_predicted_ids = matched_predicted_ids + start_instance_id
    matched_predicted_ids[invalid_matches_mask] = invalid_instance_id

    return matched_target_ids, matched_predicted_ids, metrics


def match_instances_tree_learn(  # pylint: disable=too-many-statements, too-many-locals, too-many-positional-arguments
    target: torch.Tensor,
    unique_target_ids: torch.Tensor,
    prediction: torch.Tensor,
    unique_prediction_ids: torch.Tensor,
    start_instance_id: int,
    invalid_instance_id: int,
    min_iou_treshold: Optional[float] = 0.5,
    accept_equal_iou: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    Instance matching method that is proposed in `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for
    Segmenting Individual Trees from Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888.
    <https://doi.org/10.1016/j.ecoinf.2024.102888>`__ The method uses Hungarian matching to match predicted and ground
    truth instances in such a way that the sum of the IoU scores of all matched instance pairs is maximized.
    Subsequently, matches with an IoU score less than or equal to :code:`min_iou_treshold` are discarded.

    Args:
        target: Target instance ID for each point.
        unique_target_ids: Unique target instance IDs excluding :code:`invalid_instance_id`.
        prediction: Predicted instance ID for each point.
        unique_prediction_ids: Unique predicted instance IDs excluding :code:`invalid_instance_id`.
        start_instance_id: Smallest valid instance ID. All instance IDs are expected to be consecutive.
        invalid_instance_id: ID that is assigned to points not assigned to any instance.
        min_iou_treshold: IoU threshold for instance matching. If set to a value that is not :code:`None`,
            instances are only matched if their IoU is strictly greater than this threshold. Setting it to :code:`0.5`,
            corresponds to the setting proposed by Henrich et al.
        accept_equal_iou: Whether matched pairs of instances should be accepted if their IoU is equal to
            :code:`min_iou_treshold`.

    Returns: A tuple with the following elements:
        - :code:`matched_target_ids`: IDs of the matched target instance for each predicted instance. Predicted
          instances that are not matched to a target instance are assigned :code:`invalid_tree_id`.
        - :code:`matched_predicted_ids`: IDs of the matched predicted instance for each target instance. Target
          instances that are not matched to a predicted instance are assigned :code:`invalid_tree_id`.
        - :code:`metrics`: Dictionary with the keys :code:`"tp"`, :code:`"fp"`, :code:`"fn"`. The values are
          tensors whose length is equal to the number of target instances and that contain the number of true positive,
          false positive, and false negative points between the matched instances. For target instances not matched to
          any prediction, the true and false posiitves are set to zero and the false negatives to the number of target
          points.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`unique_target_ids`: math:`(G)`
        - :code:`prediction`: :math:`(N)`
        - :code:`unique_prediction_ids`: math:`(P)`
        - Output:
            - :code:`matched_target_ids`: :math:`(P)`
            - :code:`matched_predicted_ids`: :math:`(T)`
            - :code:`metrics`: Dictionary whose values are tensors of length :math:`(T)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of predicted instances
        | :math:`T` = number of target instances
    """

    if min_iou_treshold is None:
        min_iou_treshold = -1

    device = target.device

    invalid_target_mask = target == invalid_instance_id
    target = target - start_instance_id
    target[invalid_target_mask] = -1

    invalid_prediction_mask = prediction == invalid_instance_id
    prediction = prediction - start_instance_id
    prediction[invalid_prediction_mask] = -1

    unique_target_ids = unique_target_ids - start_instance_id
    unique_prediction_ids = unique_prediction_ids - start_instance_id

    num_target_instances = len(unique_target_ids)
    num_predicted_instances = len(unique_prediction_ids)

    matching_candidates = _get_matching_candidates(target, prediction, num_predicted_instances, -1)

    target_sizes, predicted_sizes = _get_instance_sizes(
        target, prediction, num_target_instances, num_predicted_instances, -1
    )

    if matching_candidates is None:
        return _initialize_matching_results(
            num_target_instances, num_predicted_instances, target_sizes, invalid_instance_id, device
        )

    matched_target_ids, matched_predicted_ids, metrics = _initialize_matching_results(
        num_target_instances, num_predicted_instances, target_sizes, -1, device
    )

    paired_target_ids, paired_predicted_ids, pair_counts = matching_candidates

    iou_matrix = torch.zeros((num_predicted_instances, num_target_instances), dtype=torch.float, device=device)
    tp_matrix = torch.zeros((num_predicted_instances, num_target_instances), dtype=torch.long, device=device)

    tp = pair_counts.to(torch.float)
    union = predicted_sizes[paired_predicted_ids] + target_sizes[paired_target_ids] - tp

    iou_matrix[paired_predicted_ids, paired_target_ids] = tp / union
    tp_matrix[paired_predicted_ids, paired_target_ids] = pair_counts

    iou_matrix_np = iou_matrix.detach().to("cpu").numpy()
    matched_predicted_indices_np, matched_target_indices_np = linear_sum_assignment(iou_matrix_np, maximize=True)
    matched_predicted_indices = torch.from_numpy(matched_predicted_indices_np).to(device)
    matched_target_indices = torch.from_numpy(matched_target_indices_np).to(device)

    iou = iou_matrix[matched_predicted_indices, matched_target_indices]
    tp = tp_matrix[matched_predicted_indices, matched_target_indices]

    if min_iou_treshold > 0:
        if accept_equal_iou:
            keep = iou >= float(min_iou_treshold)
        else:
            keep = iou > float(min_iou_treshold)
        matched_predicted_indices = matched_predicted_indices[keep]
        matched_target_indices = matched_target_indices[keep]
        tp = tp[keep]

    matched_target_ids[matched_predicted_indices] = matched_target_indices
    matched_predicted_ids[matched_target_indices] = matched_predicted_indices

    fp = predicted_sizes[matched_predicted_indices] - tp
    fn = target_sizes[matched_target_indices] - tp

    metrics["tp"][matched_target_indices] = tp
    metrics["fp"][matched_target_indices] = fp
    metrics["fn"][matched_target_indices] = fn

    invalid_matches_mask = matched_target_ids == -1
    matched_target_ids = matched_target_ids + start_instance_id
    matched_target_ids[invalid_matches_mask] = invalid_instance_id

    invalid_matches_mask = matched_predicted_ids == -1
    matched_predicted_ids = matched_predicted_ids + start_instance_id
    matched_predicted_ids[invalid_matches_mask] = invalid_instance_id

    return matched_target_ids, matched_predicted_ids, metrics


def _initialize_matching_results(
    num_target_instances: int,
    num_predicted_instances: int,
    target_sizes: torch.Tensor,
    invalid_instance_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Initializes the output data structures for instance matching.

    Args:
        num_target_instances: Number of target instances.
        num_predicted_instances: Number of predicted instances.
        target_sizes: Number of points belonging to each target instance.
        invalid_instance_id: ID to use as default value for the matched instance IDs.
        device: Device on which to create the data structures.

    Returns: A tuple with the following elements:
        - :code:`matched_target_ids`: Tensor of length :code:`num_predicted_instances` with all values set to
            :code:`invalid_tree_id`.
        - :code:`matched_predicted_ids`: Tensor of length :code:`num_target_instances` with all values set to
            :code:`invalid_tree_id`.
        - :code:`metrics`: Dictionary with the keys :code:`"tp"`, :code:`"fp"`, :code:`"fn"`. The values are
          tensors whose length is equal to the number of target instances. code:`"tp"` and :code:`"fp"` are initialized
          with zero values, while :code:`fp` is initialized with the target sizes.
    """

    matched_target_ids = torch.full(
        (num_predicted_instances,), fill_value=invalid_instance_id, device=device, dtype=torch.long
    )
    matched_predicted_ids = torch.full(
        (num_target_instances,), fill_value=invalid_instance_id, device=device, dtype=torch.long
    )

    metrics = {
        "tp": torch.zeros(num_target_instances, device=device, dtype=torch.long),
        "fp": torch.zeros(num_target_instances, device=device, dtype=torch.long),
        "fn": target_sizes.clone().detach(),
    }

    return matched_target_ids, matched_predicted_ids, metrics


def _get_instance_sizes(
    target: torch.Tensor,
    prediction: torch.Tensor,
    num_target_instances: int,
    num_predicted_instances: int,
    invalid_instance_id: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sizes of target and predicted instances.

    Args:
        target: Ground truth instance ID for each point. The valid instance IDs are expected to be consecutive and start
            with zero.
        prediction: Predicted instance ID for each point. The valid instance IDs are expected to be consecutive and
            start with zero.
        num_target_instances: Number of target instances.
        num_predicted_instances: Number of predicted instances.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Must be a negative number.
            Defaults to `-1`.

    Returns: A Tuple with the following elements:
        - :code:`target_sizes`: Number of points belonging to each target instance.
        - :code:`predicted_sizes`: Number of points belonging to each predicted instance.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - Output:
            - :code:`target_sizes`: :math:`(T)`
            - :code:`predicted_sizes`: :math:`(P)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of predicted instances
        | :math:`T` = number of target instances
    """

    valid_target = target[target != invalid_instance_id]
    valid_prediction = prediction[prediction != invalid_instance_id]
    target_sizes = torch.bincount(valid_target, minlength=num_target_instances)
    predicted_sizes = torch.bincount(valid_prediction, minlength=num_predicted_instances)

    return target_sizes, predicted_sizes


def _get_matching_candidates(
    target: torch.Tensor,
    prediction: torch.Tensor,
    num_predicted_instances: int,
    invalid_instance_id: int = -1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Determines all pairs of target and predicted instances that overlap.

    Args:
        target: Ground truth instance ID for each point. The valid instance IDs are expected to be consecutive and start
            with zero.
        prediction: Predicted instance ID for each point. The valid instance IDs are expected to be consecutive and
            start zero.
        num_predicted_instances: Number of predicted instances.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Must be a negative number.
            Defaults to `-1`.

    Returns: :code:`None` if no instance pairs were found. Otherwise, a tuple with the following elements:
        - :code:`paired_target_ids`: Target ID for each pair of target and predicted instance. The target IDs are sorted
          in ascending order.
        - :code:`paired_predicted_ids`: Predicted ID for each pair of target and predicted instance. The predicted IDs
          paired with the same target ID are sorted in ascending order.
        - :code:`pair_counts`: Number of common points for each pair of target and predicted instance.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - Output:
            - :code:`paired_target_ids`: :math:`(P)`
            - :code:`paired_predicted_ids`: :math:`(P)`
            - :code:`pair_counts`: :math:`(P)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of pairs of target and predicted instances that overlap
    """

    valid_pair_mask = torch.logical_and((target != invalid_instance_id), (prediction != invalid_instance_id))
    valid_pair_target = target[valid_pair_mask]
    valid_pair_prediction = prediction[valid_pair_mask]

    if len(valid_pair_target) == 0:
        return None

    # assign a unique index to each possible combination of a target and a predicted ID
    pair_indices = valid_pair_target * num_predicted_instances + valid_pair_prediction

    unique_pair_indices, pair_counts = torch.unique(pair_indices, return_counts=True, sorted=True)

    paired_target_ids = torch.div(unique_pair_indices, num_predicted_instances, rounding_mode="floor").to(torch.long)
    paired_predicted_ids = (unique_pair_indices % num_predicted_instances).to(torch.long)

    return paired_target_ids, paired_predicted_ids, pair_counts
