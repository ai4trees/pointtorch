"""Tests for pointtorch.metrics.match_instances."""

import numpy as np
import pytest
import torch


from pointtorch.metrics.instance_segmentation import (
    match_instances,
    match_instances_iou,
    match_instances_point2tree,
    match_instances_tree_learn,
)


@pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
class TestMetrics:
    """Tests for pointtorch.metrics.match_instances."""

    @pytest.mark.parametrize(
        "method", ["panoptic_segmentation", "point2tree", "tree_learn", "for_instance", "for_ai_net"]
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances(self, method: str, invalid_instance_id: int, device: str):
        start_instance_id = invalid_instance_id + 1
        target = torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 3, 3, 3, 3, -1], dtype=torch.long, device=device)
        prediction = torch.tensor(
            [2, 2, 2, 2, -1, 3, 1, 0, 0, 1, 2, -1, -1, -1, -1, -1], dtype=torch.long, device=device
        )

        target += start_instance_id
        prediction += start_instance_id

        if method in ["point2tree", "for_instance"]:
            xyz = torch.zeros((len(target), 3), dtype=torch.float, device=device)
            xyz[:, 2] = torch.tensor(
                [0, 1, 10, 0, 1, 2, 5, 1, 2, 3, 4, 0, 0, 0, 15, 0], dtype=torch.float, device=device
            )
        else:
            xyz = None

        if method == "point2tree":
            expected_matched_target_ids = np.array([0, -1, 1, 2], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 2, 3, -1], dtype=np.int64)

            expected_metrics = {
                "iou": np.array([2 / 3, 4 / 5, 1 / 4, 0], dtype=np.float32),
                "precision": np.array([1, 4 / 5, 1, 0], dtype=np.float32),
                "recall": np.array([2 / 3, 1, 1 / 4, 0], dtype=np.float32),
            }
        else:
            expected_matched_target_ids = np.array([0, -1, 1, -1], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 2, -1, -1], dtype=np.int64)

            expected_metrics = {
                "iou": np.array([2 / 3, 4 / 5, 0, 0], dtype=np.float32),
                "precision": np.array([1, 4 / 5, 0, 0], dtype=np.float32),
                "recall": np.array([2 / 3, 1, 0, 0], dtype=np.float32),
            }

        expected_matched_target_ids += start_instance_id
        expected_matched_predicted_ids += start_instance_id

        matched_target_ids, matched_predicted_ids, metrics = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

        for key, expected_metric in expected_metrics.items():
            np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_all_correct(self, method: str, invalid_instance_id: int, device: str):
        start_instance_id = invalid_instance_id + 1
        if method in ["point2tree", "for_instance"]:
            xyz = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 20],
                    [1, 1, 0],
                    [1, 1, 30],
                    [2, 2, 0],
                    [2, 2, 10],
                ],
                dtype=torch.float,
                device=device,
            )
        else:
            xyz = None
        target = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device) + start_instance_id
        prediction = torch.tensor([1, 1, 0, 0, 2, 2], dtype=torch.long, device=device) + start_instance_id

        matched_target_ids, matched_predicted_ids, metrics = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        expected_matched_target_ids = np.array([1, 0, 2], dtype=np.int64) + start_instance_id
        expected_matched_predicted_ids = np.array([1, 0, 2], dtype=np.int64) + start_instance_id

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

        for key in ["iou", "precision", "recall"]:
            np.testing.assert_array_equal(
                np.ones(len(matched_target_ids), dtype=np.float32), metrics[key].cpu().numpy()
            )

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_all_false_negatives(self, method: str, invalid_instance_id: int, device: str):
        start_instance_id = invalid_instance_id + 1
        target = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device) + start_instance_id
        prediction = torch.full((len(target),), fill_value=invalid_instance_id, dtype=torch.long, device=device)

        if method in ["point2tree"]:
            xyz = np.random.randn(6, 3)
        else:
            xyz = None

        matched_target_ids, matched_predicted_ids, metrics = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        np.testing.assert_array_equal(np.array([], dtype=np.int64), matched_target_ids.cpu().numpy())
        np.testing.assert_array_equal(
            np.full((3,), fill_value=invalid_instance_id, dtype=np.int64), matched_predicted_ids.cpu().numpy()
        )

        for key in ["iou", "precision", "recall"]:
            np.testing.assert_array_equal(
                np.zeros(len(matched_predicted_ids), dtype=np.float32), metrics[key].cpu().numpy()
            )

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_all_false_positives(self, method: str, invalid_instance_id: int, device: str):
        start_instance_id = invalid_instance_id + 1

        prediction = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device) + start_instance_id
        target = torch.full((len(prediction),), fill_value=-1, dtype=torch.long, device=device) + start_instance_id

        if method in ["point2tree", "for_instance"]:
            xyz = torch.randn((len(target), 3), dtype=torch.float, device=device)
        else:
            xyz = None

        matched_target_ids, matched_predicted_ids, metrics = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        np.testing.assert_array_equal(
            np.full((3,), fill_value=invalid_instance_id, dtype=np.int64), matched_target_ids.cpu().numpy()
        )
        np.testing.assert_array_equal(np.array([], dtype=np.int64), matched_predicted_ids.cpu().numpy())

        for key in ["iou", "precision", "recall"]:
            np.testing.assert_array_equal(
                np.zeros(len(matched_predicted_ids), dtype=np.float32), metrics[key].cpu().numpy()
            )

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_no_overlap_of_target_and_prediction(self, method: str, invalid_instance_id: int, device: str):
        start_instance_id = invalid_instance_id + 1

        prediction = torch.tensor([0, 1, -1, -1, -1], dtype=torch.long, device=device) + start_instance_id
        target = torch.tensor([-1, -1, 0, 0, 0], dtype=torch.long, device=device) + start_instance_id

        if method in ["point2tree", "for_instance"]:
            xyz = torch.randn((len(target), 3), dtype=torch.float, device=device)
        else:
            xyz = None

        matched_target_ids, matched_predicted_ids, metrics = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        np.testing.assert_array_equal(
            np.full((2,), fill_value=invalid_instance_id, dtype=np.int64), matched_target_ids.cpu().numpy()
        )
        np.testing.assert_array_equal(
            np.full((1,), fill_value=invalid_instance_id, dtype=np.int64), matched_predicted_ids.cpu().numpy()
        )

        for key in ["iou", "precision", "recall"]:
            np.testing.assert_array_equal(
                np.zeros(len(matched_predicted_ids), dtype=np.float32), metrics[key].cpu().numpy()
            )

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_prediction_with_multiple_matches(self, invalid_instance_id: int, device: str):
        start_instance_id = invalid_instance_id + 1

        target = torch.tensor([0, -1, 2, 2, 3, -1, -1, 1, 1, 1], device=device) + start_instance_id
        unique_target_ids = torch.unique(target)
        unique_target_ids = unique_target_ids[unique_target_ids != invalid_instance_id]
        prediction = torch.tensor([0, 0, 0, 0, 0, -1, -1, 2, 2, 1], device=device) + start_instance_id
        unique_prediction_ids = torch.unique(prediction)
        unique_prediction_ids = unique_prediction_ids[unique_prediction_ids != invalid_instance_id]

        iou_threshold = 0.1

        expected_matched_target_ids = np.array([2, -1, 1], dtype=np.int64) + start_instance_id
        expected_matched_predicted_ids = np.array([0, 2, 0, 0], dtype=np.int64) + start_instance_id
        expected_metrics = {
            "iou": np.array([1 / 5, 2 / 3, 2 / 5, 1 / 5], dtype=np.float32),
            "precision": np.array([1 / 5, 1, 2 / 5, 1 / 5], dtype=np.float32),
            "recall": np.array([1, 2 / 3, 1, 1], dtype=np.float32),
        }

        matched_target_ids, matched_predicted_ids, metrics = match_instances_iou(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id=start_instance_id,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=0.1,
        )

        np.testing.assert_array_equal(matched_target_ids.cpu().numpy(), expected_matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids.cpu().numpy(), expected_matched_predicted_ids)
        for key, expected_metric in expected_metrics.items():
            np.testing.assert_array_equal(metrics[key].cpu().numpy(), expected_metric)

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    def test_target_and_prediction_with_different_lengths(self, method: str, device: str):
        target = torch.zeros(1, dtype=torch.long, device=device)
        prediction = torch.zeros(2, dtype=torch.long, device=device)

        with pytest.raises(ValueError):
            match_instances(target, prediction, method=method)

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    def test_non_continuous_target_ids(self, method: str, device: str):
        target = torch.tensor([0, 2], dtype=torch.long, device=device)
        prediction = torch.zeros_like(target)

        with pytest.raises(ValueError):
            match_instances(target, prediction, method=method)

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    def test_non_continuous_prediction_ids(self, method: str, device: str):
        prediction = torch.tensor([0, 2], dtype=torch.long, device=device)
        target = torch.zeros_like(prediction)

        with pytest.raises(ValueError):
            match_instances(target, prediction, method=method)

    @pytest.mark.parametrize(
        "method",
        ["point2tree", "for_instance"],
    )
    def test_missing_xyz(self, method: str, device: str):
        prediction = torch.tensor([0, 0], dtype=torch.long, device=device)
        target = torch.zeros_like(prediction)

        with pytest.raises(ValueError):
            match_instances(target, prediction, xyz=None, method=method)

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    def test_different_start_ids(self, method: str, device: str):
        target = torch.tensor([3, 4], dtype=torch.long, device=device)
        prediction = torch.tensor([5, 6], dtype=torch.long, device=device)

        with pytest.raises(ValueError):
            match_instances(target, prediction, method=method)

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    def test_invalid_invalid_instance_id(self, method: str, device: str):
        target = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        prediction = torch.zeros_like(target)

        with pytest.raises(ValueError):
            match_instances(target, prediction, method=method, invalid_instance_id=2)

    def test_invalid_method(self, device: str):
        xyz = torch.randn((5, 3), dtype=torch.float, device=device)
        target = torch.zeros(5, dtype=torch.long, device=device)
        prediction = torch.zeros(5, dtype=torch.long, device=device)

        with pytest.raises(ValueError):
            match_instances(target, prediction, xyz=xyz, method="test")

    @pytest.mark.parametrize("min_iou_treshold", [None, 0.2, 0.5])
    @pytest.mark.parametrize("accept_equal_iou", [True, False])
    @pytest.mark.parametrize("sort_by_target_height", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances_point2tree_sort_target_by_height(
        self,
        min_iou_treshold: float,
        accept_equal_iou: bool,
        sort_by_target_height: bool,
        invalid_instance_id: int,
        device: str,
    ):
        start_instance_id = invalid_instance_id + 1
        xyz = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 10],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 30],
                [2, 2, 0],
                [2, 2, 20],
            ],
            dtype=torch.long,
            device=device,
        )
        target = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2], dtype=torch.long, device=device) + start_instance_id
        prediction = torch.tensor([0, 0, 0, 0, -1, -1, 1, -1], dtype=torch.long, device=device) + start_instance_id

        unique_target_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device) + start_instance_id
        unique_prediction_ids = torch.tensor([0, 1], dtype=torch.long, device=device) + start_instance_id

        matched_target_ids, matched_predicted_ids, metrics = match_instances_point2tree(
            xyz,
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id,
            invalid_instance_id,
            min_iou_treshold=min_iou_treshold,
            accept_equal_iou=accept_equal_iou,
            sort_by_target_height=sort_by_target_height,
        )

        expected_metrics = {
            "iou": np.array([0.5, 0, 0.5], dtype=np.float32),
            "precision": np.array([0.5, 0, 1], dtype=np.float32),
            "recall": np.array([1, 0, 0.5], dtype=np.float32),
        }

        if min_iou_treshold is not None and min_iou_treshold >= 0.4:
            if accept_equal_iou:
                expected_matched_target_ids = np.array([0, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([0, -1, 1], dtype=np.int64)
            else:
                expected_matched_target_ids = np.array([-1, -1], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, -1, -1], dtype=np.int64)
        else:
            if sort_by_target_height:
                expected_matched_target_ids = np.array([1, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, 0, 1], dtype=np.int64)

                expected_metrics = {
                    "iou": np.array([0, 2 / 6, 0.5], dtype=np.float32),
                    "precision": np.array([0, 0.5, 1], dtype=np.float32),
                    "recall": np.array([0, 0.5, 0.5], dtype=np.float32),
                }
            else:
                expected_matched_target_ids = np.array([0, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([0, -1, 1], dtype=np.int64)
        expected_matched_target_ids = expected_matched_target_ids + start_instance_id
        expected_matched_predicted_ids = expected_matched_predicted_ids + start_instance_id

        for key, metric in expected_metrics.items():
            metric[expected_matched_predicted_ids == invalid_instance_id] = 0
            expected_metrics[key] = metric

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

        if min_iou_treshold == 0.5 and accept_equal_iou and sort_by_target_height:
            matched_target_ids, matched_predicted_ids, metrics = match_instances(
                target, prediction, xyz=xyz, method="for_instance", invalid_instance_id=invalid_instance_id
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
            np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

            for key, expected_metric in expected_metrics.items():
                np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

        if min_iou_treshold is None and sort_by_target_height:
            matched_target_ids, matched_predicted_ids, metrics = match_instances(
                target, prediction, xyz=xyz, method="point2tree", invalid_instance_id=invalid_instance_id
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
            np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

            for key, expected_metric in expected_metrics.items():
                np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

        if min_iou_treshold == 0.5 and accept_equal_iou and not sort_by_target_height:
            matched_target_ids, matched_predicted_ids, metrics = match_instances(
                target, prediction, method="for_ai_net", invalid_instance_id=invalid_instance_id
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
            np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

            for key, expected_metric in expected_metrics.items():
                np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

        if min_iou_treshold == 0.5 and not accept_equal_iou and not sort_by_target_height:
            matched_target_ids, matched_predicted_ids, metrics = match_instances(
                target, prediction, method="panoptic_segmentation", invalid_instance_id=invalid_instance_id
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
            np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

            for key, expected_metric in expected_metrics.items():
                np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

    @pytest.mark.parametrize("min_iou_treshold", [None, 0.2, 0.5])
    @pytest.mark.parametrize("accept_equal_iou", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1])  # , 0])
    def test_match_instances_tree_learn(
        self, min_iou_treshold: float, accept_equal_iou: bool, invalid_instance_id: int, device: str
    ):
        start_instance_id = invalid_instance_id + 1
        # test that Hungarian matching works as expected
        target = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.long, device=device) + start_instance_id
        unique_target_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device) + start_instance_id
        prediction = (
            torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 2, -1], dtype=torch.long, device=device) + start_instance_id
        )
        unique_prediction_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device) + start_instance_id
        xyz = torch.zeros((len(target), 3), dtype=torch.float, device=device)

        if min_iou_treshold is not None and min_iou_treshold == 0.5:
            if accept_equal_iou:
                expected_matched_target_ids = np.array([-1, -1, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, -1, 2], dtype=np.int64)

                expected_metrics = {
                    "iou": np.array([0, 0, 0.5], dtype=np.float32),
                    "precision": np.array([0, 0, 1], dtype=np.float32),
                    "recall": np.array([0, 0, 0.5], dtype=np.float32),
                }
            else:
                expected_matched_target_ids = np.array([-1, -1, -1], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, -1, -1], dtype=np.int64)

                expected_metrics = {
                    "iou": np.array([0, 0, 0], dtype=np.float32),
                    "precision": np.array([0, 0, 0], dtype=np.float32),
                    "recall": np.array([0, 0, 0], dtype=np.float32),
                }
        else:
            expected_matched_target_ids = np.array([1, 0, 2], dtype=np.int64)
            expected_matched_predicted_ids = np.array([1, 0, 2], dtype=np.int64)

            expected_metrics = {
                "iou": np.array([2 / 7, 2 / 7, 0.5], dtype=np.float32),
                "precision": np.array([1, 2 / 7, 1], dtype=np.float32),
                "recall": np.array([2 / 7, 1, 0.5], dtype=np.float32),
            }

        expected_matched_target_ids = expected_matched_target_ids + start_instance_id
        expected_matched_predicted_ids = expected_matched_predicted_ids + start_instance_id

        matched_target_ids, matched_predicted_ids, metrics = match_instances_tree_learn(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id,
            invalid_instance_id,
            min_iou_treshold=min_iou_treshold,
            accept_equal_iou=accept_equal_iou,
        )

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

        for key, expected_metric in expected_metrics.items():
            np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

        if min_iou_treshold == 0.5 and not accept_equal_iou:
            matched_target_ids, matched_predicted_ids, metrics = match_instances(
                target, prediction, xyz=xyz, method="tree_learn", invalid_instance_id=invalid_instance_id
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
            np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

            for key, expected_metric in expected_metrics.items():
                np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

    @pytest.mark.parametrize("min_iou_treshold", [None, 0.2, 0.4, 0.5])
    @pytest.mark.parametrize("accept_equal_iou", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances_for_ai_net_coverage(
        self, min_iou_treshold: float, accept_equal_iou: bool, invalid_instance_id: int, device: str
    ):
        start_instance_id = invalid_instance_id + 1

        target = (
            torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4], dtype=torch.long, device=device) + start_instance_id
        )
        prediction = (
            torch.tensor([0, 0, 0, 0, -1, 1, -1, -1, -1, 2, 3, 3, 3], dtype=torch.long, device=device)
            + start_instance_id
        )

        unique_target_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device) + start_instance_id
        unique_prediction_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device) + start_instance_id

        matched_target_ids, matched_predicted_ids, metrics = match_instances_iou(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            start_instance_id,
            invalid_instance_id,
            min_iou_treshold=min_iou_treshold,
            accept_equal_iou=accept_equal_iou,
        )

        expected_matched_target_ids = np.array([], dtype=np.int64)
        expected_matched_predicted_ids = np.array([], dtype=np.int64)
        if min_iou_treshold is None or min_iou_treshold == 0.2 or (min_iou_treshold == 0.4 and accept_equal_iou):
            expected_matched_target_ids = np.array([0, 2, -1, 4], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 0, 1, -1, 3], dtype=np.int64)
        elif (min_iou_treshold == 0.4 and not accept_equal_iou) or (min_iou_treshold == 0.5 and accept_equal_iou):
            expected_matched_target_ids = np.array([0, 2, -1, 4], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, -1, 1, -1, 3], dtype=np.int64)
        elif min_iou_treshold == 0.5 and not accept_equal_iou:
            expected_matched_target_ids = np.array([-1, -1, -1, 4], dtype=np.int64)
            expected_matched_predicted_ids = np.array([-1, -1, -1, -1, 3], dtype=np.int64)

        expected_metrics = {
            "iou": torch.tensor([0.5, 2 / 5, 0.5, 0, 3 / 4]),
            "precision": torch.tensor([0.5, 0.5, 1, 0, 1]),
            "recall": torch.tensor([1, 2 / 3, 0.5, 0, 3 / 4]),
        }
        for key, metric in expected_metrics.items():
            metric[expected_matched_predicted_ids == -1] = 0
            expected_metrics[key] = metric

        expected_matched_target_ids = expected_matched_target_ids + start_instance_id
        expected_matched_predicted_ids = expected_matched_predicted_ids + start_instance_id

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

        for key, expected_metric in expected_metrics.items():
            np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())

        if min_iou_treshold is None:
            matched_target_ids, matched_predicted_ids, metrics = match_instances(
                target, prediction, method="for_ai_net_coverage", invalid_instance_id=invalid_instance_id
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids.cpu().numpy())
            np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids.cpu().numpy())

            for key, expected_metric in expected_metrics.items():
                np.testing.assert_array_equal(expected_metric, metrics[key].cpu().numpy())
