"""Tests for pointtorch.operations.numpy.non_max_suppression."""

import numpy as np
import pytest

from pointtorch.operations.numpy import compute_pairwise_ious, non_max_suppression


class TestNonMaximumSuppression:
    """Tests for pointtorch.operations.numpy.non_max_suppression."""

    def test_compute_pairwise_ious(self):
        instances = np.array([0, 1, 2, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        instance_sizes = np.array([3, 4, 2], dtype=np.int64)

        expected_pairwise_ious = np.array([[1, 2 / 5, 0], [2 / 5, 1, 0], [0, 0, 1]], dtype=np.float64)

        pairwise_ious = compute_pairwise_ious(instances, instance_sizes)

        np.testing.assert_almost_equal(expected_pairwise_ious, pairwise_ious)

    def test_compute_pairwise_ious_empty_input(self):
        instances = np.array([], dtype=np.int64)
        instance_sizes = np.array([0], dtype=np.int64)

        pairwise_ious = compute_pairwise_ious(instances, instance_sizes)

        assert len(pairwise_ious) == 0

    @pytest.mark.parametrize("iou_threshold", [0.9, 0.5, 0.1])
    def test_non_max_suppression(self, iou_threshold: float):
        ious = np.array([[1, 0.6, 0.2], [0.6, 1, 0], [0.2, 0, 1]], dtype=np.float64)
        scores = np.array([0.9, 0.7, 0.1], dtype=np.float64)

        if iou_threshold >= 0.9:
            expected_picked_instances = np.array([0, 1, 2], dtype=np.int64)
        elif iou_threshold >= 0.5:
            expected_picked_instances = np.array([0, 2], dtype=np.int64)
        else:
            expected_picked_instances = np.array([0], dtype=np.int64)

        picked_instances = non_max_suppression(ious, scores, iou_threshold=iou_threshold)

        np.testing.assert_array_equal(expected_picked_instances, picked_instances)

    def test_non_max_suppression_empty_input(self):
        ious = np.empty((0, 0), dtype=np.float64)
        scores = np.empty((0,), dtype=np.float64)

        picked_instances = non_max_suppression(ious, scores, iou_threshold=0.5)

        assert len(picked_instances) == 0
