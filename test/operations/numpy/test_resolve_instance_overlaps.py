"""Tests for pointtorch.operations.numpy.resolve_instance_overlaps."""

import numpy as np

from pointtorch.operations.numpy import resolve_instance_overlaps


class TestResolveInstanceOverlaps:  # pylint: disable = too-few-public-methods
    """Tests for pointtorch.operations.numpy.resolve_instance_overlaps."""

    def test_resolve_instance_overlaps(self):
        instances = np.array([0, 1, 0, 1, 2, 3, 4, 2, 4, 5, 6, 10, 10, 12], dtype=np.int64)
        instance_sizes = np.array([2, 5, 4, 1, 2], dtype=np.int64)
        scores = np.array([0.5, 0.9, 0.4, 0.7, 0.8], dtype=np.float32)

        expected_filtered_instances = np.array([0, 1, 2, 3, 4, 5, 6, 10, 12], dtype=np.int64)
        expected_filtered_instance_batch_indices = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
        expected_filtered_instance_sizes = np.array([5, 2, 2], dtype=np.int64)
        expected_selected_indices = np.array([1, 2, 4])

        filtered_instances, filtere_instance_batch_indices, filtered_instance_sizes, selected_indices = (
            resolve_instance_overlaps(instances, instance_sizes, scores)
        )

        np.testing.assert_array_equal(expected_filtered_instances, filtered_instances)
        np.testing.assert_array_equal(expected_filtered_instance_batch_indices, filtere_instance_batch_indices)
        np.testing.assert_array_equal(expected_filtered_instance_sizes, filtered_instance_sizes)
        np.testing.assert_array_equal(expected_selected_indices, selected_indices)

    def test_empty_input(self):
        instances = np.array([], dtype=np.int64)
        instance_sizes = np.array([], dtype=np.int64)
        scores = np.array([], dtype=np.float32)

        filtered_instances, filtere_instance_batch_indices, filtered_instance_sizes, selected_indices = (
            resolve_instance_overlaps(instances, instance_sizes, scores)
        )

        assert len(filtered_instances) == 0
        assert len(filtere_instance_batch_indices) == 0
        assert len(filtered_instance_sizes) == 0
        assert len(selected_indices) == 0

    def test_single_point(self):
        instances = np.array([0], dtype=np.int64)
        instance_sizes = np.array([1], dtype=np.int64)
        scores = np.array([0.7], dtype=np.float32)

        expected_filtered_instance_batch_indices = np.array([0], dtype=np.int64)

        filtered_instances, filtere_instance_batch_indices, filtered_instance_sizes, selected_indices = (
            resolve_instance_overlaps(instances, instance_sizes, scores)
        )

        np.testing.assert_array_equal(instances, filtered_instances)
        np.testing.assert_array_equal(expected_filtered_instance_batch_indices, filtere_instance_batch_indices)
        np.testing.assert_array_equal(instance_sizes, filtered_instance_sizes)
        np.testing.assert_array_equal(np.zeros(1, dtype=np.int64), selected_indices)
