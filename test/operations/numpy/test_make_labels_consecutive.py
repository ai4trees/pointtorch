""" Tests for pointtorch.operations.np.make_labels_consecutive. """

from typing import Optional

import numpy as np
import pytest

from pointtorch.operations.numpy import make_labels_consecutive


class TestMakeLabelsConsecutive:
    """Tests for pointtorch.operations.numpy.make_labels_consecutive."""

    @pytest.mark.parametrize("ignore_id", [-1, None])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("return_unique_labels", [True, False])
    def test_make_labels_consecutive_empty_input(
        self, ignore_id: Optional[int], inplace: bool, return_unique_labels: bool
    ):
        output = make_labels_consecutive(
            np.array([], dtype=np.int64),
            ignore_id=ignore_id,
            inplace=inplace,
            return_unique_labels=return_unique_labels,
        )

        if return_unique_labels:
            transformed_labels, unique_labels = output
            assert len(unique_labels) == 0
        else:
            transformed_labels = output

        assert len(transformed_labels) == 0

    def test_make_labels_consecutive_no_remapping_necessary(self):
        num_labels = 10
        labels = np.arange(num_labels, dtype=np.int64)

        transformed_labels, unique_labels = make_labels_consecutive(
            labels, ignore_id=None, inplace=False, return_unique_labels=True
        )

        np.testing.assert_array_equal(labels, transformed_labels)
        np.testing.assert_array_equal(labels, unique_labels)

    @pytest.mark.parametrize("ignore_id", [-1, None])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("return_unique_labels", [True, False])
    def test_make_labels_consecutive_remapping_necessary(
        self, ignore_id: Optional[int], inplace: bool, return_unique_labels: bool
    ):
        labels = np.array([10, -1, 20, 20, 10, 30])
        start_id = 5
        if ignore_id is not None:
            expected_transformed_labels = np.array([5, -1, 6, 6, 5, 7])
            expected_unique_labels = np.arange(start_id, start_id + 3)
        else:
            expected_transformed_labels = np.array([6, 5, 7, 7, 6, 8])
            expected_unique_labels = np.arange(start_id, start_id + 4)

        output = make_labels_consecutive(
            labels, start_id, ignore_id=ignore_id, inplace=inplace, return_unique_labels=return_unique_labels
        )

        if return_unique_labels:
            transformed_labels, unique_labels = output

            np.testing.assert_array_equal(expected_unique_labels, unique_labels)
        else:
            transformed_labels = output

        np.testing.assert_array_equal(expected_transformed_labels, transformed_labels)

        if inplace:
            np.testing.assert_array_equal(labels, transformed_labels)
        else:
            assert (labels != transformed_labels).any()
