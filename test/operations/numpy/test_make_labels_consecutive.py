""" Tests for pointtorch.operations.numpy.make_labels_consecutive. """

from typing import Optional

import numpy as np
import pytest

from pointtorch.operations.numpy import make_labels_consecutive


class TestMakeLabelsConsecutive:
    """Tests for pointtorch.operations.numpy.make_labels_consecutive"""

    @pytest.mark.parametrize("ignore_id", [-1, None])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_make_labels_consecutive_empty(self, ignore_id: Optional[int], inplace: bool):
        remapped_instance_ids, unique_instance_ids = make_labels_consecutive(
            np.array([], dtype=np.int64), ignore_id=ignore_id, inplace=inplace, return_unique_labels=True
        )

        assert len(remapped_instance_ids) == 0
        assert len(unique_instance_ids) == 0

    @pytest.mark.parametrize("ignore_id", [-1, None])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_make_labels_consecutive_no_remapping_necessary(self, ignore_id: Optional[int], inplace: bool):
        num_instances = 10
        instance_ids = np.arange(num_instances, dtype=np.int64)

        remapped_instance_ids, unique_instance_ids = make_labels_consecutive(
            instance_ids, ignore_id=ignore_id, inplace=inplace, return_unique_labels=True
        )

        np.testing.assert_array_equal(instance_ids, remapped_instance_ids)
        np.testing.assert_array_equal(instance_ids, unique_instance_ids)

    @pytest.mark.parametrize("ignore_id", [-1, None])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_make_labels_consecutive_remapping_necessary(self, ignore_id: Optional[int], inplace: bool):
        offset = 5
        num_instances = 10
        instance_ids = np.concatenate(
            [
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(offset + num_instances / 2, offset + num_instances, dtype=np.int64),
            ]
        )

        expected_remapped_instance_ids = np.concatenate(
            [
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(num_instances / 2, num_instances, dtype=np.int64),
            ]
        )
        expected_unique_instance_ids = np.arange(num_instances, dtype=np.int64)

        remapped_instance_ids, unique_instance_ids = make_labels_consecutive(
            instance_ids, ignore_id=ignore_id, inplace=inplace, return_unique_labels=True
        )

        np.testing.assert_array_equal(expected_remapped_instance_ids, remapped_instance_ids)
        np.testing.assert_array_equal(expected_unique_instance_ids, unique_instance_ids)
