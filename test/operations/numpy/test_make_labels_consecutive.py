""" Tests for pointtorch.operations.numpy.make_labels_consecutive. """

import numpy as np

from pointtorch.operations.numpy import make_labels_consecutive


class TestMakeLabelsConsecutive:  # pylint: disable=too-few-public-methods
    """Tests for pointtorch.operations.numpy.make_labels_consecutive."""

    def test_make_labels_consecutive(self):
        labels = np.array([10, 20, 20, 10, 30])
        start_id = 5
        expected_output = np.array([5, 6, 6, 5, 7])
        output = make_labels_consecutive(labels, start_id)

        np.testing.assert_array_equal(output, expected_output)
