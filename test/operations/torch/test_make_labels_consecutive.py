""" Tests for pointtorch.operations.numpy.make_labels_consecutive. """

import numpy as np
import pytest
import torch

from pointtorch.operations.torch import make_labels_consecutive


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestMakeLabelsConsecutive:  # pylint: disable=too-few-public-methods
    """Tests for pointtorch.operations.numpy.make_labels_consecutive."""

    def test_make_labels_consecutive(self, device: str):
        labels = torch.tensor([10, 20, 20, 10, 30], device=device)
        start_id = 5
        expected_output = torch.tensor([5, 6, 6, 5, 7], device=device)
        output = make_labels_consecutive(labels, start_id)
        np.testing.assert_array_equal(output.cpu().numpy(), expected_output.cpu().numpy())
