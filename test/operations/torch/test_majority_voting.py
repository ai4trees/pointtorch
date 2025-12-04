"""Tests for pointtorch.operations.torch.majority_voting."""

import numpy as np
import pytest
import torch

from pointtorch.operations.torch import majority_voting


class TestMajorityVoting:
    """Tests for pointtorch.operations.torch.majority_voting."""

    @pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    def test_majority_voting(self, device: str):
        labels = torch.tensor([0, 1, 1, 2, 2, 2, 3], dtype=torch.long, device=device)
        batch_indices = torch.tensor([0] * 4 + [1] * 3, dtype=torch.long, device=device)

        expected_majority_labels = np.array([1, 2], dtype=np.int64)

        majority_labels = majority_voting(labels, batch_indices)

        np.testing.assert_array_equal(expected_majority_labels, majority_labels.cpu().numpy())
