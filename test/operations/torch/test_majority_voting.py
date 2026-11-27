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

    @pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    def test_negativ_labels(self, device: str):
        labels = torch.tensor([0, 1, 1, 2, -1, -1, 3], dtype=torch.long, device=device)
        batch_indices = torch.tensor([0] * 4 + [1] * 3, dtype=torch.long, device=device)

        expected_majority_labels = np.array([1, -1], dtype=np.int64)

        majority_labels = majority_voting(labels, batch_indices)

        np.testing.assert_array_equal(expected_majority_labels, majority_labels.cpu().numpy())

    @pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    def test_sparse_labels(self, device: str):
        labels = torch.tensor([10, 10, 1000000, 1000001, 1000001], dtype=torch.long, device=device)
        batch_indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device)

        expected_majority_labels = np.array([10, 1000001], dtype=np.int64)

        majority_labels = majority_voting(labels, batch_indices)

        np.testing.assert_array_equal(expected_majority_labels, majority_labels.cpu().numpy())

    @pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    def test_ties(self, device: str):
        labels = torch.tensor([3, 1, 1, 3, 2, 5, 4, -1, -2, 7], dtype=torch.long, device=device)
        batch_indices = torch.tensor([0] * 5 + [1] * 2 + [2] * 2 + [3], dtype=torch.long, device=device)

        # the smallest of the most common labels is selected
        expected_majority_labels = np.array([1, 4, -2, 7], dtype=np.int64)

        majority_labels = majority_voting(labels, batch_indices)

        np.testing.assert_array_equal(expected_majority_labels, majority_labels.cpu().numpy())

    @pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    def test_ties_random_input(self, device: str):
        num_labels = 4
        num_batch_items = 50000
        # few labels per batch item, so that many batch items have several most common labels
        generator = np.random.default_rng(0)
        labels = generator.integers(0, num_labels, 200000)
        batch_indices = np.sort(generator.integers(0, num_batch_items, 200000))
        batch_indices = np.unique(batch_indices, return_inverse=True)[1]

        label_counts = np.zeros((batch_indices.max() + 1, num_labels), dtype=np.int64)
        np.add.at(label_counts, (batch_indices, labels), 1)
        # np.argmax returns the first, i.e., the smallest, of the most common labels
        expected_majority_labels = label_counts.argmax(axis=1)
        assert ((label_counts == label_counts.max(axis=1)[:, None]).sum(axis=1) > 1).any()

        # repeated calls return the same result, also on CUDA devices
        for _ in range(5):
            majority_labels = majority_voting(
                torch.from_numpy(labels).to(device), torch.from_numpy(batch_indices).to(device)
            )
            np.testing.assert_array_equal(expected_majority_labels, majority_labels.cpu().numpy())
