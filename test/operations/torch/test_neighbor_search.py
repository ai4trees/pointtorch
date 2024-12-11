"""Tests for the pointtorch.operations.neighbor_search_cdist function."""

from typing import Optional

import hypothesis
import numpy as np
import pytest
import torch

from pointtorch.operations.torch import neighbor_search_cdist


class TestNeighorSearch:
    """Tests for the pointtorch.operations.neighbor_search_cdist function."""

    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        ),
        return_sorted=hypothesis.strategies.booleans(),
    )
    def test_radius_and_k_none(self, device: torch.device, return_sorted: bool):
        coords = torch.randn((30, 3), device=device, dtype=torch.float)
        point_cloud_sizes = torch.tensor([10, 20], device=device, dtype=torch.long)
        with pytest.raises(ValueError):
            neighbor_search_cdist(
                coords, coords, point_cloud_sizes, point_cloud_sizes, radius=None, k=None, return_sorted=return_sorted
            )

    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        ),
        k=hypothesis.strategies.one_of(
            hypothesis.strategies.none(), hypothesis.strategies.integers(min_value=1, max_value=15)
        ),
        radius=hypothesis.strategies.floats(min_value=0),
        return_sorted=hypothesis.strategies.booleans(),
    )
    def test_too_large_input(self, radius: float, k: int, device: torch.device, return_sorted: bool):
        point_cloud_size = 10**6
        coords = torch.randn((point_cloud_size * 2, 3), device=device, dtype=torch.float)
        point_cloud_sizes = torch.tensor([point_cloud_size, point_cloud_size], device=device, dtype=torch.long)

        with pytest.raises(ValueError):
            neighbor_search_cdist(
                coords, coords, point_cloud_sizes, point_cloud_sizes, radius=radius, k=k, return_sorted=return_sorted
            )

    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        ),
        k=hypothesis.strategies.one_of(
            hypothesis.strategies.none(), hypothesis.strategies.integers(min_value=1, max_value=15)
        ),
        return_sorted=hypothesis.strategies.booleans(),
    )
    def test_empty_neighborhood(self, device: torch.device, k: int, return_sorted: bool):
        batch_size = 2
        num_points = 10
        coords_support_points = torch.zeros((batch_size * num_points, 3), device=device, dtype=torch.float)
        coords_query_points = torch.ones((batch_size * num_points, 3), device=device, dtype=torch.float)
        point_cloud_sizes = torch.tensor([num_points, num_points], device=device, dtype=torch.long)

        neighbor_indices, neighbor_distances = neighbor_search_cdist(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes,
            point_cloud_sizes,
            radius=0.5,
            k=k,
            return_sorted=return_sorted,
        )

        expected_neighbor_indices = torch.empty((batch_size * num_points, 0), dtype=torch.long)
        expected_neighbor_distances = torch.empty((batch_size * num_points, 0), dtype=torch.float)

        np.testing.assert_array_equal(expected_neighbor_indices.numpy(), neighbor_indices.cpu().numpy())
        np.testing.assert_array_equal(expected_neighbor_distances.numpy(), neighbor_distances.cpu().numpy())

    @pytest.mark.parametrize("radius", [5, None])
    @hypothesis.settings(deadline=None)
    @hypothesis.given(
        device=hypothesis.strategies.sampled_from(
            [torch.device("cpu"), torch.device("cuda:0")]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else [torch.device("cpu")]
        ),
        k=hypothesis.strategies.one_of(
            hypothesis.strategies.none(), hypothesis.strategies.integers(min_value=1, max_value=15)
        ),
        return_sorted=hypothesis.strategies.booleans(),
    )
    def test_neighbor_search(  # pylint: disable=too-many-locals
        self, radius: Optional[float], device: torch.device, k: Optional[int], return_sorted: bool
    ):
        if radius is None and k is None:
            return

        coords_query_points = torch.tensor(
            [
                [0, 5, 0],
                [20, 20, 0],
                [3, 1, 0],
                [7, 1, 0],
                [10, 3, 0],
            ],
            dtype=torch.float,
            device=device,
        )

        coords_support_points = torch.tensor(
            [
                [0, 5, 0],
                [20, 18, 0],
                [4, 5, 0],
                [20, 20, 1],
                [20, 20, 0],
                [3, 1, 0],
                [4, 1, 0],
                [8, 2, 0],
                [8, 3, 0],
                [9, 3, 0],
                [10, 3, 0],
            ],
            dtype=torch.float,
            device=device,
        )

        point_cloud_sizes_query_points = torch.tensor([2, 3], dtype=torch.long, device=device)
        point_cloud_sizes_support_points = torch.tensor([5, 6], dtype=torch.long, device=device)

        if radius is None:
            expected_neighbor_indices = np.array(
                [
                    [0, 2, 1, 4, 3, 11],
                    [4, 3, 1, 2, 0, 11],
                    [5, 6, 7, 8, 9, 10],
                    [7, 8, 9, 6, 10, 5],
                    [10, 9, 8, 7, 6, 5],
                ]
            )

            expected_neighbor_distances = np.array(
                [
                    [
                        0,
                        4,
                        np.linalg.norm(np.array([0, 5, 0]) - np.array([20, 18, 0])),
                        25,
                        np.linalg.norm(np.array([0, 5, 0]) - np.array([20, 20, 1])),
                        np.inf,
                    ],
                    [0, 1, 2, np.linalg.norm(np.array([20, 20, 0]) - np.array([4, 5, 0])), 25, np.inf],
                    [
                        0,
                        1,
                        np.linalg.norm(np.array([3, 1, 0]) - np.array([8, 2, 0])),
                        np.linalg.norm(np.array([3, 1, 0]) - np.array([8, 3, 0])),
                        np.linalg.norm(np.array([3, 1, 0]) - np.array([9, 3, 0])),
                        np.linalg.norm(np.array([3, 1, 0]) - np.array([10, 3, 0])),
                    ],
                    [
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([8, 2, 0])),
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([8, 3, 0])),
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([9, 3, 0])),
                        3,
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([10, 3, 0])),
                        4,
                    ],
                    [
                        0,
                        1,
                        2,
                        np.linalg.norm(np.array([10, 3, 0]) - np.array([8, 2, 0])),
                        np.linalg.norm(np.array([10, 3, 0]) - np.array([4, 1, 0])),
                        np.linalg.norm(np.array([10, 3, 0]) - np.array([3, 1, 0])),
                    ],
                ]
            )
        else:
            expected_neighbor_indices = np.array(
                [
                    [0, 2, 11, 11, 11, 11],
                    [4, 3, 1, 11, 11, 11],
                    [5, 6, 11, 11, 11, 11],
                    [7, 8, 9, 6, 10, 5],
                    [10, 9, 8, 7, 11, 11],
                ]
            )
            expected_neighbor_distances = np.array(
                [
                    [0, 4, np.inf, np.inf, np.inf, np.inf],
                    [0, 1, 2, np.inf, np.inf, np.inf],
                    [0, 1, np.inf, np.inf, np.inf, np.inf],
                    [
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([8, 2, 0])),
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([8, 3, 0])),
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([9, 3, 0])),
                        3,
                        np.linalg.norm(np.array([7, 1, 0]) - np.array([10, 3, 0])),
                        4,
                    ],
                    [
                        0,
                        1,
                        2,
                        np.linalg.norm(np.array([10, 3, 0]) - np.array([8, 2, 0])),
                        np.inf,
                        np.inf,
                    ],
                ]
            )

        invalid_neighbor_index = 11

        neighbor_indices, neighbor_distances = neighbor_search_cdist(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            radius=radius,
            k=k,
            return_sorted=return_sorted,
        )
        neighbor_indices_np = neighbor_indices.cpu().numpy()
        neighbor_distances_np = neighbor_distances.cpu().numpy()

        if return_sorted:
            if k is not None:
                expected_neighbor_indices = expected_neighbor_indices[:, :k]
                expected_neighbor_distances = expected_neighbor_distances[:, :k]

            assert expected_neighbor_indices.shape == neighbor_indices.shape
            np.testing.assert_array_equal(expected_neighbor_indices, neighbor_indices_np)

            assert expected_neighbor_distances.shape == neighbor_distances.shape
            np.testing.assert_almost_equal(expected_neighbor_distances, neighbor_distances_np, decimal=6)
        else:
            if k is not None:
                for idx, row in enumerate(expected_neighbor_indices):
                    valid_indices = (neighbor_indices_np[idx] < invalid_neighbor_index).sum()
                    valid_distances = (np.isfinite(neighbor_distances_np[idx])).sum()
                    expected_valid_indices = (row < invalid_neighbor_index).sum()

                    assert k >= len(neighbor_indices_np[idx])
                    assert np.isin(neighbor_indices_np[idx], row).all()
                    assert valid_indices == min(k, expected_valid_indices)
                    assert valid_distances == min(k, expected_valid_indices)
            else:
                assert expected_neighbor_indices.shape == neighbor_indices.shape
                assert expected_neighbor_distances.shape == neighbor_distances.shape

                sorting_indices_expected = np.argsort(expected_neighbor_indices, axis=-1)
                sorting_indices = np.argsort(neighbor_indices_np, axis=-1)

                np.testing.assert_array_equal(
                    np.take_along_axis(expected_neighbor_indices, sorting_indices_expected, -1),
                    np.take_along_axis(neighbor_indices_np, sorting_indices, -1),
                )

                np.testing.assert_almost_equal(
                    np.take_along_axis(expected_neighbor_distances, sorting_indices_expected, -1),
                    np.take_along_axis(neighbor_distances_np, sorting_indices, -1),
                    decimal=6,
                )
