"""Tests for the kNN implementations in pointtorch.operations """

from typing import Callable, Tuple

from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np
import pytest
import torch

from pointtorch.operations.torch import (
    knn_search,
    knn_search_cdist,
    knn_search_pytorch3d,
    knn_search_open3d,
    knn_search_torch_cluster,
)
from pointtorch.config import open3d_is_available, pytorch3d_is_available


class TestKnnSearch:
    """Tests for the kNN implementations in pointtorch.operations"""

    @staticmethod
    def _naive_knn_search(  # pylint: disable=too-many-locals
        coords_support_points: torch.Tensor,
        coords_query_points: torch.Tensor,
        point_cloud_sizes_support_points: torch.Tensor,
        point_cloud_sizes_query_points: torch.Tensor,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Naive implementation of knn search to compute expected results for arbitrary inputs.

        Returns: Expected neighbor indices and distances.
        """

        coords_support_points = coords_support_points.cpu().numpy()
        coords_query_points = coords_query_points.cpu().numpy()
        point_cloud_sizes_support_points = point_cloud_sizes_support_points.cpu().numpy()
        point_cloud_sizes_query_points = point_cloud_sizes_query_points.cpu().numpy()

        all_neighbor_indices = []
        all_neighbor_dists = []
        min_support_point = 0
        max_support_point = int(point_cloud_sizes_support_points[0].item())
        batch_idx = 0

        for q_idx, point in enumerate(coords_query_points):
            neighbor_indices = []
            neighbor_dists = []
            max_dist = np.inf
            if q_idx == point_cloud_sizes_query_points[batch_idx]:
                min_support_point = int(point_cloud_sizes_support_points[batch_idx].item())
                max_support_point = int(point_cloud_sizes_support_points[batch_idx + 1].item())
                batch_idx += 1
            for s_idx, support_point in enumerate(coords_support_points[min_support_point:max_support_point, :]):
                dist = np.linalg.norm(point - support_point)
                if dist < max_dist:
                    neighbor_indices.append(s_idx)
                    neighbor_dists.append(dist)
                    sorted_indices = np.argsort(neighbor_dists)
                    neighbor_indices = list(np.array(neighbor_indices)[sorted_indices])[:k]
                    neighbor_dists = list(np.array(neighbor_dists)[sorted_indices])[:k]
                    if len(neighbor_dists) == k:
                        max_dist = float(neighbor_dists[-1])
            all_neighbor_indices.append(neighbor_indices)
            all_neighbor_dists.append(neighbor_dists)

        max_neighbors = max(len(neighbor_indices) for neighbor_indices in all_neighbor_indices)

        invalid_neighbor_index = len(coords_support_points)
        all_neighbor_indices_np = np.full((len(coords_query_points), max_neighbors), fill_value=invalid_neighbor_index)
        all_neighbor_dists_np = np.full((len(coords_query_points), max_neighbors), fill_value=np.inf)

        for idx, neighbor_indices in enumerate(all_neighbor_indices):
            all_neighbor_indices_np[idx, : len(neighbor_indices)] = np.array(neighbor_indices)
            all_neighbor_dists_np[idx, : len(neighbor_indices)] = all_neighbor_dists[idx]

        return all_neighbor_indices_np, all_neighbor_dists_np

    @pytest.mark.parametrize(
        "knn_search_implementation,pytorch3d_available",
        [
            (knn_search, True),
            (knn_search, False),
            (knn_search_cdist, True),
            (knn_search_open3d, True),
            (knn_search_pytorch3d, True),
            (knn_search_torch_cluster, True),
        ],
    )
    @pytest.mark.parametrize("return_sorted", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    @given(
        num_query_points=st.integers(min_value=1, max_value=50),
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck(9)])
    def test_knn_search_random_inputs(  # pylint: disable=too-many-locals, too-many-branches
        self,
        knn_search_implementation: Callable,
        pytorch3d_available: bool,
        num_query_points: int,
        k: int,
        return_sorted: bool,
        device: torch.device,
        monkeypatch,
    ):
        if not pytorch3d_available:
            monkeypatch.setattr("pointtorch.operations.torch._knn_search.pytorch3d_is_available", lambda: False)

        if knn_search_implementation == knn_search_open3d and (  # pylint: disable=comparison-with-callable
            not open3d_is_available() or not return_sorted or device == "cuda:0"
        ):
            # skip tests of Open3D implementation if Open3D is not installed or return_sorted is False or the device is
            # a GPU (the Open3D implementation does not offer a return_sorted parameter and only works on CPU)
            return
        if (  # pylint: disable=comparison-with-callable
            knn_search_implementation == knn_search_pytorch3d and not pytorch3d_is_available()
        ):
            # skip tests of PyTorch3D implementation if PyTorch3D is not installed
            return

        coords_query_points = torch.rand((num_query_points, 3), device=device) * 5
        coords_support_points = torch.round(torch.rand((num_query_points * 2, 3), device=device) * 5, decimals=4)
        coords_support_points = torch.unique(coords_support_points, dim=0)

        batch_indices_query_points = torch.zeros(len(coords_query_points), dtype=torch.long, device=device)
        batch_indices_support_points = torch.zeros(len(coords_support_points), dtype=torch.long, device=device)
        point_cloud_sizes_query_points = torch.tensor([len(coords_query_points)], dtype=torch.long, device=device)
        point_cloud_sizes_support_points = torch.tensor([len(coords_support_points)], dtype=torch.long, device=device)

        if knn_search_implementation in [knn_search_cdist, knn_search_open3d]:
            batch_idx_inputs = [point_cloud_sizes_support_points, point_cloud_sizes_query_points]
        elif knn_search_implementation in [knn_search_torch_cluster]:
            batch_idx_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
            ]
        elif knn_search_implementation in [knn_search_pytorch3d]:
            batch_idx_inputs = [
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
            ]
        else:
            batch_idx_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
            ]
        if knn_search_implementation in [knn_search, knn_search_cdist, knn_search_pytorch3d]:
            kwargs = {"return_sorted": return_sorted}
        else:
            kwargs = {}

        neighbor_indices, neighbor_dists = knn_search_implementation(
            coords_support_points, coords_query_points, *batch_idx_inputs, k, **kwargs
        )

        expected_neighbor_indices, expected_neighbor_dists = self._naive_knn_search(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            k,
        )

        assert expected_neighbor_indices.shape == neighbor_indices.shape
        assert expected_neighbor_dists.shape == expected_neighbor_dists.shape

        if not return_sorted:
            neighbor_dists, sorting_indices = torch.sort(neighbor_dists)
            neighbor_indices = torch.gather(neighbor_indices, -1, sorting_indices)

        neighbor_dists = neighbor_dists.cpu().numpy()
        neighbor_indices = neighbor_indices.cpu().numpy()

        for idx, current_expected_neighbor_dists in enumerate(expected_neighbor_dists):
            expected_equal_indices = []
            equal_indices = []
            previouds_dist = None
            eps = 1e-4
            for neighbor_idx, expected_neighbor_dist in enumerate(current_expected_neighbor_dists):
                neighbor_dist = neighbor_dists[idx, neighbor_idx]
                assert expected_neighbor_dist == pytest.approx(neighbor_dist, 4)

                if previouds_dist is not None and np.abs(neighbor_dist - previouds_dist) > eps:
                    np.testing.assert_array_equal(
                        np.sort(np.array(expected_equal_indices)), np.sort(np.array(equal_indices))
                    )
                    expected_equal_indices = []
                    equal_indices = []
                    previouds_dist = None

                # when two support points roughly have the same distance to a query point, they may appear in different
                # orders in the expected_neighbor_indices and the neighbor_indices tensor
                # to account for this, all neighbors with roughly the same distance are collected and the collected
                # indices are sorted and compared to the expected indices afterwards
                if expected_neighbor_indices[idx, neighbor_idx] != neighbor_indices[idx, neighbor_idx]:
                    expected_equal_indices.append(expected_neighbor_indices[idx, neighbor_idx])
                    equal_indices.append(neighbor_indices[idx, neighbor_idx])
                    previouds_dist = neighbor_dist

    @pytest.mark.parametrize(
        "knn_search_implementation",
        [
            knn_search,
            knn_search_cdist,
            knn_search_open3d,
            knn_search_pytorch3d,
            knn_search_torch_cluster,
        ],
    )
    @pytest.mark.parametrize("k", [2, 4])
    @pytest.mark.parametrize("return_sorted", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_knn_search_small_example_input(  # pylint: disable=too-many-locals
        self,
        knn_search_implementation: Callable,
        k: int,
        return_sorted: bool,
        device: torch.device,
    ):
        if knn_search_implementation == knn_search_open3d and (  # pylint: disable=comparison-with-callable
            not open3d_is_available() or not return_sorted or device == "cuda:0"
        ):
            # skip tests of Open3D implementation if Open3D is not installed or return_sorted is False or the device is
            # a GPU (the Open3D implementation does not offer a return_sorted parameter and only works on CPU)
            return

        if (  # pylint: disable=comparison-with-callable
            knn_search_implementation == knn_search_torch_cluster and not return_sorted
        ):
            # skip tests of torch-cluster implementation if return_sorted is False (the torch-cluster implementation
            # does not offer a return_sorted parameter)
            return

        if (  # pylint: disable=comparison-with-callable
            knn_search_implementation == knn_search_pytorch3d and not pytorch3d_is_available()
        ):
            # skip tests of PyTorch3D implementation if PyTorch3D is not installed
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

        batch_indices_query_points = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device)
        batch_indices_support_points = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        point_cloud_sizes_query_points = torch.tensor([2, 3], dtype=torch.long, device=device)
        point_cloud_sizes_support_points = torch.tensor([5, 6], dtype=torch.long, device=device)
        expected_neighbor_indices = np.array(
            [
                [0, 2, 1, 4, 3, 11],
                [4, 3, 1, 2, 0, 11],
                [5, 6, 7, 8, 9, 10],
                [7, 8, 9, 6, 10, 5],
                [10, 9, 8, 7, 6, 5],
            ]
        )
        expected_neighbor_indices = expected_neighbor_indices[:, :k]

        # compute expected neighbor distances
        max_neighbors = min(k, int(point_cloud_sizes_support_points.amax().item()))
        expanded_coords = np.expand_dims(coords_support_points.cpu().numpy(), axis=1)
        expanded_coords = np.tile(expanded_coords, (1, max_neighbors, 1))
        expanded_expected_neighbor_indices = np.expand_dims(expected_neighbor_indices, axis=-1)
        expanded_expected_neighbor_indices = np.tile(expanded_expected_neighbor_indices, (1, 1, 3))
        neighbor_coords = np.take_along_axis(expanded_coords, expanded_expected_neighbor_indices, axis=0)
        expected_neighbor_distances = np.linalg.norm(
            neighbor_coords - np.expand_dims(coords_query_points.cpu().numpy(), axis=1), axis=-1
        )

        # set arguments that differ between implementations
        if knn_search_implementation == knn_search_cdist:  # pylint: disable=comparison-with-callable
            variable_inputs = [point_cloud_sizes_support_points, point_cloud_sizes_query_points, k, return_sorted]
        elif knn_search_implementation == knn_search_open3d:  # pylint: disable=comparison-with-callable
            variable_inputs = [point_cloud_sizes_support_points, point_cloud_sizes_query_points, k]
        elif knn_search_implementation == knn_search_pytorch3d:  # pylint: disable=comparison-with-callable
            variable_inputs = [
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
                k,
                return_sorted,
            ]
        elif knn_search_implementation == knn_search_torch_cluster:  # pylint: disable=comparison-with-callable
            variable_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                k,
            ]
        else:
            variable_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
                k,
                return_sorted,
            ]

        neighbor_indices, neighbor_distances = knn_search_implementation(
            coords_support_points, coords_query_points, *variable_inputs
        )

        if not return_sorted:
            neighbor_distances, sorting_indices = torch.sort(neighbor_distances)
            neighbor_indices = torch.gather(neighbor_indices, -1, sorting_indices)

        assert expected_neighbor_indices.shape == neighbor_indices.shape
        assert expected_neighbor_distances.shape == neighbor_distances.shape

        np.testing.assert_array_equal(expected_neighbor_indices, neighbor_indices.cpu().numpy())
        np.testing.assert_almost_equal(neighbor_distances.cpu().numpy(), expected_neighbor_distances, decimal=4)

    @pytest.mark.parametrize("k", [2, 100])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_consistency(self, k: int, device: torch.device):  # pylint: disable=too-many-locals
        # Test with randomly distributed input points
        coords_support_points = torch.randn((11, 3), device=device)
        coords_query_points = torch.stack(
            [coords_support_points[0], coords_support_points[5], coords_support_points[6]]
        )

        batch_indices_query_points = torch.tensor([0, 1, 1], dtype=torch.long, device=device)
        batch_indices_support_points = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        point_cloud_sizes_query_points = torch.tensor([1, 2], dtype=torch.long, device=device)
        point_cloud_sizes_support_points = torch.tensor([5, 6], dtype=torch.long, device=device)

        ind_cdist, dists_cdist = knn_search_cdist(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            k,
        )

        ind_torch_cluster, dists_torch_cluster = knn_search_torch_cluster(
            coords_support_points,
            coords_query_points,
            batch_indices_support_points,
            batch_indices_query_points,
            point_cloud_sizes_support_points,
            k,
        )

        assert ind_cdist.size() == ind_torch_cluster.size()
        assert dists_cdist.size() == dists_torch_cluster.size()

        np.testing.assert_array_equal(ind_cdist.cpu().numpy(), ind_torch_cluster.cpu().numpy())
        np.testing.assert_almost_equal(dists_cdist.cpu().numpy(), dists_torch_cluster.cpu().numpy(), decimal=4)

        if pytorch3d_is_available():
            ind_pytorch3d, dists_pytorch3d = knn_search_pytorch3d(
                coords_support_points,
                coords_query_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
                k,
            )

            assert ind_cdist.size() == ind_pytorch3d.size()
            assert dists_cdist.size() == dists_pytorch3d.size()

            np.testing.assert_array_equal(ind_cdist.cpu().numpy(), ind_pytorch3d.cpu().numpy())
            np.testing.assert_almost_equal(dists_cdist.cpu().numpy(), dists_pytorch3d.cpu().numpy(), decimal=4)

        if open3d_is_available() and device != "cuda:0":
            ind_open3d, dists_open3d = knn_search_open3d(
                coords_support_points,
                coords_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
                k,
            )

            assert ind_cdist.size() == ind_open3d.size()
            assert dists_cdist.size() == dists_open3d.size()

            np.testing.assert_array_equal(ind_cdist.cpu().numpy(), ind_open3d.cpu().numpy())
            np.testing.assert_almost_equal(dists_cdist.cpu().numpy(), dists_open3d.cpu().numpy(), decimal=4)

    def test_invalid_batch_indices_support_points(self):
        with pytest.raises(ValueError):
            knn_search(
                torch.randn((10, 3), dtype=torch.float),
                torch.randn((5, 3), dtype=torch.float),
                torch.zeros(8, dtype=torch.long),
                torch.zeros(5, dtype=torch.long),
                torch.tensor([10], dtype=torch.long),
                torch.tensor([5], dtype=torch.long),
                k=1,
            )

    def test_invalid_batch_indices_query_points(self):
        with pytest.raises(ValueError):
            knn_search(
                torch.randn((10, 3), dtype=torch.float),
                torch.randn((5, 3), dtype=torch.float),
                torch.zeros(10, dtype=torch.long),
                torch.zeros(8, dtype=torch.long),
                torch.tensor([10], dtype=torch.long),
                torch.tensor([5], dtype=torch.long),
                k=1,
            )

    def test_invalid_point_cloud_sizes_support_points(self):
        with pytest.raises(ValueError):
            knn_search(
                torch.randn((10, 3), dtype=torch.float),
                torch.randn((5, 3), dtype=torch.float),
                torch.zeros(10, dtype=torch.long),
                torch.zeros(5, dtype=torch.long),
                torch.tensor([8], dtype=torch.long),
                torch.tensor([5], dtype=torch.long),
                k=1,
            )

    def test_invalid_point_cloud_sizes_query_points(self):
        with pytest.raises(ValueError):
            knn_search(
                torch.randn((10, 3), dtype=torch.float),
                torch.randn((5, 3), dtype=torch.float),
                torch.zeros(10, dtype=torch.long),
                torch.zeros(5, dtype=torch.long),
                torch.tensor([10], dtype=torch.long),
                torch.tensor([8], dtype=torch.long),
                k=1,
            )
