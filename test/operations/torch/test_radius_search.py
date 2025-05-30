"""Tests for the radius search implementations in pointtorch.operations"""

from typing import Callable, Optional

from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.typing as npt
import pytest
import torch

from pointtorch.config import open3d_is_available, pytorch3d_is_available
from pointtorch.operations.torch import (
    radius_search,
    radius_search_cdist,
    radius_search_open3d,
    radius_search_pytorch3d,
    radius_search_torch_cluster,
)
from pointtorch.operations.numpy import voxel_downsampling
from pointtorch.type_aliases import LongArray


class TestRadiusSearch:
    """Tests for the radius search implementations in pointtorch.operations"""

    @staticmethod
    def _naive_radius_search(  # pylint: disable=too-many-locals
        coords_support_points: torch.Tensor,
        coords_query_points: torch.Tensor,
        point_cloud_sizes_support_points: torch.Tensor,
        point_cloud_sizes_query_points: torch.Tensor,
        radius: float,
        k: Optional[int],
        return_sorted: bool = False,
    ) -> LongArray:
        """
        Naive implementation of radius search to compute expected results for arbitrary inputs.

        Returns: Expected neighbor indices.
        """

        coords_support_points = coords_support_points.cpu().numpy()
        coords_query_points = coords_query_points.cpu().numpy()
        point_cloud_sizes_support_points = point_cloud_sizes_support_points.cpu().numpy()
        point_cloud_sizes_query_points = point_cloud_sizes_query_points.cpu().numpy()

        all_neighbor_indices = []
        min_support_point = 0
        max_support_point = point_cloud_sizes_support_points[0]
        batch_idx = 0

        for q_idx, point in enumerate(coords_query_points):
            neighbor_indices = []
            neighbor_dists = []
            max_dist = float("inf")
            if q_idx == point_cloud_sizes_query_points[batch_idx]:
                min_support_point = int(point_cloud_sizes_support_points[batch_idx].item())
                max_support_point = point_cloud_sizes_support_points[batch_idx + 1]
                batch_idx += 1
            for s_idx, support_point in enumerate(coords_support_points[min_support_point:max_support_point, :]):
                dist = np.linalg.norm(point - support_point)
                if dist > radius:
                    continue
                if k is None:
                    neighbor_indices.append(s_idx)
                else:
                    if return_sorted and dist < max_dist:
                        neighbor_indices.append(s_idx)
                        neighbor_dists.append(dist)
                        sorted_indices = np.argsort(neighbor_dists)
                        neighbor_indices = list(np.array(neighbor_indices)[sorted_indices])[:k]
                        neighbor_dists = list(np.array(neighbor_dists)[sorted_indices])[:k]
                        if len(neighbor_dists) == k:
                            max_dist = float(neighbor_dists[-1])
                    if not return_sorted and len(neighbor_indices) < k:
                        neighbor_indices.append(s_idx)
            all_neighbor_indices.append(neighbor_indices)

        max_neighbors = max(len(neighbor_indices) for neighbor_indices in all_neighbor_indices)

        invalid_neighbor_index = len(coords_support_points)
        all_neighbor_indices_np = np.full((len(coords_query_points), max_neighbors), fill_value=invalid_neighbor_index)

        for idx, neighbor_indices in enumerate(all_neighbor_indices):
            all_neighbor_indices_np[idx, : len(neighbor_indices)] = np.array(neighbor_indices)

        return all_neighbor_indices_np

    @pytest.mark.parametrize(
        "radius_search_implementation",
        [
            radius_search,
            radius_search_cdist,
            radius_search_open3d,
            radius_search_pytorch3d,
            radius_search_torch_cluster,
        ],
    )
    @pytest.mark.parametrize("return_sorted", [True, False])
    @pytest.mark.parametrize("pass_voxel_size", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    @given(
        num_query_points=st.integers(min_value=1, max_value=50),
        k=st.one_of(st.none(), st.integers(min_value=1, max_value=50)),
        radius=st.floats(min_value=0.01, max_value=10),
    )
    @settings(deadline=None)
    def test_radius_search_random_inputs(  # pylint: disable=too-many-locals
        self,
        radius_search_implementation: Callable,
        num_query_points: int,
        k: Optional[int],
        radius: float,
        return_sorted: bool,
        pass_voxel_size: bool,
        device: str,
    ):

        if (  # pylint: disable=comparison-with-callable
            radius_search_implementation == radius_search_open3d and not open3d_is_available()
        ):
            # skip tests of Open3D implementation if PyTorch3D is not installed
            return

        if (  # pylint: disable=comparison-with-callable
            radius_search_implementation == radius_search_pytorch3d and not pytorch3d_is_available()
        ):
            # skip tests of PyTorch3D implementation if PyTorch3D is not installed
            return

        coords_query_points_np = np.random.random((num_query_points, 3)) * 10
        coords_support_points_np = np.random.random((num_query_points * 2, 3)) * 10

        voxel_size = 0.05

        coords_query_points_np, _, _ = voxel_downsampling(coords_query_points_np, voxel_size)
        coords_support_points_np, _, _ = voxel_downsampling(coords_support_points_np, voxel_size)

        coords_query_points = torch.from_numpy(coords_query_points_np).to(device)
        coords_support_points = torch.from_numpy(coords_support_points_np).to(device)

        batch_indices_query_points = torch.zeros(len(coords_query_points), dtype=torch.long, device=device)
        batch_indices_support_points = torch.zeros(len(coords_support_points), dtype=torch.long, device=device)
        point_cloud_sizes_query_points = torch.tensor([len(coords_query_points)], dtype=torch.long, device=device)
        point_cloud_sizes_support_points = torch.tensor([len(coords_support_points)], dtype=torch.long, device=device)

        if radius_search_implementation in [radius_search_cdist, radius_search_open3d, radius_search_pytorch3d]:
            batch_idx_inputs = [point_cloud_sizes_support_points, point_cloud_sizes_query_points]
        elif radius_search_implementation in [radius_search_torch_cluster]:
            batch_idx_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
            ]
        else:
            batch_idx_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
            ]

        if (  # pylint: disable=comparison-with-callable
            radius_search_implementation != radius_search_open3d and pass_voxel_size
        ):
            kwargs = {"voxel_size": voxel_size}
        else:
            kwargs = {}

        neighbor_indices = (
            radius_search_implementation(
                coords_support_points,
                coords_query_points,
                *batch_idx_inputs,
                radius,
                k=k,
                return_sorted=return_sorted,
                **kwargs,
            )
            .cpu()
            .numpy()
        )

        expected_neighbor_indices = self._naive_radius_search(
            coords_support_points,
            coords_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            radius,
            k=k,
            return_sorted=return_sorted,
        )

        assert expected_neighbor_indices.shape == neighbor_indices.shape

        if k is None:
            np.testing.assert_array_equal(
                np.sort(expected_neighbor_indices, axis=-1), np.sort(neighbor_indices, axis=-1)
            )

    @pytest.mark.parametrize(
        "radius_search_implementation",
        [
            radius_search,
            radius_search_cdist,
            radius_search_open3d,
            radius_search_pytorch3d,
            radius_search_torch_cluster,
        ],
    )
    @pytest.mark.parametrize("return_sorted", [True, False])
    @pytest.mark.parametrize("pass_voxel_size", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    @given(k=st.one_of(st.none(), st.integers(min_value=1, max_value=15)))
    @settings(deadline=None)
    def test_radius_search_small_example_input(  # pylint: disable=too-many-locals
        self,
        radius_search_implementation: Callable,
        k: Optional[int],
        return_sorted: bool,
        pass_voxel_size: bool,
        device: str,
    ):

        if (  # pylint: disable=comparison-with-callable
            radius_search_implementation == radius_search_open3d and not open3d_is_available()
        ):
            # skip tests of Open3D implementation if PyTorch3D is not installed
            return

        if (  # pylint: disable=comparison-with-callable
            radius_search_implementation == radius_search_pytorch3d and not pytorch3d_is_available()
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
                [4, 0, 0],
                [8, 2, 0],
                [8, 3, 0],
                [9, 3, 0],
                [10, 3, 0],
            ],
            dtype=torch.float,
            device=device,
        )

        voxel_size = 1

        batch_indices_query_points = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device)
        batch_indices_support_points = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        point_cloud_sizes_query_points = torch.tensor([2, 3], dtype=torch.long, device=device)
        point_cloud_sizes_support_points = torch.tensor([5, 6], dtype=torch.long, device=device)

        expected_neighbor_indices = np.array(
            [
                [0, 2, 11, 11, 11, 11],
                [4, 3, 1, 11, 11, 11],
                [5, 6, 11, 11, 11, 11],
                [7, 8, 9, 6, 10, 5],
                [10, 9, 8, 7, 11, 11],
            ]
        )

        invalid_neighbor_index = 11
        radius = 5

        if radius_search_implementation in [radius_search_cdist, radius_search_open3d, radius_search_pytorch3d]:
            batch_idx_inputs = [point_cloud_sizes_support_points, point_cloud_sizes_query_points]
        elif radius_search_implementation in [radius_search_torch_cluster]:
            batch_idx_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
            ]
        else:
            batch_idx_inputs = [
                batch_indices_support_points,
                batch_indices_query_points,
                point_cloud_sizes_support_points,
                point_cloud_sizes_query_points,
            ]

        if (  # pylint: disable=comparison-with-callable
            pass_voxel_size and radius_search_implementation != radius_search_open3d
        ):
            kwargs = {"voxel_size": voxel_size}
        else:
            kwargs = {}

        neighbor_indices = radius_search_implementation(
            coords_support_points,
            coords_query_points,
            *batch_idx_inputs,
            radius,
            k=k,
            return_sorted=return_sorted,
            **kwargs,
        )
        neighbor_indices_np = neighbor_indices.cpu().numpy()

        if return_sorted:
            if k is not None:
                expected_neighbor_indices = expected_neighbor_indices[:, :k]

            assert expected_neighbor_indices.shape == neighbor_indices.shape
            np.testing.assert_array_equal(expected_neighbor_indices, neighbor_indices_np)
        else:
            if k is not None:
                for idx, row in enumerate(expected_neighbor_indices):
                    valid_indices = (neighbor_indices_np[idx] < invalid_neighbor_index).sum()
                    expected_valid_indices = (row < invalid_neighbor_index).sum()

                    assert k >= len(neighbor_indices_np[idx])
                    assert np.isin(neighbor_indices_np[idx], row).all()
                    assert valid_indices == min(k, expected_valid_indices)
            else:
                assert expected_neighbor_indices.shape == neighbor_indices.shape
                np.testing.assert_array_equal(
                    np.sort(expected_neighbor_indices, axis=-1), np.sort(neighbor_indices_np, axis=-1)
                )

    @pytest.mark.parametrize(
        "radius_search_implementation",
        [
            radius_search_cdist,
            radius_search_pytorch3d,
            radius_search_torch_cluster,
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_radius_search_voxel_optimization(self, radius_search_implementation: Callable, device: str):

        if (  # pylint: disable=comparison-with-callable
            radius_search_implementation == radius_search_pytorch3d and not pytorch3d_is_available()
        ):
            # skip tests of PyTorch3D implementation if PyTorch3D is not installed
            return

        x_coords, y_coords, z_coords = torch.meshgrid(
            torch.arange(20), torch.arange(20), torch.arange(20), indexing="ij"
        )

        x_coords = x_coords.reshape(-1)
        y_coords = y_coords.reshape(-1)
        z_coords = z_coords.reshape(-1)

        coords = torch.stack([x_coords, y_coords, z_coords], dim=1).to(device)
        # scale coordinates so that distance between points is 2cm
        coords = coords / 50 + 0.01
        batch_indices = torch.zeros(len(coords), dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([len(coords)], device=device)

        radius = 0.04

        if radius_search_implementation in [radius_search_cdist, radius_search_pytorch3d]:
            batch_idx_inputs = [point_cloud_sizes, point_cloud_sizes]
        elif radius_search_implementation in [radius_search_torch_cluster]:
            batch_idx_inputs = [batch_indices, batch_indices, point_cloud_sizes]
        else:
            batch_idx_inputs = [batch_indices, batch_indices, point_cloud_sizes, point_cloud_sizes]

        neighbor_indices = radius_search_implementation(
            coords, coords, *batch_idx_inputs, radius, k=None, return_sorted=True, voxel_size=0.02
        )

        expected_max_num_neighbors = 33
        assert expected_max_num_neighbors == neighbor_indices.size(1)

    @pytest.mark.parametrize(
        "radius_search_implementation",
        [
            radius_search_cdist,
            radius_search_torch_cluster,
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_radius_search_cdist_too_large_input(self, radius_search_implementation: Callable, device: str):
        num_points = 10**8
        k = 10**4
        coords = torch.randn((num_points, 3), device=device, dtype=torch.float)
        batch_indices = torch.zeros((num_points,), device=device, dtype=torch.long)
        point_cloud_sizes = torch.tensor([num_points], device=device, dtype=torch.long)
        radius = 2.5

        if radius_search_implementation in [radius_search_cdist]:
            batch_idx_inputs = [point_cloud_sizes, point_cloud_sizes]
        else:
            batch_idx_inputs = [batch_indices, batch_indices, point_cloud_sizes]

        with pytest.raises(ValueError):
            radius_search_implementation(coords, coords, *batch_idx_inputs, radius, k=k)
