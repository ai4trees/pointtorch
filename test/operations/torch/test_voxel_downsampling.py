"""Tests for pointtorch.operations.torch.voxel_downsampling."""

from typing import Literal, Optional, Tuple

from hypothesis import given, strategies as st, settings
import numpy as np
import pytest
import torch

from pointtorch.operations.torch import voxel_downsampling
from pointtorch.type_aliases import FloatArray, LongArray


class TestSampling:
    """Tests for pointtorch.operations.torch.voxel_downsampling."""

    def naive_voxel_downsampling(  # pylint: disable=too-many-locals, too-many-branches
        self,
        coords: torch.Tensor,
        point_cloud_sizes: torch.Tensor,
        voxel_size: float,
        features: Optional[torch.Tensor] = None,
        point_aggregation: Literal["mean", "nearest_neighbor"] = "mean",
        feature_aggregation: Literal["max", "mean", "min", "nearest_neighbor"] = "mean",
        start: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Naive implementation of voxel downsampling to compute expected results for arbitrary inputs.

        Returns:
            Expected results, i.e., coordinates, features, batch indices and point cloud sizes of the downsampled point
            clouds.
        """

        device = coords.device

        downsampled_coords = []
        downsampled_features = []
        downsampled_batch_indices = []
        downsampled_point_cloud_sizes = []

        if start is None:
            start_coords = torch.zeros((len(point_cloud_sizes), 3), dtype=torch.float, device=device)
        else:
            start_coords = start

        start_idx = 0
        for batch_idx, point_cloud_size in enumerate(point_cloud_sizes):
            downsampled_point_cloud_size = 0
            end_idx = start_idx + int(point_cloud_size.item())
            current_coords = coords[start_idx:end_idx]

            shifted_coords = current_coords - start_coords[batch_idx]

            min_voxel_indices = torch.floor_divide(shifted_coords, voxel_size).long().amin(dim=0)
            max_voxel_indices = torch.floor_divide(shifted_coords, voxel_size).long().amax(dim=0)

            for x in range(min_voxel_indices[0], max_voxel_indices[0] + 1):
                for y in range(min_voxel_indices[1], max_voxel_indices[1] + 1):
                    for z in range(min_voxel_indices[2], max_voxel_indices[2] + 1):
                        lower_voxel_border = torch.tensor([[x, y, z]], device=device, dtype=torch.float) * voxel_size
                        voxel_center = lower_voxel_border + 0.5 * voxel_size
                        mask = torch.logical_and(
                            ((shifted_coords) >= lower_voxel_border).all(dim=1),
                            ((shifted_coords) < (lower_voxel_border + voxel_size)).all(dim=1),
                        )
                        if mask.sum() == 0:
                            continue
                        downsampled_batch_indices.append(batch_idx)
                        downsampled_point_cloud_size += 1

                        coords_in_voxel = current_coords[mask]
                        shifted_coords_in_voxel = shifted_coords[mask]
                        if point_aggregation == "nearest_neighbor":
                            dists = torch.linalg.norm(  # pylint: disable=not-callable
                                shifted_coords_in_voxel - voxel_center, dim=1
                            )
                            downsampled_coords.append(coords_in_voxel[dists.argmin()])
                        elif point_aggregation == "mean":
                            downsampled_coords.append(coords_in_voxel.mean(dim=0))
                        else:
                            raise ValueError("Invalid value for point_aggregation.")

                        if features is None:
                            continue

                        current_features = features[start_idx:end_idx]
                        features_in_voxel = current_features[mask]
                        if feature_aggregation == "nearest_neighbor":
                            dists = torch.linalg.norm(  # pylint: disable=not-callable
                                shifted_coords_in_voxel - voxel_center, dim=1
                            )
                            downsampled_features.append(features_in_voxel[dists.argmin()])
                        elif feature_aggregation == "max":
                            downsampled_features.append(features_in_voxel.amax(dim=0))
                        elif feature_aggregation == "mean":
                            downsampled_features.append(features_in_voxel.mean(dim=0))
                        elif feature_aggregation == "min":
                            downsampled_features.append(features_in_voxel.amin(dim=0))
                        else:
                            raise ValueError("Invalid value for feature_aggregation.")

            downsampled_point_cloud_sizes.append(downsampled_point_cloud_size)
            start_idx = end_idx

        return (
            torch.row_stack(downsampled_coords),
            torch.row_stack(downsampled_features) if features is not None else None,
            torch.tensor(downsampled_batch_indices, dtype=torch.long, device=device),
            torch.tensor(downsampled_point_cloud_sizes, dtype=torch.long, device=device),
        )

    def sort_results(
        self,
        downsampled_coords: FloatArray,
        downsampled_features: FloatArray,
        downsampled_batch_indices: LongArray,
    ) -> Tuple[FloatArray, FloatArray]:
        """Sorts downsampled points to allow for comparisons."""

        downsampled_coords = downsampled_coords.copy()
        downsampled_features = downsampled_features.copy()

        for batch_index in np.unique(downsampled_batch_indices):
            # numpy unique returns the elements in sorted order
            downsampled_coords[downsampled_batch_indices == batch_index], sorted_index = np.unique(
                downsampled_coords[downsampled_batch_indices == batch_index], axis=0, return_index=True
            )
            downsampled_features[downsampled_batch_indices == batch_index] = downsampled_features[
                downsampled_batch_indices == batch_index
            ][sorted_index]

        return downsampled_coords, downsampled_features

    @pytest.mark.parametrize("point_aggregation", ["mean", "nearest_neighbor"])
    @pytest.mark.parametrize("feature_aggregation", ["max", "mean", "min", "nearest_neighbor"])
    @pytest.mark.parametrize("preserve_order", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    @given(voxel_size=st.floats(min_value=0.01, max_value=10))
    @settings(deadline=None)
    def test_voxel_downsampling_random_input(  # pylint: disable=too-many-locals
        self,
        point_aggregation: Literal["mean", "nearest_neighbor"],
        feature_aggregation: Literal["max", "mean", "min", "nearest_neighbor"],
        preserve_order: bool,
        device: torch.device,
        voxel_size: float,
    ):
        coords = torch.rand((50, 3), device=device, dtype=torch.float) * 4 * voxel_size - 2 * voxel_size
        features = torch.randn((50, 5), device=device, dtype=torch.float) * 10
        batch_indices = torch.tensor([0] * 25 + [1] * 25, dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([25, 25], dtype=torch.long, device=device)

        expected_result = self.naive_voxel_downsampling(
            coords,
            point_cloud_sizes,
            voxel_size,
            features=features,
            feature_aggregation=feature_aggregation,
            point_aggregation=point_aggregation,
        )
        (
            exp_downsampled_coords,
            exp_downsampled_feat,
            exp_downsampled_batch_indices,
            exp_downsampled_pc_sizes,
        ) = expected_result

        result = voxel_downsampling(
            coords,
            batch_indices,
            point_cloud_sizes,
            voxel_size=voxel_size,
            features=features,
            feature_aggregation=feature_aggregation,
            point_aggregation=point_aggregation,
            preserve_order=preserve_order,
        )
        downsampled_coords, downsampled_feat, downsampled_batch_indices, downsampled_pc_sizes, _ = result

        exp_downsampled_coords_np, exp_downsampled_feat_np = self.sort_results(
            exp_downsampled_coords.cpu().numpy(),
            exp_downsampled_feat.cpu().numpy(),  # type: ignore[union-attr]
            exp_downsampled_batch_indices.cpu().numpy(),
        )
        downsampled_coords_np, downsampled_feat_np = self.sort_results(
            downsampled_coords.cpu().numpy(),
            downsampled_feat.cpu().numpy(),  # type: ignore[union-attr]
            downsampled_batch_indices.cpu().numpy(),
        )
        downsampled_batch_indices_np = downsampled_batch_indices.cpu().numpy()

        assert exp_downsampled_coords_np.shape == downsampled_coords_np.shape

        np.testing.assert_array_equal(exp_downsampled_batch_indices.cpu().numpy(), downsampled_batch_indices_np)
        np.testing.assert_almost_equal(exp_downsampled_coords_np, downsampled_coords_np, decimal=5)
        np.testing.assert_almost_equal(exp_downsampled_feat_np, downsampled_feat_np, decimal=5)
        np.testing.assert_array_equal(exp_downsampled_pc_sizes.cpu().numpy(), downsampled_pc_sizes.cpu().numpy())
        if preserve_order and point_aggregation == "nearest_neighbor":
            coords_np = coords.cpu().numpy()
            batch_indices_np = batch_indices.cpu().numpy()
            is_selected = np.zeros(len(coords), dtype=bool)
            idx = 0
            for batch_idx in np.unique(batch_indices_np):
                for coord in coords_np[batch_indices_np == batch_idx]:
                    if coord in downsampled_coords_np[downsampled_batch_indices_np == batch_idx]:
                        is_selected[idx] = True
                    idx += 1
            np.testing.assert_almost_equal(coords_np[is_selected], downsampled_coords.cpu().numpy(), decimal=5)

    @pytest.mark.parametrize("feature_aggregation", ["max", "mean", "min", "nearest_neighbor"])
    @pytest.mark.parametrize("point_aggregation", ["mean", "nearest_neighbor"])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_voxel_downsampling_small_example_input(  # pylint: disable=too-many-locals
        self,
        point_aggregation: Literal["mean", "nearest_neighbor"],
        feature_aggregation: Literal["max", "mean", "min", "nearest_neighbor"],
        device: torch.device,
    ):

        coords = torch.tensor(
            [
                [0, 0.5, 0],
                [0.5, 0, 0],
                [1, 2.5, 0],
                [1, 1, 0],
                [2.5, 1, 0],
                [3, 1, 0],
                [3, 0, 0],
                [4.5, 1, 0],
                [4, 2, 0],
                [5.5, 1, 0],
            ],
            dtype=torch.float,
            device=device,
        )
        coords = torch.cat([coords, coords])

        features = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]],
            dtype=torch.float,
            device=device,
        )
        features = torch.cat([features, features])

        batch_indices = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([10, 10], dtype=torch.long, device=device)

        if point_aggregation == "mean":
            exp_downsampled_coords = np.array(
                [
                    [1, 1, 0],
                    [4, 1, 0],
                    [1, 1, 0],
                    [4, 1, 0],
                ]
            )
        elif point_aggregation == "nearest_neighbor":
            exp_downsampled_coords = np.array(
                [
                    [1, 1, 0],
                    [4.5, 1, 0],
                    [1, 1, 0],
                    [4.5, 1, 0],
                ]
            )

        if feature_aggregation == "max":
            exp_downsampled_features = np.array(
                [
                    [4, 5],
                    [9, 10],
                    [4, 5],
                    [9, 10],
                ]
            )
        elif feature_aggregation == "mean":
            exp_downsampled_features = np.array(
                [
                    [2, 3],
                    [7, 8],
                    [2, 3],
                    [7, 8],
                ]
            )
        elif feature_aggregation == "min":
            exp_downsampled_features = np.array(
                [
                    [0, 1],
                    [5, 6],
                    [0, 1],
                    [5, 6],
                ]
            )
        elif feature_aggregation == "nearest_neighbor":
            exp_downsampled_features = np.array(
                [
                    [3, 4],
                    [7, 8],
                    [3, 4],
                    [7, 8],
                ]
            )

        exp_voxel_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

        exp_downsampled_batch_indices = np.array([0, 0, 1, 1])
        exp_downsampled_point_cloud_sizes = np.array([2, 2])

        result = voxel_downsampling(
            coords,
            batch_indices,
            point_cloud_sizes,
            voxel_size=3,
            features=features,
            feature_aggregation=feature_aggregation,
            point_aggregation=point_aggregation,
        )
        downsampled_coords, downsampled_feat, downsampled_batch_indices, downsampled_pc_sizes, voxel_indices = result

        exp_downsampled_coords, exp_downsampled_features = self.sort_results(
            exp_downsampled_coords,  # pylint: disable=used-before-assignment
            exp_downsampled_features,  # pylint: disable=used-before-assignment
            exp_downsampled_batch_indices,
        )
        downsampled_coords_np, downsampled_feat_np = self.sort_results(
            downsampled_coords.cpu().numpy(),
            downsampled_feat.cpu().numpy(),  # type: ignore[union-attr]
            downsampled_batch_indices.cpu().numpy(),
        )

        assert exp_downsampled_coords.shape == downsampled_coords.shape

        np.testing.assert_array_equal(exp_downsampled_batch_indices, downsampled_batch_indices.cpu().numpy())
        np.testing.assert_array_equal(exp_downsampled_coords, downsampled_coords_np)
        np.testing.assert_array_equal(exp_downsampled_features, downsampled_feat_np)
        np.testing.assert_array_equal(exp_downsampled_point_cloud_sizes, downsampled_pc_sizes.cpu().numpy())
        np.testing.assert_array_equal(exp_voxel_indices, voxel_indices.cpu().numpy())

    @pytest.mark.parametrize("start", [np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0], [1, 0.0]])])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_voxel_downsampling_invalid_start(self, start: FloatArray, device: torch.device):
        coords = torch.zeros((20, 3), dtype=torch.float, device=device)
        batch_indices = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([10, 10], dtype=torch.long, device=device)

        with pytest.raises(ValueError):
            voxel_downsampling(coords, batch_indices, point_cloud_sizes, 1, start=torch.from_numpy(start).to(device))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_voxel_downsampling_valid_start(self, device: torch.device):  # pylint: disable=too-many-locals

        start = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float, device=device)

        coords = torch.tensor(
            [
                [0, 0.5, 0],
                [0.5, 0, 0],
                [1, 2, 0],
                [1, 1, 0],
                [2, 1, 0],
                [3, 1, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 2, 0],
                [6, 1, 0],
            ],
            dtype=torch.float,
            device=device,
        )
        coords = torch.cat([coords, coords])

        features = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]],
            dtype=torch.float,
            device=device,
        )
        features = torch.cat([features, features])

        batch_indices = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([10, 10], dtype=torch.long, device=device)

        exp_downsampled_coords = np.array(
            [
                [0.25, 0.25, 0],
                [2, 1, 0],
                [5, 1, 0],
                [0.25, 0.25, 0],
                [2, 1, 0],
                [5, 1, 0],
            ]
        )
        exp_downsampled_features = np.array(
            [
                [0.5, 1.5],
                [4, 5],
                [8, 9],
                [0.5, 1.5],
                [4, 5],
                [8, 9],
            ]
        )

        exp_voxel_indices = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5])

        exp_downsampled_batch_indices = np.array([0, 0, 0, 1, 1, 1])
        exp_downsampled_point_cloud_sizes = np.array([3, 3])

        result = voxel_downsampling(
            coords, batch_indices, point_cloud_sizes, voxel_size=3, features=features, start=start
        )
        downsampled_coords, downsampled_feat, downsampled_batch_indices, downsampled_pc_sizes, voxel_indices = result

        exp_downsampled_coords, exp_downsampled_features = self.sort_results(
            exp_downsampled_coords, exp_downsampled_features, exp_downsampled_batch_indices
        )
        downsampled_coords_np, downsampled_feat_np = self.sort_results(
            downsampled_coords.cpu().numpy(),
            downsampled_feat.cpu().numpy(),  # type: ignore[union-attr]
            downsampled_batch_indices.cpu().numpy(),
        )

        assert exp_downsampled_coords.shape == downsampled_coords.shape

        np.testing.assert_array_equal(exp_downsampled_batch_indices, downsampled_batch_indices.cpu().numpy())
        np.testing.assert_array_equal(exp_downsampled_coords, downsampled_coords_np)
        np.testing.assert_array_equal(exp_downsampled_features, downsampled_feat_np)
        np.testing.assert_array_equal(exp_downsampled_point_cloud_sizes, downsampled_pc_sizes.cpu().numpy())
        np.testing.assert_array_equal(exp_voxel_indices, voxel_indices.cpu().numpy())

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    def test_voxel_indices(self, device: torch.device):
        num_points = 1024
        voxel_size = 3
        coords = torch.rand((num_points, 3), device=device, dtype=torch.float) * 4 * voxel_size - 2 * voxel_size
        batch_indices = torch.tensor(
            [0] * int(num_points / 2) + [1] * int(num_points / 2), device=device, dtype=torch.long
        )  # torch.randint(0, 10, (num_points,), dtype=torch.long, device=device)
        point_cloud_sizes = torch.tensor([int(num_points / 2), int(num_points / 2)], device=device, dtype=torch.long)

        downsampled_coords, _, downsampled_batch_indices, _, voxel_indices = voxel_downsampling(
            coords, batch_indices, point_cloud_sizes, voxel_size=3, feature_aggregation="mean", point_aggregation="mean"
        )

        # test that batch indices are sorted
        sorted_downsampled_batch_indices, _ = torch.sort(downsampled_batch_indices)

        np.testing.assert_array_equal(
            sorted_downsampled_batch_indices.cpu().numpy(), downsampled_batch_indices.cpu().numpy()
        )

        # test that voxel indices are in range
        assert voxel_indices.amax().item() < len(downsampled_coords)
