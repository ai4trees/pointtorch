"""Tests for pointtorch.operations.numpy.voxel_downsampling."""

from typing import Literal, Optional, Tuple

from hypothesis import given, strategies as st, settings
import numpy as np
import pytest

from pointtorch.operations.numpy import voxel_downsampling


class TestVoxelDownSampling:
    """Tests for pointtorch.operations.np.voxel_downsampling."""

    def _naive_voxel_downsampling(  # pylint: disable=too-many-locals
        self,
        points: np.ndarray,
        voxel_size: float,
        point_aggregation: Literal["nearest_neighbor"],
        start: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Naive implementation of voxel downsampling to compute expected results for arbitrary inputs.

        Returns:
            Expected results, i.e., the downsampled point cloud, the indices of the sampled points in the original
            point cloud, and the indices of the voxels to which each point in the original point cloud belongs.
        """

        if start is None:
            start_coords = np.array([0, 0, 0])
        else:
            start_coords = start

        shifted_points = points.copy()
        shifted_points[:, :3] -= start_coords
        min_voxel_indices = np.floor_divide(shifted_points, voxel_size).astype(np.int64).min(axis=0)
        max_voxel_indices = np.floor_divide(shifted_points, voxel_size).astype(np.int64).max(axis=0)

        selected_indices = []
        selected_points = []
        inverse_indices = np.empty(len(points), dtype=np.int64)

        indices = np.arange(len(points))

        idx = 0
        for x in range(min_voxel_indices[0], max_voxel_indices[0] + 1):
            for y in range(min_voxel_indices[1], max_voxel_indices[1] + 1):
                for z in range(min_voxel_indices[2], max_voxel_indices[2] + 1):
                    lower_voxel_border = np.array([[x, y, z]]) * voxel_size
                    voxel_center = lower_voxel_border + 0.5 * voxel_size
                    mask = np.logical_and(
                        ((shifted_points[:, :3]) >= lower_voxel_border).all(axis=1),
                        ((shifted_points[:, :3]) < (lower_voxel_border + voxel_size)).all(axis=1),
                    )

                    if mask.sum() == 0:
                        continue

                    inverse_indices[mask] = idx
                    idx += 1
                    shifted_points_in_voxel = shifted_points[mask]
                    points_in_voxel = points[mask]
                    indices_in_voxel = indices[mask]
                    if point_aggregation == "nearest_neighbor":
                        dists = np.linalg.norm(shifted_points_in_voxel[:, :3] - voxel_center, axis=1)
                        assert (dists <= np.linalg.norm([0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size])).all()
                        selected_idx = dists.argmin()
                    else:
                        raise ValueError("Invalid value for point_aggregation.")
                    selected_indices.append(indices_in_voxel[selected_idx])
                    selected_points.append(points_in_voxel[selected_idx])

        return np.array(selected_points), np.array(selected_indices), inverse_indices

    def _point_grid(self, grid_size: int) -> np.ndarray:
        """
        Creates a point cloud with the points being positioned on a regular grid. The created grid ranges from
        `-grid_size` to `+grid_size` along each axis.

        Args:
            grid_size: Number of grid steps along each axis.

        Returns: The coordinates of the generated point cloud.
        """

        x, y, z = np.meshgrid(
            np.arange(-grid_size, grid_size + 1),
            np.arange(-grid_size, grid_size + 1),
            np.arange(-grid_size, grid_size + 1),
        )
        coords = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1), np.expand_dims(z, -1)], axis=-1).reshape(
            (-1, 3)
        )

        return coords.astype(float)

    def test_voxel_downsampling_negative_voxel_size(self):
        points = np.random.uniform(low=-2, high=2, size=(50, 3))
        downsampled_points, downsampled_indices, inverse_indices = voxel_downsampling(points, -1)

        np.testing.assert_array_equal(np.unique(points, axis=0), np.unique(downsampled_points, axis=0))
        np.testing.assert_array_equal(np.arange(len(points)), np.sort(downsampled_indices))
        np.testing.assert_array_equal(np.arange(len(points)), inverse_indices)

    @pytest.mark.parametrize("point_aggregation", ["nearest_neighbor"])
    @pytest.mark.parametrize("preserve_order", [True, False])
    @pytest.mark.parametrize("start", [None, np.array([0.5, 0.5, 0.5])])
    @given(voxel_size=st.floats(min_value=0.01, max_value=10))
    @settings(deadline=None)
    def test_voxel_downsampling_random_input(
        self,
        voxel_size: float,
        point_aggregation: Literal["nearest_neighbor"],
        preserve_order: bool,
        start: Optional[np.ndarray],
    ):
        points = np.random.uniform(low=-2 * voxel_size, high=voxel_size * 2, size=(10, 3))

        expected_downsampled_points, _, expected_inverse_indices = self._naive_voxel_downsampling(
            points, voxel_size, "nearest_neighbor", start=start
        )

        downsampled_points, downsampled_indices, inverse_indices = voxel_downsampling(
            points, voxel_size, point_aggregation=point_aggregation, preserve_order=preserve_order, start=start
        )

        assert len(expected_downsampled_points) == len(downsampled_points)
        assert len(expected_downsampled_points) == len(downsampled_indices)
        assert len(points) == len(inverse_indices)

        np.testing.assert_array_equal(
            np.unique(expected_downsampled_points, axis=0), np.unique(downsampled_points, axis=0)
        )
        np.testing.assert_array_equal(downsampled_points, points[downsampled_indices])
        if preserve_order:
            np.testing.assert_array_equal(np.sort(downsampled_indices), downsampled_indices)
        np.testing.assert_array_equal(
            expected_downsampled_points[expected_inverse_indices], downsampled_points[inverse_indices]
        )

    @pytest.mark.parametrize("point_aggregation", ["nearest_neighbor", "random"])
    @pytest.mark.parametrize("preserve_order", [True, False])  # , False])
    @given(voxel_size=st.floats(min_value=0.001, max_value=10))
    # @given(voxel_size=st.floats(min_value=0.1, max_value=0.1))
    @settings(deadline=None)
    def test_voxel_downsampling_one_point_per_voxel(
        self, voxel_size: float, point_aggregation: Literal["nearest_neighbor", "random"], preserve_order: bool
    ):
        points = self._point_grid(1) * voxel_size

        downsampled_points, downsampled_indices, inverse_indices = voxel_downsampling(
            points, voxel_size, point_aggregation=point_aggregation, preserve_order=preserve_order
        )

        assert len(points) == len(downsampled_points)
        assert len(points) == len(downsampled_indices)
        np.testing.assert_array_equal(np.unique(points, axis=0), np.unique(downsampled_points, axis=0))
        np.testing.assert_array_equal(np.arange(len(points)), np.sort(downsampled_indices))
        if preserve_order:
            np.testing.assert_array_equal(np.sort(downsampled_indices), downsampled_indices)
        assert len(inverse_indices) == len(np.unique(inverse_indices))
        np.testing.assert_array_equal(points, downsampled_points[inverse_indices])

    @pytest.mark.parametrize("point_aggregation", ["nearest_neighbor", "random"])
    @pytest.mark.parametrize("preserve_order", [True, False])
    @given(voxel_size=st.floats(min_value=0.001, max_value=10))
    @settings(deadline=None)
    def test_voxel_downsampling_two_points_per_voxel(
        self, voxel_size: float, point_aggregation: Literal["nearest_neighbor", "random"], preserve_order: bool
    ):
        points = self._point_grid(1) * voxel_size
        duplicated_points = np.row_stack([points, points])

        downsampled_points, downsampled_indices, inverse_indices = voxel_downsampling(
            duplicated_points, voxel_size, point_aggregation=point_aggregation, preserve_order=preserve_order
        )

        assert len(points) == len(downsampled_points)
        assert len(points) == len(downsampled_indices)
        np.testing.assert_array_equal(np.unique(points, axis=0), np.unique(downsampled_points, axis=0))
        np.testing.assert_array_equal(downsampled_points, duplicated_points[downsampled_indices])
        if preserve_order:
            np.testing.assert_array_equal(np.sort(downsampled_indices), downsampled_indices)
        np.testing.assert_array_equal(duplicated_points, downsampled_points[inverse_indices])

    def test_voxel_downsampling_invalid_start(self):
        start = np.array([1.1, 0.0])
        points = np.zeros((20, 3))
        voxel_size = 1

        with pytest.raises(ValueError):
            voxel_downsampling(points, voxel_size, start=start)
