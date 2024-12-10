""" Tests for the pointtorch.core.PointCloud class. """

import pathlib
import os
import shutil
from typing import Union

import numpy as np
import pytest

from pointtorch import PointCloud, PointCloudSeries, read


class TestPointCloud:
    """
    Tests for the pointtorch.core.PointCloud class.
    """

    @pytest.fixture
    def data(self):
        return np.array([[0, 0, 0, 1], [1, 1, 1, 2]])

    @pytest.fixture
    def identifier(self):
        return "test_point_cloud"

    @pytest.fixture
    def point_cloud(self, data, identifier):
        return PointCloud(data, columns=["x", "y", "z", "intensity"], identifier=identifier)

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/core/TestPointCloud"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    def test_xyz(self, data, point_cloud):
        assert (data[:, :3] == point_cloud.xyz()).all()

    def test_xyz_invalid(self, point_cloud):
        point_cloud = point_cloud.drop("x", axis=1)
        with pytest.raises(RuntimeError):
            point_cloud.xyz()

    def test_slicing(self, point_cloud, identifier):
        point_cloud_slice = point_cloud[["x", "y"]]
        assert isinstance(point_cloud_slice, PointCloud)
        assert identifier == point_cloud_slice.identifier

    def test_series_slicing(self, point_cloud, identifier):
        point_cloud_slice = point_cloud["x"]
        assert isinstance(point_cloud_slice, PointCloudSeries)
        assert identifier == point_cloud_slice.identifier

    def test_identifier(self, point_cloud, identifier):
        assert identifier == point_cloud.identifier

    def test_x_max_resolution(self, point_cloud):
        x_max_resolution = 0.1
        point_cloud = PointCloud(point_cloud, x_max_resolution=x_max_resolution)
        assert x_max_resolution == point_cloud.x_max_resolution

    def test_y_max_resolution(self, point_cloud):
        y_max_resolution = 0.1
        point_cloud = PointCloud(point_cloud, y_max_resolution=y_max_resolution)
        assert y_max_resolution == point_cloud.y_max_resolution

    def test_z_max_resolution(self, point_cloud):
        z_max_resolution = 0.1
        point_cloud = PointCloud(point_cloud, z_max_resolution=z_max_resolution)
        assert z_max_resolution == point_cloud.z_max_resolution

    @pytest.mark.parametrize("file_format", ["csv", "txt", "h5", "hdf", "las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_to(self, file_format: str, use_pathlib: bool, cache_dir, point_cloud):
        point_cloud = PointCloud(point_cloud)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud.to(file_path)

        read_point_cloud_data = read(file_path)

        assert (point_cloud.to_numpy() == read_point_cloud_data.to_numpy()).all()
