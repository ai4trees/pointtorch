"""Tests for the pointtorch.core.PointCloud class."""

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

    def test_structured_dtype(self, point_cloud):
        structured_dtype = point_cloud.structured_dtype()

        assert ("x", "y", "z", "intensity") == structured_dtype.names
        assert point_cloud["x"].dtype == structured_dtype["x"]
        assert point_cloud["intensity"].dtype == structured_dtype["intensity"]

    def test_structured_dtype_with_extra_fields(self, point_cloud):
        structured_dtype = point_cloud.structured_dtype(extra_fields=[("instance", np.int32)])

        assert ("x", "y", "z", "intensity", "instance") == structured_dtype.names
        assert np.dtype(np.int32) == structured_dtype["instance"]

    def test_to_structured_array(self, point_cloud):
        structured_array = point_cloud.to_structured_array()

        assert ("x", "y", "z", "intensity") == structured_array.dtype.names
        assert point_cloud["x"].dtype == structured_array.dtype["x"]
        assert point_cloud["intensity"].dtype == structured_array.dtype["intensity"]

        for column in point_cloud.columns:
            np.testing.assert_array_equal(point_cloud[column].to_numpy(), structured_array[column])

    def test_to_structured_array_custom_dtype(self, point_cloud):
        structured_dtype = np.dtype([("x", np.float32), ("z", np.float32), ("instance", np.int32)])
        structured_array = point_cloud.to_structured_array(dtype=structured_dtype)

        assert structured_dtype == structured_array.dtype
        np.testing.assert_array_equal(point_cloud["x"].to_numpy(dtype=np.float32), structured_array["x"])
        np.testing.assert_array_equal(point_cloud["z"].to_numpy(dtype=np.float32), structured_array["z"])

    def test_from_structured_array(self):
        structured_array = np.array(
            [(0.0, 0.0, 0.0, 1), (1.0, 1.0, 1.0, 2)],
            dtype=[("x", np.float64), ("y", np.float64), ("z", np.float64), ("intensity", np.int32)],
        )

        epsg_code = "EPSG:4326"
        point_cloud_identifier = "test_point_cloud"

        point_cloud = PointCloud.from_structured_array(
            structured_array,
            crs=epsg_code,
            identifier=point_cloud_identifier,
            x_max_resolution=0.1,
            y_max_resolution=0.1,
            z_max_resolution=0.01,
        )

        assert isinstance(point_cloud, PointCloud)
        assert list(structured_array.dtype.names) == list(point_cloud.columns)

        for column in point_cloud.columns:
            np.testing.assert_array_equal(structured_array[column], point_cloud[column].to_numpy())

        assert epsg_code == point_cloud.crs
        assert point_cloud_identifier == point_cloud.identifier
        assert 0.1 == point_cloud.x_max_resolution
        assert 0.1 == point_cloud.y_max_resolution
        assert 0.01 == point_cloud.z_max_resolution

    def test_from_structured_array_empty_without_named_fields(self):
        point_cloud = PointCloud.from_structured_array(
            np.array([]),
            crs="EPSG:4326",
            identifier="test_point_cloud",
            x_max_resolution=0.1,
            y_max_resolution=0.01,
            z_max_resolution=1.0,
        )

        assert isinstance(point_cloud, PointCloud)
        assert len(point_cloud) == 0
        assert "EPSG:4326" == point_cloud.crs
        assert "test_point_cloud" == point_cloud.identifier
        assert 0.1 == point_cloud.x_max_resolution
        assert 0.01 == point_cloud.y_max_resolution
        assert 1.0 == point_cloud.z_max_resolution

    def test_from_structured_array_without_named_fields(self):
        xyz = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            PointCloud.from_structured_array(
                xyz,
                crs="EPSG:4326",
                identifier="test_point_cloud",
                x_max_resolution=0.1,
                y_max_resolution=0.01,
                z_max_resolution=1.0,
            )

    def test_slicing(self, point_cloud, identifier):
        point_cloud_slice = point_cloud[["x", "y"]]
        assert isinstance(point_cloud_slice, PointCloud)
        assert identifier == point_cloud_slice.identifier

    def test_series_slicing(self, point_cloud, identifier):
        point_cloud_slice = point_cloud["x"]
        assert isinstance(point_cloud_slice, PointCloudSeries)
        assert identifier == point_cloud_slice.identifier

    def test_crs(self, point_cloud):
        crs = "EPSG:4326"
        point_cloud = PointCloud(point_cloud, crs=crs)
        assert crs == point_cloud.crs

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
