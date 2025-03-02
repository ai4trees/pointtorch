"""Tests for the pointtorch.io.PointCloudReader class."""

import os
import pathlib
import shutil
from typing import Union

import numpy as np
import pandas as pd
import pytest

from pointtorch.io import PointCloudIoData, PointCloudReader, PointCloudWriter


class TestPointCloudReader:
    """Tests for the pointtorch.io.PointCloudReader class."""

    @pytest.fixture
    def point_cloud_reader(self):
        return PointCloudReader()

    @pytest.fixture
    def point_cloud_writer(self):
        return PointCloudWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestPointCloudReader"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["csv", "txt", "h5", "hdf", "las", "laz", "pcd", "ply"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(
        self,
        point_cloud_reader: PointCloudReader,
        point_cloud_writer: PointCloudWriter,
        cache_dir: str,
        file_format: str,
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122], [1, 1, 1, 0, 23]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = point_cloud_reader.read(file_path)

        assert (point_cloud_df.to_numpy() == read_point_cloud_data.data.to_numpy()).all()

    def test_read_unsupported_format(self, point_cloud_reader: PointCloudReader, cache_dir: str):
        file_path = pathlib.Path(cache_dir) / "test_file.invalid"
        file_path.touch()

        with pytest.raises(ValueError):
            point_cloud_reader.read(str(file_path))

    @pytest.mark.parametrize("file_format", ["csv", "txt", "h5", "hdf", "las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_max_resolutions(
        self,
        point_cloud_reader: PointCloudReader,
        point_cloud_writer: PointCloudWriter,
        cache_dir: str,
        file_format: str,
        use_pathlib: bool,
    ):
        expected_x_max_resolution = 0.1 if file_format in ["las", "laz"] else 0.01
        expected_y_max_resolution = 0.01
        expected_z_max_resolution = 1 if file_format in ["las", "laz"] else 0.01

        point_cloud_df = pd.DataFrame([[0.1, 0.0, 0.0], [1.0, 1.06, 1.0]], columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(
            point_cloud_df,
            x_max_resolution=expected_x_max_resolution,
            y_max_resolution=expected_y_max_resolution,
            z_max_resolution=expected_z_max_resolution,
        )
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = point_cloud_reader.read(file_path)

        assert expected_x_max_resolution == read_point_cloud_data.x_max_resolution
        assert expected_y_max_resolution == read_point_cloud_data.y_max_resolution
        assert expected_z_max_resolution == read_point_cloud_data.z_max_resolution

    @pytest.mark.parametrize("file_format", ["h5", "hdf", "las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_crs(
        self,
        point_cloud_reader: PointCloudReader,
        point_cloud_writer: PointCloudWriter,
        cache_dir: str,
        file_format: str,
        use_pathlib: bool,
    ):
        expected_crs = "EPSG:4326"

        point_cloud_df = pd.DataFrame(np.random.randn(5, 3), columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(
            point_cloud_df,
            crs=expected_crs,
        )
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = point_cloud_reader.read(file_path)

        assert expected_crs == read_point_cloud_data.crs
