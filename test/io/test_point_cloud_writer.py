"""Tests for the pointtorch.io.PointCloudWriter class."""

import os
import pathlib
import shutil
from typing import Union

import numpy as np
import pandas as pd
import pytest

from pointtorch.io import PointCloudIoData, PointCloudWriter, PointCloudReader


class TestPointCloudWriter:
    """Tests for the pointtorch.io.PointCloudWriter class."""

    @pytest.fixture
    def point_cloud_reader(self):
        return PointCloudReader()

    @pytest.fixture
    def point_cloud_writer(self):
        return PointCloudWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestPointCloudWriter"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize(
        "file_format",
        ["csv", "CSV", "txt", "TXT", "h5", "H5", "hdf", "HDF", "las", "LAS", "laz", "LAZ", "pcd", "PCD", "ply", "PLY"],
    )
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_writer(
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

    def test_write_unsupported_format(self, point_cloud_writer: PointCloudWriter, cache_dir: str):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_file.invalid")

        with pytest.raises(ValueError):
            point_cloud_writer.write(point_cloud_data, file_path)

    @pytest.mark.parametrize("file_format", ["csv", "txt", "h5", "hdf", "las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions(
        self,
        point_cloud_reader: PointCloudReader,
        point_cloud_writer: PointCloudWriter,
        cache_dir: str,
        file_format: str,
        use_pathlib: bool,
    ):
        expected_x_max_resolution = 0.1
        expected_y_max_resolution = 0.01
        expected_z_max_resolution = 1

        point_cloud_df = pd.DataFrame([[0.1, 0, 0], [1, 1.06, 1]], columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(
            point_cloud_df,
            x_max_resolution=expected_x_max_resolution,
            y_max_resolution=expected_y_max_resolution,
            z_max_resolution=expected_z_max_resolution,
        )

        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_file.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = point_cloud_reader.read(file_path)

        assert (point_cloud_df.to_numpy() == read_point_cloud_data.data.to_numpy()).all()

    @pytest.mark.parametrize("file_format", ["h5", "hdf", "las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_crs(
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

        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_file.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = point_cloud_reader.read(file_path)

        assert expected_crs == read_point_cloud_data.crs
