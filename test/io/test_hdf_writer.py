"""Tests for the pointtorch.io.HdfWriter class."""

import os
import pathlib
import shutil
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest

from pointtorch.io import HdfWriter, HdfReader, PointCloudIoData


class TestHdfWriter:
    """Tests for the pointtorch.io.HdfWriter class."""

    @pytest.fixture
    def hdf_reader(self):
        return HdfReader()

    @pytest.fixture
    def hdf_writer(self):
        return HdfWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestHdfWriter"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["h5", "hdf"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_writer(
        self,
        hdf_reader: HdfReader,
        hdf_writer: HdfWriter,
        cache_dir: str,
        file_format: str,
        columns: Optional[list[str]],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122], [1, 1, 1, 0, 23]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        point_cloud_data.identifier = "test"
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        hdf_writer.write(point_cloud_data, file_path, columns=columns)

        if columns is not None:
            # Test that the x, y, and z columns are always written.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

            point_cloud_df = point_cloud_df[columns]

        read_point_cloud = hdf_reader.read(file_path)

        assert (point_cloud_df.to_numpy() == read_point_cloud.data.to_numpy()).all()
        assert "test" == read_point_cloud.identifier

    def test_write_unsupported_format(self, hdf_writer: HdfWriter, cache_dir: str):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.invalid")

        with pytest.raises(ValueError):
            hdf_writer.write(point_cloud_data, file_path)

    @pytest.mark.parametrize(
        "file_format, columns",
        [
            ("h5", ["classification"]),
            ("hdf", ["classification", "intensity"]),
        ],
    )
    def test_write_missing_column(
        self, hdf_writer: HdfWriter, cache_dir: str, file_format: str, columns: Optional[list[str]]
    ):
        point_cloud_df = pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path = os.path.join(cache_dir, f"test_point_cloud.{file_format}")

        with pytest.raises(ValueError):
            hdf_writer.write(point_cloud_data, file_path, columns=columns)

    @pytest.mark.parametrize("file_format", ["h5", "hdf"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions(self, hdf_writer: HdfWriter, use_pathlib: bool, cache_dir: str, file_format: str):
        expected_x_max_resolution = 0.1
        expected_y_max_resolution = 0.01
        expected_z_max_resolution = 1

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
        hdf_writer.write(point_cloud_data, file_path)

        read_point_cloud_df = pd.read_hdf(file_path, key="point_cloud")
        max_resolution = pd.read_hdf(file_path, key="max_resolution")

        assert (point_cloud_df.to_numpy() == read_point_cloud_df.to_numpy()).all()
        assert expected_x_max_resolution == max_resolution["x_max_resolution"].iloc[0]

    @pytest.mark.parametrize("file_format", ["h5", "hdf"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions_none(
        self, hdf_writer: HdfWriter, cache_dir: str, file_format: str, use_pathlib: bool
    ):
        point_cloud_df = pd.DataFrame([[0.1, 0.0, 0.0], [1.0, 1.06, 1.0]], columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        hdf_writer.write(point_cloud_data, file_path)

        read_point_cloud_df = pd.read_hdf(file_path, key="point_cloud")

        assert (point_cloud_df.to_numpy() == read_point_cloud_df.to_numpy()).all()

    @pytest.mark.parametrize("file_format", ["h5", "hdf"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_crs(
        self, hdf_reader: HdfReader, hdf_writer: HdfWriter, cache_dir: str, file_format: str, use_pathlib: bool
    ):
        expected_crs = "EPSG:4326"

        point_cloud_df = pd.DataFrame(np.random.randn(5, 3), columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df, crs=expected_crs)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        hdf_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = hdf_reader.read(file_path)

        assert expected_crs == read_point_cloud_data.crs
