"""Tests for the pointtorch.io.PlyWriter class."""

import os
import pathlib
import shutil
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest

from pointtorch.io import PlyWriter, PlyReader, PointCloudIoData


class TestPlyWriter:
    """Tests for the pointtorch.io.PlyWriter class."""

    @pytest.fixture
    def ply_reader(self):
        return PlyReader()

    @pytest.fixture
    def ply_writer(self):
        return PlyWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestPlyWriter"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["ply", "PLY"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    @pytest.mark.parametrize("file_type", ["binary", "ascii"])
    def test_writer(
        self, ply_reader: PlyReader, cache_dir: str, file_format: str, columns: Optional[list[str]], use_pathlib: bool, file_type: str
    ):
        ply_writer = PlyWriter(file_type=file_type)
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122, 56, 28, 245], [1, 1, 1, 0, 23, 128, 128, 128]],
            columns=["x", "y", "z", "classification", "instance", "r", "g", "b"],
            dtype=np.int32,
        )
        point_cloud_df[["x", "y", "z"]] = point_cloud_df[["x", "y", "z"]].astype(np.float64)
        point_cloud_df[["instance"]] = point_cloud_df[["instance"]].astype(np.int64)
        point_cloud_df[["r", "g", "b"]] = point_cloud_df[["r", "g", "b"]].astype(np.uint32)
        point_cloud_data = PointCloudIoData(point_cloud_df)
        point_cloud_data.identifier = "test"
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud_data, file_path, columns=columns)

        if columns is not None:
            # Test that the x, y, and z columns are always written.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

            point_cloud_df = point_cloud_df[columns].copy()

        read_point_cloud_data = ply_reader.read(file_path)

        expected_columns = sorted(point_cloud_df.columns)
        columns = sorted(read_point_cloud_data.data.columns)

        assert expected_columns == columns
        assert (
            point_cloud_df[expected_columns].to_numpy() == read_point_cloud_data.data[expected_columns].to_numpy()
        ).all()
        assert "test" == read_point_cloud_data.identifier

        for column in point_cloud_df.columns:
            if point_cloud_df[column].dtype != np.int64:
                assert point_cloud_df[column].dtype == read_point_cloud_data.data[column].dtype
            else:
                assert np.int32 == read_point_cloud_data.data[column].dtype

    def test_write_unsupported_format(self, ply_writer: PlyWriter, cache_dir: str):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.invalid")

        with pytest.raises(ValueError):
            ply_writer.write(point_cloud_data, file_path)

    @pytest.mark.parametrize("columns", [["classification"]])
    def test_write_missing_column(self, ply_writer: PlyWriter, cache_dir: str, columns: Optional[list[str]]):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.ply")

        with pytest.raises(ValueError):
            ply_writer.write(point_cloud_data, file_path, columns=columns)

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions(
        self,
        ply_reader: PlyReader,
        ply_writer: PlyWriter,
        cache_dir: str,
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

        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, "test_file.ply")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = ply_reader.read(file_path)

        assert read_point_cloud_data.x_max_resolution == expected_x_max_resolution
        assert read_point_cloud_data.y_max_resolution == expected_y_max_resolution
        assert read_point_cloud_data.z_max_resolution == expected_z_max_resolution

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_crs(self, ply_reader: PlyReader, ply_writer: PlyWriter, cache_dir: str, use_pathlib: bool):
        expected_crs = "EPSG:4326"

        point_cloud_df = pd.DataFrame(np.random.randn(5, 3), columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df, crs=expected_crs)

        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, "test_point_cloud.ply")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = ply_reader.read(file_path)

        assert expected_crs == read_point_cloud_data.crs
