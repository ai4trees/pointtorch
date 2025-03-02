"""Tests for the pointtorch.io.PlyReader class."""

import os
import pathlib
import shutil
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest

from pointtorch.io import PlyReader, PlyWriter, PointCloudIoData


class TestPlyReader:
    """Tests for the pointtorch.io.PlyReader class."""

    @pytest.fixture
    def ply_reader(self):
        return PlyReader()

    @pytest.fixture
    def ply_writer(self):
        return PlyWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestPlyReader"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("num_rows", [None, 2])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(
        self,
        ply_reader: PlyReader,
        ply_writer: PlyWriter,
        cache_dir: str,
        columns: Optional[list[str]],
        num_rows: Optional[int],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 12], [1, 1, 1, 0, 23], [2, 2, 2, 0, 1]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        point_cloud_data.identifier = "test"
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, "test_point_cloud.ply")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud_data, file_path)

        if columns is not None:
            # Test that the x, y, and z columns are always read.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)
            point_cloud_df = point_cloud_df[columns]
        if num_rows is not None:
            point_cloud_df = point_cloud_df.head(num_rows)

        read_point_cloud_data = ply_reader.read(file_path, columns=columns, num_rows=num_rows)

        assert (point_cloud_df.to_numpy() == read_point_cloud_data.data.to_numpy()).all()
        assert "test" == read_point_cloud_data.identifier

    def test_read_unsupported_format(self, ply_reader: PlyReader, cache_dir: str):
        file_path = pathlib.Path(cache_dir) / "test_file.invalid"
        file_path.touch()

        with pytest.raises(ValueError):
            ply_reader.read(str(file_path))

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_max_resolutions(
        self, ply_reader: PlyReader, ply_writer: PlyWriter, cache_dir: str, use_pathlib: bool
    ):
        expected_x_max_resolution = 0.01
        expected_y_max_resolution = 0.01
        expected_z_max_resolution = 0.01

        point_cloud = PointCloudIoData(
            pd.DataFrame([[0.1, 0.0, 0.0], [1.0, 1.06, 1.0]], columns=["x", "y", "z"]),
            x_max_resolution=expected_x_max_resolution,
            y_max_resolution=expected_y_max_resolution,
            z_max_resolution=expected_z_max_resolution,
        )
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.ply")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud, file_path)

        read_point_cloud = ply_reader.read(file_path)

        assert expected_x_max_resolution == read_point_cloud.x_max_resolution
        assert expected_y_max_resolution == read_point_cloud.y_max_resolution
        assert expected_z_max_resolution == read_point_cloud.z_max_resolution

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_max_resolutions_invalid(
        self, ply_reader: PlyReader, ply_writer: PlyWriter, cache_dir: str, use_pathlib: bool
    ):
        point_cloud = PointCloudIoData(
            pd.DataFrame([[0.1, 0.0, 0.0], [1.0, 1.06, 1.0]], columns=["x", "y", "z"]),
            x_max_resolution="test",
            y_max_resolution="test",
            z_max_resolution="test",
        )
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.ply")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud, file_path)

        read_point_cloud = ply_reader.read(file_path)

        assert read_point_cloud.x_max_resolution is None
        assert read_point_cloud.y_max_resolution is None
        assert read_point_cloud.z_max_resolution is None

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_crs(self, ply_reader: PlyReader, ply_writer: PlyWriter, cache_dir: str, use_pathlib: bool):
        expected_crs = "EPSG:4326"

        point_cloud = PointCloudIoData(
            pd.DataFrame(np.random.randn(5, 3), columns=["x", "y", "z"]),
            crs=expected_crs,
        )
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.ply")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        ply_writer.write(point_cloud, file_path)

        read_point_cloud = ply_reader.read(file_path)

        assert expected_crs == read_point_cloud.crs
