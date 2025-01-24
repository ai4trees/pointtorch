""" Tests for the pointtorch.io.LasWriter class. """

import os
import pathlib
import shutil
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pytest

from pointtorch.io import LasWriter, LasReader, PointCloudIoData


class TestLasWriter:
    """Tests for the pointtorch.io.LasWriter class."""

    @pytest.fixture
    def las_reader(self):
        return LasReader()

    @pytest.fixture
    def las_writer(self):
        return LasWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestLasWriter"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["las", "laz"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_writer(
        self,
        las_reader: LasReader,
        las_writer: LasWriter,
        cache_dir: str,
        file_format: str,
        columns: Optional[List[str]],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122, 56, 28, 245], [1, 1, 1, 0, 23, 128, 128, 128]],
            columns=["x", "y", "z", "classification", "instance", "r", "g", "b"],
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        las_writer.write(point_cloud_data, file_path, columns=columns)

        if columns is not None:
            # Test that the x, y, and z columns are always written.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

            point_cloud_df = point_cloud_df[columns].copy()

        point_cloud_df.rename({"r": "red", "g": "green", "b": "blue"}, axis=1, inplace=True)

        read_point_cloud_data = las_reader.read(file_path)

        expected_columns = sorted(point_cloud_df.columns)
        columns = sorted(read_point_cloud_data.data.columns)

        assert expected_columns == columns
        assert (
            point_cloud_df[expected_columns].to_numpy() == read_point_cloud_data.data[expected_columns].to_numpy()
        ).all()

    def test_write_unsupported_format(self, las_writer: LasWriter, cache_dir: str):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.invalid")

        with pytest.raises(ValueError):
            las_writer.write(point_cloud_data, file_path)

    @pytest.mark.parametrize(
        "file_format,columns",
        [
            (
                "las",
                ["classification"],
            ),
            (
                "laz",
                ["classification", "intensity"],
            ),
        ],
    )
    def test_write_missing_column(
        self, las_writer: LasWriter, cache_dir: str, file_format: str, columns: Optional[List[str]]
    ):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, f"test_point_cloud.{file_format}")

        with pytest.raises(ValueError):
            las_writer.write(point_cloud_data, file_path, columns=columns)

    @pytest.mark.parametrize("file_format", ["las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions(
        self, las_reader: LasReader, las_writer: LasWriter, cache_dir: str, file_format: str, use_pathlib: bool
    ):
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

        las_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = las_reader.read(file_path)

        assert (point_cloud_df.to_numpy() == read_point_cloud_data.data.to_numpy()).all()

    @pytest.mark.parametrize("file_format", ["las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_crs(
        self, las_reader: LasReader, las_writer: LasWriter, cache_dir: str, file_format: str, use_pathlib: bool
    ):
        expected_crs = "EPSG:4326"

        point_cloud_df = pd.DataFrame(np.random.randn(5, 3), columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df, crs=expected_crs)

        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        las_writer.write(point_cloud_data, file_path)

        read_point_cloud_data = las_reader.read(file_path)

        assert expected_crs == read_point_cloud_data.crs
