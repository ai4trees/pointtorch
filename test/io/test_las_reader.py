""" Tests for the pointtorch.io.LasReader class. """

import os
import pathlib
import shutil
from typing import List, Optional, Union

import pandas as pd
import pytest

from pointtorch.io import LasReader, LasWriter, PointCloudIoData


class TestLasReader:
    """Tests for the pointtorch.io.LasReader class."""

    @pytest.fixture
    def las_reader(self):
        return LasReader()

    @pytest.fixture
    def las_writer(self):
        return LasWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestLasReader"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["las", "laz"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("num_rows", [None, 2])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(
        self,
        las_reader: LasReader,
        las_writer: LasWriter,
        cache_dir: str,
        file_format: str,
        columns: Optional[List[str]],
        num_rows: Optional[int],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122], [1, 1, 1, 0, 23]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        las_writer.write(point_cloud_data, file_path)

        if columns is not None:
            # Test that the x, y, and z columns are always read.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)
            point_cloud_df = point_cloud_df[columns].head(num_rows)

        read_point_cloud_data = las_reader.read(file_path, columns=columns, num_rows=num_rows)

        assert (point_cloud_df.to_numpy() == read_point_cloud_data.data.to_numpy()).all()

    def test_read_unsupported_format(self, las_reader: LasReader, cache_dir: str):
        file_path = pathlib.Path(cache_dir) / "test_file.invalid"
        file_path.touch()

        with pytest.raises(ValueError):
            las_reader.read(str(file_path))

    @pytest.mark.parametrize("file_format", ["las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_max_resolutions(
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

        assert expected_x_max_resolution == read_point_cloud_data.x_max_resolution
        assert expected_y_max_resolution == read_point_cloud_data.y_max_resolution
        assert expected_z_max_resolution == read_point_cloud_data.z_max_resolution
