"""Tests for the pointtorch.io.CsvReader class."""

import os
import pathlib
from pathlib import Path
import shutil
from typing import Optional, Union

import pandas as pd
import pytest

from pointtorch.io import CsvReader, CsvWriter, PointCloudIoData


class TestCsvReader:
    """Tests for the pointtorch.io.CsvReader class."""

    @pytest.fixture
    def csv_reader(self):
        return CsvReader()

    @pytest.fixture
    def csv_writer(self):
        return CsvWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestCsvReader"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["csv", "txt"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("num_rows", [None, 2])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(
        self,
        csv_reader: CsvReader,
        cache_dir: str,
        file_format: str,
        columns: Optional[list[str]],
        num_rows: Optional[int],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 12], [1, 1, 1, 0, 23], [2, 2, 2, 0, 1]], columns=["x", "y", "z", "classification", "instance"]
        )
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)
        point_cloud_df.to_csv(file_path, index=False, sep="," if file_format == "csv" else " ")

        read_point_cloud = csv_reader.read(file_path, columns=columns, num_rows=num_rows)

        if columns is not None:
            # Test that the x, y, and z columns are always read.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)
            point_cloud_df = point_cloud_df[columns]
        if num_rows is not None:
            point_cloud_df = point_cloud_df.head(num_rows)

        assert (point_cloud_df.to_numpy() == read_point_cloud.data.to_numpy()).all()

    def test_read_unsupported_format(self, csv_reader: CsvReader, cache_dir: str):
        file_path = Path(cache_dir) / "test_file.invalid"
        file_path.touch()

        with pytest.raises(ValueError):
            csv_reader.read(str(file_path))

    @pytest.mark.parametrize("file_format", ["csv", "txt"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_max_resolutions(
        self, csv_reader: CsvReader, csv_writer: CsvWriter, cache_dir: str, file_format: str, use_pathlib: bool
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
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        csv_writer.write(point_cloud, file_path)

        read_point_cloud = csv_reader.read(file_path)

        assert expected_x_max_resolution == read_point_cloud.x_max_resolution
        assert expected_y_max_resolution == read_point_cloud.y_max_resolution
        assert expected_z_max_resolution == read_point_cloud.z_max_resolution
