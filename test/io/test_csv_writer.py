""" Tests for the pointtorch.io.CsvWriter class. """

import os
import pathlib
import shutil
from typing import List, Optional, Union

import pandas as pd
import pytest

from pointtorch.io import CsvWriter, PointCloudIoData


class TestCsvWriter:
    """Tests for the pointtorch.io.CsvWriter class."""

    @pytest.fixture
    def csv_writer(self):
        return CsvWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestCsvWriter"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["csv", "txt"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_writer(
        self, csv_writer: CsvWriter, cache_dir: str, file_format: str, columns: Optional[List[str]], use_pathlib: bool
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122], [1, 1, 1, 0, 23]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        csv_writer.write(point_cloud_data, file_path, columns=columns)

        if columns is not None:
            # Test that the x, y, and z columns are always written.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

            point_cloud_df = point_cloud_df[columns]

        read_point_cloud_df = pd.read_csv(file_path, sep="," if file_format == "csv" else " ")

        assert (point_cloud_df.to_numpy() == read_point_cloud_df.to_numpy()).all()

    def test_write_unsupported_format(self, csv_writer: CsvWriter, cache_dir: str):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.invalid")

        with pytest.raises(ValueError):
            csv_writer.write(point_cloud_data, file_path)

    @pytest.mark.parametrize(
        "file_format, columns",
        [
            ("csv", ["classification"]),
            ("txt", ["classification", "intensity"]),
        ],
    )
    def test_write_missing_column(
        self, csv_writer: CsvWriter, cache_dir: str, file_format: str, columns: Optional[List[str]]
    ):
        point_cloud_df = pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path = os.path.join(cache_dir, f"test_point_cloud.{file_format}")

        with pytest.raises(ValueError):
            csv_writer.write(point_cloud_data, file_path, columns=columns)

    @pytest.mark.parametrize("file_format", ["csv", "txt"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions(self, csv_writer: CsvWriter, cache_dir: str, file_format: str, use_pathlib: bool):
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

        csv_writer.write(point_cloud_data, file_path)

        read_point_cloud_df = pd.read_csv(file_path, sep="," if file_format == "csv" else " ")

        assert (point_cloud_df.to_numpy() == read_point_cloud_df.to_numpy()).all()

    @pytest.mark.parametrize("file_format", ["csv", "txt"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_write_max_resolutions_none(
        self, csv_writer: CsvWriter, cache_dir: str, file_format: str, use_pathlib: bool
    ):
        point_cloud_df = pd.DataFrame([[0.1, 0.0, 0.0], [1.0, 1.06, 1.0]], columns=["x", "y", "z"])
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        csv_writer.write(point_cloud_data, file_path)

        read_point_cloud_df = pd.read_csv(file_path, sep="," if file_format == "csv" else " ")

        assert (point_cloud_df.to_numpy() == read_point_cloud_df.to_numpy()).all()
