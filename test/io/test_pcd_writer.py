"""Tests for the pointtorch.io.PcdWriter class."""

import os
import pathlib
import shutil
from typing import Optional, Union

import pandas as pd
import pytest

from pointtorch.io import PcdWriter, PcdReader, PointCloudIoData


class TestPcdWriter:
    """Tests for the pointtorch.io.PcdWriter class."""

    @pytest.fixture
    def pcd_reader(self):
        return PcdReader()

    @pytest.fixture
    def pcd_writer(self):
        return PcdWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestPcdWriter"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_writer(
        self,
        pcd_reader: PcdReader,
        pcd_writer: PcdWriter,
        cache_dir: str,
        columns: Optional[list[str]],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122, 56, 28, 245], [1, 1, 1, 0, 23, 128, 128, 128]],
            columns=["x", "y", "z", "classification", "instance", "r", "g", "b"],
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, "test_point_cloud.pcd")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        pcd_writer.write(point_cloud_data, file_path, columns=columns)

        if columns is not None:
            # Test that the x, y, and z columns are always written.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

            point_cloud_df = point_cloud_df[columns].copy()

        read_point_cloud_data = pcd_reader.read(file_path)

        expected_columns = sorted(point_cloud_df.columns)
        columns = sorted(read_point_cloud_data.data.columns)

        assert expected_columns == columns
        assert (
            point_cloud_df[expected_columns].to_numpy() == read_point_cloud_data.data[expected_columns].to_numpy()
        ).all()

    def test_write_unsupported_format(self, pcd_writer: PcdWriter, cache_dir: str):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.invalid")

        with pytest.raises(ValueError):
            pcd_writer.write(point_cloud_data, file_path)

    @pytest.mark.parametrize("columns", [["classification"]])
    def test_write_missing_column(self, pcd_writer: PcdWriter, cache_dir: str, columns: Optional[list[str]]):
        point_cloud_data = PointCloudIoData(pd.DataFrame([[0, 0, 0]], columns=["x", "y", "z"]))
        file_path = os.path.join(cache_dir, "test_point_cloud.pcd")

        with pytest.raises(ValueError):
            pcd_writer.write(point_cloud_data, file_path, columns=columns)
