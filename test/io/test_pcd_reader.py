"""Tests for the pointtorch.io.PcdReader class."""

import os
import pathlib
import shutil
from typing import List, Optional, Union

import pandas as pd
import pytest

from pointtorch.io import PcdReader, PcdWriter, PointCloudIoData


class TestPcdReader:
    """Tests for the pointtorch.io.PcdReader class."""

    @pytest.fixture
    def pcd_reader(self):
        return PcdReader()

    @pytest.fixture
    def pcd_writer(self):
        return PcdWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestPcdReader"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("num_rows", [None, 2])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(
        self,
        pcd_reader: PcdReader,
        pcd_writer: PcdWriter,
        cache_dir: str,
        columns: Optional[List[str]],
        num_rows: Optional[int],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 12], [1, 1, 1, 0, 23], [2, 2, 2, 0, 1]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, "test_point_cloud.pcd")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        pcd_writer.write(point_cloud_data, file_path)

        if columns is not None:
            # Test that the x, y, and z columns are always read.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)
            point_cloud_df = point_cloud_df[columns]
        if num_rows is not None:
            point_cloud_df = point_cloud_df.head(num_rows)

        read_point_cloud_data = pcd_reader.read(file_path, columns=columns, num_rows=num_rows)

        assert (point_cloud_df.to_numpy() == read_point_cloud_data.data.to_numpy()).all()

    def test_read_unsupported_format(self, pcd_reader: PcdReader, cache_dir: str):
        file_path = pathlib.Path(cache_dir) / "test_file.invalid"
        file_path.touch()

        with pytest.raises(ValueError):
            pcd_reader.read(str(file_path))
