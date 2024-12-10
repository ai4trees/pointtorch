""" Tests for the pointtorch.core.read method. """

import os
import pathlib
import shutil
from typing import Union

import pandas
import pytest

from pointtorch.core import read
from pointtorch.io import PointCloudIoData, PointCloudWriter


class TestRead:
    """Tests for the pointtorch.core.read method."""

    @pytest.fixture
    def point_cloud_writer(self):
        return PointCloudWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestRead"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["csv", "txt", "h5", "hdf", "las", "laz"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(self, point_cloud_writer: PointCloudWriter, cache_dir: str, file_format: str, use_pathlib: bool):
        point_cloud_df = pandas.DataFrame(
            [[0, 0, 0, 1, 122], [1, 1, 1, 0, 23]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud_data = PointCloudIoData(point_cloud_df)
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        point_cloud_writer.write(point_cloud_data, file_path)

        read_point_cloud = read(file_path)

        assert (point_cloud_df.to_numpy() == read_point_cloud.to_numpy()).all()
