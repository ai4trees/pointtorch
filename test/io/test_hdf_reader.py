""" Tests for the pointtorch.io.HdfReader class. """

import os
import pathlib
import shutil
from typing import List, Optional, Union

import pandas as pd
import pytest

from pointtorch.io import HdfReader, HdfWriter, PointCloudIoData


class TestHdfReader:
    """Tests for the pointtorch.io.HdfReader class."""

    @pytest.fixture
    def hdf_reader(self):
        return HdfReader()

    @pytest.fixture
    def hdf_writer(self):
        return HdfWriter()

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestHdfReader"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("file_format", ["h5", "hdf"])
    @pytest.mark.parametrize("columns", [None, ["classification"], ["x", "y", "z", "classification"]])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read(
        self,
        hdf_reader: HdfReader,
        hdf_writer: HdfWriter,
        cache_dir: str,
        file_format: str,
        columns: Optional[List[str]],
        use_pathlib: bool,
    ):
        point_cloud_df = pd.DataFrame(
            [[0, 0, 0, 1, 122], [1, 1, 1, 0, 23]], columns=["x", "y", "z", "classification", "instance"]
        )
        point_cloud = PointCloudIoData(point_cloud_df)
        point_cloud.identifier = "test"
        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, f"test_point_cloud.{file_format}")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        hdf_writer.write(point_cloud, file_path)

        read_point_cloud = hdf_reader.read(file_path, columns=columns)

        if columns is not None:
            # Test that the x, y, and z columns are always read.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)
            point_cloud_df = point_cloud_df[columns]

        assert (point_cloud_df.to_numpy() == read_point_cloud.data.to_numpy()).all()
        assert "test" == read_point_cloud.identifier

    def test_read_unsupported_format(self, hdf_reader: HdfReader, cache_dir: str):
        file_path = pathlib.Path(cache_dir) / "test_file.invalid"
        file_path.touch()

        with pytest.raises(ValueError):
            hdf_reader.read(str(file_path))

    @pytest.mark.parametrize("file_format", ["h5", "hdf"])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_read_max_resolutions(
        self, hdf_reader: HdfReader, hdf_writer: HdfWriter, cache_dir: str, file_format: str, use_pathlib: bool
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

        hdf_writer.write(point_cloud, file_path)

        read_point_cloud = hdf_reader.read(file_path)

        assert expected_x_max_resolution == read_point_cloud.x_max_resolution
        assert expected_y_max_resolution == read_point_cloud.y_max_resolution
        assert expected_z_max_resolution == read_point_cloud.z_max_resolution
