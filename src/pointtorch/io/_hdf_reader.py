""" Point cloud file reader for h5 and hdf files. """

__all__ = ["HdfReader"]

import pathlib
from typing import List, Optional, Tuple, Union

import h5py
import pandas as pd

from ._base_point_cloud_reader import BasePointCloudReader
from ._point_cloud_io_data import PointCloudIoData


class HdfReader(BasePointCloudReader):
    """Point cloud file reader for h5 and hdf files."""

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

        return ["h5", "hdf"]

    def read(self, file_path: Union[str, pathlib.Path], columns: Optional[List[str]] = None) -> PointCloudIoData:
        """
        Reads a point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.

        Returns:
            Point cloud object.

        Raises:
            ValueError: If the point cloud format is not supported by the reader.
        """
        # The method from the base is called explicitly so that the read method appears in the documentation of this
        # class.
        return super().read(file_path, columns=columns)

    def _read_points(self, file_path: pathlib.Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reads point data from a point cloud file in h5 and hdf format.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.

        Returns:
            Point cloud data.
        """

        return pd.read_hdf(file_path, columns=columns, key="point_cloud")  # type: ignore[return-value]

    @staticmethod
    def _read_max_resolutions(file_path: pathlib.Path) -> Tuple[float, float, float]:
        """
        Reads the maximum resolution for each coordinate dimension from the point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.

        Returns:
            Maximum resolution of the x-, y-, and z-coordinates of the point cloud.
        """

        x_max_resolution = float(pd.read_hdf(file_path, key="max_resolution")["x_max_resolution"].iloc[0])
        y_max_resolution = float(pd.read_hdf(file_path, key="max_resolution")["y_max_resolution"].iloc[0])
        z_max_resolution = float(pd.read_hdf(file_path, key="max_resolution")["z_max_resolution"].iloc[0])

        return x_max_resolution, y_max_resolution, z_max_resolution

    @staticmethod
    def _read_identifier(file_path: pathlib.Path) -> Optional[str]:
        """
        Reads the point cloud identifier from the point cloud file.

        Returns:
            Point cloud identifier.
        """

        with h5py.File(file_path, "r") as h5file:
            identifier = h5file.attrs["identifier"]
        return identifier if len(identifier) > 0 else None
