"""Point cloud file reader for h5 and hdf files."""

__all__ = ["HdfReader"]

import pathlib
from typing import Optional, Tuple, Union

import h5py
import pandas as pd

from ._base_point_cloud_reader import BasePointCloudReader
from ._point_cloud_io_data import PointCloudIoData


class HdfReader(BasePointCloudReader):
    """Point cloud file reader for h5 and hdf files."""

    def supported_file_formats(self) -> list[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

        return ["h5", "hdf"]

    def read(
        self, file_path: Union[str, pathlib.Path], columns: Optional[list[str]] = None, num_rows: Optional[int] = None
    ) -> PointCloudIoData:
        """
        Reads a point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.
            num_rows: Number of rows to read. If set to :code:`None`, all rows are read. Defaults to :code:`None`.

        Returns:
            Point cloud object.

        Raises:
            ValueError: If the point cloud format is not supported by the reader.
        """
        # The method from the base is called explicitly so that the read method appears in the documentation of this
        # class.
        return super().read(file_path, columns=columns, num_rows=num_rows)

    def _read_points(
        self, file_path: pathlib.Path, columns: Optional[list[str]] = None, num_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Reads point data from a point cloud file in h5 and hdf format.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.
            num_rows: Number of rows to read. If set to :code:`None`, all rows are read. Defaults to :code:`None`.

        Returns:
            Point cloud data.
        """

        start = None
        stop = None
        if num_rows is not None:
            start = 0
            stop = num_rows

        return pd.read_hdf(  # type: ignore[return-value]
            file_path, columns=columns, key="point_cloud", start=start, stop=stop
        )

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

    @staticmethod
    def _read_crs(file_path: pathlib.Path) -> Optional[str]:  # pylint: disable=unused-argument
        """
        Reads the EPSG code of the coordinate reference system from the point cloud file. Information about the
        coordinate reference system is not supported by all file formats and :code:`None` may be returned when no
        coordinate reference system is stored in a file.

        Returns:
            EPSG code of the coordinate reference system or :code:`None` if no coordinate reference system is stored in
            the file.
        """
        with h5py.File(file_path, "r") as h5file:
            crs = h5file.attrs["crs"]
        return crs if len(crs) > 0 else None
