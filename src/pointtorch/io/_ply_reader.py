"""Point cloud file reader for ply files."""

__all__ = ["PlyReader"]

import pathlib
from typing import List, Optional, Tuple, Union

import pandas as pd
from plyfile import PlyData

from ._base_point_cloud_reader import BasePointCloudReader
from ._point_cloud_io_data import PointCloudIoData


class PlyReader(BasePointCloudReader):
    """Point cloud file reader for ply files."""

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

        return ["ply"]

    def read(
        self, file_path: Union[str, pathlib.Path], columns: Optional[List[str]] = None, num_rows: Optional[int] = None
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
        self, file_path: pathlib.Path, columns: Optional[List[str]] = None, num_rows: Optional[int] = None
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

        plydata = PlyData.read(file_path)

        print("plydata", plydata)
        print("plydata['vertex']", plydata["vertex"].data)

        point_cloud = plydata["vertex"].data[:num_rows]

        if columns is not None:
            point_cloud = point_cloud[columns]

        return pd.DataFrame(point_cloud)

    @staticmethod
    def _read_max_resolutions(file_path: pathlib.Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Reads the maximum resolution for each coordinate dimension from the point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.

        Returns:
            Always returns :code:`None` values, since the ply format does not provide information on the maximum
            resolution.
        """

        plydata = PlyData.read(file_path)

        x_max_resolution = None
        y_max_resolution = None
        z_max_resolution = None

        for comment in plydata.comments:
            if "x_max_resolution" in comment:
                try:
                    x_max_resolution = float(comment.removeprefix("x_max_resolution "))
                except ValueError:
                    pass
            if "y_max_resolution" in comment:
                try:
                    y_max_resolution = float(comment.removeprefix("y_max_resolution "))
                except ValueError:
                    pass
            if "z_max_resolution" in comment:
                try:
                    z_max_resolution = float(comment.removeprefix("z_max_resolution "))
                except ValueError:
                    pass
            if x_max_resolution is not None and y_max_resolution is not None and z_max_resolution is not None:
                break

        return x_max_resolution, y_max_resolution, z_max_resolution

    @staticmethod
    def _read_identifier(file_path: pathlib.Path) -> Optional[str]:
        """
        Reads the point cloud identifier from the point cloud file.

        Returns:
            Always returns :code:`None`, since the ply format does not provide a point cloud identifier.
        """

        plydata = PlyData.read(file_path)

        identifier = None
        for comment in plydata.comments:
            if "identifier" in comment:
                identifier = comment.removeprefix("identifier ")
                break

        return identifier

    @staticmethod
    def _read_crs(file_path: pathlib.Path) -> Optional[str]:  # pylint: disable=unused-argument
        """
        Reads the EPSG code of the coordinate reference system from the point cloud file. Information about the
        coordinate reference system is not supported by all file formats and :code:`None` may be returned when no
        coordinate reference system is stored in a file.

        Returns:
            Always returns :code:`None`, since the ply format does not provide information on the CRS.
        """

        plydata = PlyData.read(file_path)

        crs = None
        for comment in plydata.comments:
            if "crs" in comment:
                crs = comment.removeprefix("crs ")
                break

        return crs
