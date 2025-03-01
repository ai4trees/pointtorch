"""Point cloud file reader for pcd files."""

__all__ = ["PcdReader"]

import pathlib
from typing import List, Optional, Tuple, Union

from pypcd.pypcd import PointCloud
import pandas as pd

from ._base_point_cloud_reader import BasePointCloudReader
from ._point_cloud_io_data import PointCloudIoData


class PcdReader(BasePointCloudReader):
    """Point cloud file reader for pcd files."""

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

        return ["pcd"]

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

        point_cloud = PointCloud.from_path(file_path).pc_data[:num_rows]

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
            Always returns :code:`None` values, since the pcd format does not provide information on the maximum
            resolution.
        """

        return None, None, None

    @staticmethod
    def _read_identifier(file_path: pathlib.Path) -> Optional[str]:
        """
        Reads the point cloud identifier from the point cloud file.

        Returns:
            Always returns :code:`None`, since the pcd format does not provide a point cloud identifier.
        """

        return None

    @staticmethod
    def _read_crs(file_path: pathlib.Path) -> Optional[str]:  # pylint: disable=unused-argument
        """
        Reads the EPSG code of the coordinate reference system from the point cloud file. Information about the
        coordinate reference system is not supported by all file formats and :code:`None` may be returned when no
        coordinate reference system is stored in a file.

        Returns:
            Always returns :code:`None`, since the pcd format does not provide information on the CRS.
        """

        return None
