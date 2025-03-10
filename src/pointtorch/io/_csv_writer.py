"""Point cloud file writer for csv and txt files."""

__all__ = ["CsvWriter"]

import math
import pathlib
from typing import Optional, Union

import pandas as pd

from ._base_point_cloud_writer import BasePointCloudWriter
from ._point_cloud_io_data import PointCloudIoData


class CsvWriter(BasePointCloudWriter):
    """Point cloud file writer for csv and txt files."""

    def supported_file_formats(self) -> list[str]:
        """
        Returns:
            File formats supported by the point cloud file writer.
        """

        return ["csv", "txt"]

    def write(
        self, point_cloud: PointCloudIoData, file_path: Union[str, pathlib.Path], columns: Optional[list[str]] = None
    ) -> None:
        """
        Writes a point cloud to a file.

        Args:
            point_cloud: Point cloud to be written.
            file_path: Path of the output file.
            columns: Point cloud columns to be written. The x, y, and z columns are always written.

        Raises:
            ValueError: If the point cloud format is not supported by the writer or if `columns` contains a column name
                that is not existing in the point cloud.
        """
        # The method from the base is called explicitly so that the read method appears in the documentation of this
        # class.
        super().write(point_cloud, file_path, columns=columns)

    def _write_data(
        self,
        point_cloud: pd.DataFrame,
        file_path: pathlib.Path,
        *,
        crs: Optional[str] = None,
        identifier: Optional[str] = None,
        x_max_resolution: Optional[float] = None,
        y_max_resolution: Optional[float] = None,
        z_max_resolution: Optional[float] = None,
    ) -> None:
        """
        Writes a point cloud to a file.

        Args:
            point_cloud: Point cloud to be written.
            file_path: Path of the output file.
            crs (str, optional): EPSG code of the coordinate reference system of the point cloud. Defaults to
                :code:`None`.
            identifier: Identifier of the point cloud. Defaults to :code:`None`.
            x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to :code:`None`.
            y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to :code:`None`.
            z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to :code:`None`.
        """

        num_decimals = None

        if x_max_resolution is not None:
            num_decimals = math.ceil(-1 * min(0, math.log(x_max_resolution, 10)))
        if y_max_resolution is not None:
            num_decimals_y = math.ceil(-1 * min(0, math.log(y_max_resolution, 10)))
            num_decimals = max(num_decimals, num_decimals_y) if num_decimals is not None else num_decimals_y
        if z_max_resolution is not None:
            num_decimals_z = math.ceil(-1 * min(0, math.log(z_max_resolution, 10)))
            num_decimals = max(num_decimals, num_decimals_z) if num_decimals is not None else num_decimals_z

        float_format = f"%.{num_decimals}f" if num_decimals is not None else None

        file_format = file_path.suffix.lstrip(".")
        point_cloud.to_csv(file_path, sep="," if file_format == "csv" else " ", index=False, float_format=float_format)
