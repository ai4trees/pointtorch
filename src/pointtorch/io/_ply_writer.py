"""Point cloud file writer for ply files."""

__all__ = ["PlyWriter"]

import pathlib
from typing import List, Literal, Optional, Union
import warnings

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

from ._base_point_cloud_writer import BasePointCloudWriter
from ._point_cloud_io_data import PointCloudIoData


class PlyWriter(BasePointCloudWriter):
    """
    Point cloud file writer for ply files.

    Args:
        file_type: File type to use: :code:`"ascii"` | :code:`"binary"`. Defaults to
            :code:`"binray"`.
    """

    def __init__(self, file_type: Literal["binary", "ascii"] = "binary"):
        self._file_type = file_type

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file writer.
        """

        return ["ply"]

    def write(
        self, point_cloud: PointCloudIoData, file_path: Union[str, pathlib.Path], columns: Optional[List[str]] = None
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
        file_type: Literal["ascii", "binary", "binary_compressed"] = "binary_compressed",
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
            file_type: File type to use: :code:`"ascii"`, :code:`"binary"`, :code:`"binary_compressed"`. Defaults to
                :code:`"binary_compressed"`.
        """

        data_types = dict(point_cloud.dtypes)
        for column in point_cloud.columns:
            if data_types[column] == np.int64:
                warnings.warn(
                    f"Column {column} has int64 type but ply files only support 32 bit integers, converting"
                    + " to int32"
                )
                data_types[column] = np.int32

        records = point_cloud.to_records(index=False, column_dtypes=data_types)

        point_cloud_structured_array = np.array(records, dtype=records.dtype.descr)

        element = PlyElement.describe(point_cloud_structured_array, "vertex")

        comments = []
        for name, metadata in [
            ("identifier", identifier),
            ("crs", crs),
            ("x_max_resolution", x_max_resolution),
            ("y_max_resolution", y_max_resolution),
            ("z_max_resolution", z_max_resolution),
        ]:
            if metadata is not None:
                comments.append(f"{name} {metadata}")

        PlyData([element], text=self._file_type == "ascii", comments=comments).write(str(file_path.resolve()))
