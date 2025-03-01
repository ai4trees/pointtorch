"""Point cloud file writer for pcd files."""

__all__ = ["PcdWriter"]

import pathlib
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pypcd.pypcd import PointCloud, numpy_pcd_type_mappings

from ._base_point_cloud_writer import BasePointCloudWriter
from ._point_cloud_io_data import PointCloudIoData


class PcdWriter(BasePointCloudWriter):
    """Point cloud file writer for pcd files."""

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file writer.
        """

        return ["pcd"]

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
        Writes a point cloud to a file. The point coordinates are always stored as 32 bit floating point numbers, as
        some PCD readers require this.

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

        dtypes = dict(point_cloud.dtypes)
        dtypes["x"] = np.float32
        dtypes["y"] = np.float32
        dtypes["z"] = np.float32

        records = point_cloud.to_records(index=False, column_dtypes=dtypes)
        point_cloud_structured_array = np.array(records, dtype=records.dtype.descr)

        type_mapping = dict(numpy_pcd_type_mappings)

        metadata = {
            "version": "0.7",
            "fields": point_cloud.columns,
            "type": [type_mapping[point_cloud_structured_array[column].dtype][0] for column in point_cloud.columns],
            "size": [type_mapping[point_cloud_structured_array[column].dtype][1] for column in point_cloud.columns],
            "width": len(point_cloud),
            "height": 1,
            "points": len(point_cloud),
            "viewpoint": "0 0 0 1 0 0 0",
            "data": file_type,
            "count": [1 for _ in range(len(point_cloud.columns))],
        }

        point_cloud_pypcd = PointCloud(metadata, point_cloud_structured_array)
        point_cloud_pypcd.save_pcd(file_path, compression=file_type)
