"""Method for reading point cloud files."""

__all__ = ["read"]

import pathlib
from typing import List, Optional, Union

from pointtorch.core._point_cloud import PointCloud
from pointtorch.io import PointCloudReader


def read(
    file_path: Union[str, pathlib.Path], columns: Optional[List[str]] = None, num_rows: Optional[int] = None
) -> PointCloud:
    """
    Method for reading point cloud files.

    Args:
        file_path: Path of the point cloud file to be read.
        columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.
        num_rows: Number of rows to read. If set to :code:`None`, all rows are read. Defaults to :code:`None`.

    Returns:
        Point cloud object.

    Raises:
        ValueError: If the point cloud format is not supported by the reader.
    """

    reader = PointCloudReader()
    point_cloud_data = reader.read(file_path, columns=columns, num_rows=num_rows)
    point_cloud = PointCloud(
        point_cloud_data.data,
        crs=point_cloud_data.crs,
        identifier=point_cloud_data.identifier,
        x_max_resolution=point_cloud_data.x_max_resolution,
        y_max_resolution=point_cloud_data.y_max_resolution,
        z_max_resolution=point_cloud_data.z_max_resolution,
    )

    return point_cloud
