"""Abstract base class for implementing point cloud file writers."""

__all__ = ["BasePointCloudWriter"]

import abc
import pathlib
from typing import Optional, Union

import pandas as pd

from ._point_cloud_io_data import PointCloudIoData


class BasePointCloudWriter(abc.ABC):
    """Abstract base class for implementing point cloud file writers."""

    @abc.abstractmethod
    def supported_file_formats(self) -> list[str]:
        """
        Returns:
            File formats supported by the point cloud file writer.
        """

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

        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        file_format = file_path.suffix.lstrip(".")
        if file_format not in self.supported_file_formats():
            raise ValueError(f"The {file_format} format is not supported by the point cloud writer.")

        point_cloud_df = point_cloud.data
        # pylint: disable=duplicate-code
        if columns is not None:
            columns = columns.copy()

            # The x, y, z coordinates are always written.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

            missing_columns = set(columns).difference(set(point_cloud_df.columns))
            if len(missing_columns) == 1:
                raise ValueError(f"The point cloud does not contain a column named {list(missing_columns)[0]}.")
            if len(missing_columns) > 1:
                missing_columns_list = list(missing_columns)
                missing_columns_str = f"{', '.join(missing_columns_list[:-1])}, and {missing_columns_list[-1]}"
                raise ValueError(f"The point cloud does not contain columns named {missing_columns_str}.")
            point_cloud_df = point_cloud_df[columns]

        self._write_data(
            point_cloud_df,
            file_path,
            crs=point_cloud.crs,
            identifier=point_cloud.identifier,
            x_max_resolution=point_cloud.x_max_resolution,
            y_max_resolution=point_cloud.y_max_resolution,
            z_max_resolution=point_cloud.z_max_resolution,
        )

    @abc.abstractmethod
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
        Writes a point cloud to a file. This method has to be overriden by child classes.

        Args:
            point_cloud: Point cloud to be written.
            file_path: Path of the output file.
            crs: EPSG code of the coordinate reference system of the point cloud. Defaults to :code:`None`.
            identifier: Identifier of the point cloud. Defaults to :code:`None`.
            x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to :code:`None`.
            y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to :code:`None`.
            z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to :code:`None`.
        """
