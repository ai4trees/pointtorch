""" Point cloud file writer for h5 and hdf files. """

__all__ = ["HdfWriter"]

import pathlib
from typing import List, Optional, Union

import h5py
import numpy as np
import pandas as pd

from ._base_point_cloud_writer import BasePointCloudWriter
from ._point_cloud_io_data import PointCloudIoData


class HdfWriter(BasePointCloudWriter):
    """Point cloud file writer for h5 and hdf files."""

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file writer.
        """

        return ["h5", "hdf"]

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
            identifier: Identifier of the point cloud.
            x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to `None`.
            y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to `None`.
            z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to `None`.
        """

        with h5py.File(file_path, "w") as h5file:
            h5file.attrs["identifier"] = identifier if identifier is not None else ""

        point_cloud.to_hdf(file_path, key="point_cloud", format="t", data_columns=True, index=False)

        max_resolutions = pd.DataFrame(
            [
                {
                    "x_max_resolution": x_max_resolution,
                    "y_max_resolution": y_max_resolution,
                    "z_max_resolution": z_max_resolution,
                }
            ]
        )
        max_resolutions["x_max_resolution"] = max_resolutions["x_max_resolution"].astype(np.float64)
        max_resolutions["y_max_resolution"] = max_resolutions["y_max_resolution"].astype(np.float64)
        max_resolutions["z_max_resolution"] = max_resolutions["z_max_resolution"].astype(np.float64)
        max_resolutions.to_hdf(file_path, key="max_resolution", index=False)
