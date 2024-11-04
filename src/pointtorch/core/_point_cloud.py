""" Point cloud object. """

__all__ = ["PointCloud"]

from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from pointtorch.io import PointCloudIoData, PointCloudWriter


class PointCloud(pd.DataFrame):
    """Point cloud object. Subclass of
    `pd.DataFrame <https://pd.pydata.org/docs/reference/api/pd.DataFrame.html>`_

    Args:
        identifier: Point cloud identifier. Defaults to `None`.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to `None`.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to `None`.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to `None`.

    For a documentation of other parameters, see the documentation of \
   `pd.DataFrame <https://pd.pydata.org/docs/reference/api/pd.DataFrame.html>`_.

    Attributes:
        identifier: Point cloud identifier.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter.
    """

    _metadata = ["identifier", "x_max_resolution", "y_max_resolution", "z_max_resolution"]

    def __init__(
        self,
        data: Union[np.ndarray, Iterable, Dict, pd.DataFrame],
        index: Optional[Union[pd.Index, np.ndarray]] = None,
        columns: Optional[Union[pd.Index, np.ndarray, List[str]]] = None,
        dtype: Optional[np.dtype] = None,
        copy: Optional[bool] = True,
        identifier: Optional[str] = None,
        x_max_resolution: Optional[float] = None,
        y_max_resolution: Optional[float] = None,
        z_max_resolution: Optional[float] = None,
    ) -> None:
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)  # type: ignore[call-arg]
        self.identifier = identifier
        self.x_max_resolution = x_max_resolution
        self.y_max_resolution = y_max_resolution
        self.z_max_resolution = z_max_resolution

    @property
    def _constructor(self):
        return PointCloud

    @property
    def _constructor_sliced(self):
        return PointCloudSeries

    def xyz(self) -> np.ndarray:
        """
        Returns:
            x, y, and z coordinates of the points in the point cloud.

        Raises:
            RuntimeError: if "x", "y", or "z" are not in `self.columns`.
        """

        if "x" not in self.columns or "y" not in self.columns or "z" not in self.columns:
            raise RuntimeError("")

        return self[["x", "y", "z"]].to_numpy()

    def to(self, file_path: Union[str, Path], columns: Optional[List[str]] = None) -> None:
        """
        Writes the point cloud to a file.

        Args:
            file_path: Path of the output file.
            columns: Point cloud columns to be written. The x, y, and z columns are always written.

        Raises:
            ValueError: If the point cloud format is not supported by the writer or if `columns` contains a column name
                that is not existing in the point cloud.
        """

        writer = PointCloudWriter()
        point_cloud_data = PointCloudIoData(
            self,
            identifier=self.identifier,
            x_max_resolution=self.x_max_resolution,
            y_max_resolution=self.y_max_resolution,
            z_max_resolution=self.z_max_resolution,
        )
        writer.write(point_cloud_data, file_path, columns=columns)


class PointCloudSeries(pd.Series):
    """
    A data series that represents a point cloud column. Subclass of
    `pd.Series <https://pd.pydata.org/pandas-docs/stable/reference/api/pd.Series.html>`_.

    Args:
        identifier: Point cloud identifier. Defaults to `None`.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to `None`.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to `None`.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to `None`.

    For a documentation of other parameters, see the documentation of \
        `pd.Series <https://pd.pydata.org/pandas-docs/stable/reference/api/pd.Series.html>`_.


    Attributes:
        identifier: Point cloud identifier.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter.
    """

    _metadata = ["identifier", "x_max_resolution", "y_max_resolution", "z_max_resolution"]

    def __init__(
        self,
        data: Optional[Union[np.ndarray, Iterable, Dict, int, float, str]] = None,
        index: Optional[Union[pd.Index, np.ndarray]] = None,
        dtype: Optional[Union[str, np.dtype, pd.api.extensions.ExtensionDtype]] = None,
        name: Optional[Hashable] = None,
        copy: Optional[bool] = True,
        identifier: Optional[str] = None,
        x_max_resolution: Optional[float] = None,
        y_max_resolution: Optional[float] = None,
        z_max_resolution: Optional[float] = None,
    ) -> None:
        super().__init__(data=data, index=index, dtype=dtype, name=name, copy=copy)  # type: ignore[call-arg]
        self.identifier = identifier
        self.x_max_resolution = x_max_resolution
        self.y_max_resolution = y_max_resolution
        self.z_max_resolution = z_max_resolution

    @property
    def _constructor(self):
        return PointCloudSeries

    @property
    def _constructor_expanddim(self):
        return PointCloud
