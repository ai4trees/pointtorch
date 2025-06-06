"""Point cloud object."""

__all__ = ["PointCloud", "PointCloudSeries"]

from collections.abc import Hashable
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from pointtorch.io import PointCloudIoData, PointCloudWriter
from pointtorch.type_aliases import LongArray


class PointCloud(pd.DataFrame):
    """Point cloud object. Subclass of
    `pd.DataFrame <https://pd.pydata.org/docs/reference/api/pd.DataFrame.html>`_

    Args:
        identifier: Point cloud identifier. Defaults to :code:`None`.
        crs: ESPG code of the point cloud's coordinate reference system. Defaults to :code:`None`
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to :code:`None`.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to :code:`None`.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to :code:`None`.

    For a documentation of other parameters, see the documentation of \
   `pd.DataFrame <https://pd.pydata.org/docs/reference/api/pd.DataFrame.html>`_.

    Attributes:
        crs: ESPG code of the point cloud's coordinate reference system.
        identifier: Point cloud identifier.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter.
    """

    _metadata = ["identifier", "x_max_resolution", "y_max_resolution", "z_max_resolution"]

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        data: Union[npt.ArrayLike, Iterable, dict, pd.DataFrame],
        index: Optional[Union[pd.Index, LongArray]] = None,
        columns: Optional[Union[pd.Index, npt.ArrayLike, list[str]]] = None,
        dtype: Optional[np.dtype] = None,
        copy: Optional[bool] = True,
        crs: Optional[str] = None,
        identifier: Optional[str] = None,
        x_max_resolution: Optional[float] = None,
        y_max_resolution: Optional[float] = None,
        z_max_resolution: Optional[float] = None,
    ) -> None:
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)  # type: ignore[call-arg]
        self.crs = crs
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

    def xyz(self) -> npt.NDArray[np.float64]:
        """
        Returns:
            x, y, and z coordinates of the points in the point cloud.

        Raises:
            RuntimeError: if "x", "y", or "z" are not in :code:`self.columns`.
        """

        if "x" not in self.columns or "y" not in self.columns or "z" not in self.columns:
            raise RuntimeError("The point cloud does not contain xyz coordinates.")

        return self[["x", "y", "z"]].astype(np.float64).to_numpy()

    def to(self, file_path: Union[str, Path], columns: Optional[list[str]] = None) -> None:
        """
        Writes the point cloud to a file.

        Args:
            file_path: Path of the output file.
            columns: Point cloud columns to be written. The x, y, and z columns are always written.

        Raises:
            ValueError: If the point cloud format is not supported by the writer or if :code:`columns` contains a column
                name that is not existing in the point cloud.
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
        identifier: Point cloud identifier. Defaults to :code:`None`.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to :code:`None`.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to :code:`None`.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to :code:`None`.

    For a documentation of other parameters, see the documentation of \
        `pd.Series <https://pd.pydata.org/pandas-docs/stable/reference/api/pd.Series.html>`_.


    Attributes:
        identifier: Point cloud identifier.
        x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter.
        y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter.
        z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter.
    """

    _metadata = ["identifier", "x_max_resolution", "y_max_resolution", "z_max_resolution"]

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        data: Optional[Union[npt.ArrayLike, Iterable, dict, int, float, str]] = None,
        index: Optional[Union[pd.Index, LongArray]] = None,
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
