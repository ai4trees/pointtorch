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

    def structured_dtype(self, extra_fields: Optional[list[tuple[str, np.dtype]]] = None) -> np.dtype:
        """
        Builds a structured numpy dtype from the point cloud columns.

        Returns:
            A numpy structured dtype representing the current point cloud columns.

        Args:
            extra_fields: Optional list of additional fields to append to the dtype.
        """

        dtype_fields = [(str(column), self[column].to_numpy().dtype) for column in self.columns]
        if extra_fields is not None:
            dtype_fields.extend(extra_fields)
        return np.dtype(dtype_fields)

    def to_structured_array(self, dtype: Optional[np.dtype] = None) -> npt.NDArray:
        """
        Converts the point cloud to a numpy structured array.

        Args:
            dtype: Structured dtype to use for the output. If not specified, one is derived from the point cloud.

        Returns:
            Structured array containing the point cloud columns.

        Raises:
            ValueError: If :code:`dtype` does not define named fields and the point cloud is non-empty.
        """

        if dtype is None:
            dtype = self.structured_dtype()

        structured_array = np.empty(len(self), dtype=dtype)
        if dtype.names is None:
            if len(self) == 0:
                return structured_array
            raise ValueError("The dtype must define named fields.")

        for column in dtype.names:
            if column in self.columns:
                structured_array[column] = self[column].to_numpy(dtype=dtype[column])

        return structured_array

    @classmethod
    def from_structured_array(
        cls,
        structured_array: npt.NDArray,
        *,
        crs: Optional[str] = None,
        identifier: Optional[str] = None,
        x_max_resolution: Optional[float] = None,
        y_max_resolution: Optional[float] = None,
        z_max_resolution: Optional[float] = None,
    ) -> "PointCloud":
        """
        Creates a point cloud from a numpy structured array.

        Args:
            structured_array: Structured array containing the point cloud columns.
            crs: ESPG code of the point cloud's coordinate reference system. Defaults to :code:`None`
            identifier: Point cloud identifier.
            x_max_resolution: Maximum resolution of the point cloud's x-coordinates in meter. Defaults to :code:`None`.
            y_max_resolution: Maximum resolution of the point cloud's y-coordinates in meter. Defaults to :code:`None`.
            z_max_resolution: Maximum resolution of the point cloud's z-coordinates in meter. Defaults to :code:`None`.

        Returns:
            Point cloud built from the given structured array.

        Raises:
            ValueError: If :code:`structured_array` is non-empty and does not define named fields.
        """

        if structured_array.dtype.names is None:
            if len(structured_array) == 0:
                return cls(
                    [],
                    crs=crs,
                    identifier=identifier,
                    x_max_resolution=x_max_resolution,
                    y_max_resolution=y_max_resolution,
                    z_max_resolution=z_max_resolution,
                )
            raise ValueError("The structured array must define named fields.")

        point_cloud_df = pd.DataFrame.from_records(structured_array, columns=list(structured_array.dtype.names))
        return cls(
            point_cloud_df,
            crs=crs,
            identifier=identifier,
            x_max_resolution=x_max_resolution,
            y_max_resolution=y_max_resolution,
            z_max_resolution=z_max_resolution,
        )

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
