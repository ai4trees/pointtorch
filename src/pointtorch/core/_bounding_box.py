"""Axis-aligned bounding box."""

__all__ = ["BoundingBox"]

import numpy


class BoundingBox:
    """
    Axis-aligned bounding box.

    Args:
        min_coord: Minimum coordinates of the bounding box.
        max_coord: Maximum coordinates of the bounding box.

    Raise:
        ValueError: If :attr:`min_coord` and :attr:`max_coord` have different shapes or if an entry of :attr:`min_coord`
            is larger then the corresponding entry in :attr:`max_coord`.
    """

    def __init__(self, min_coord: numpy.ndarray, max_coord: numpy.ndarray):
        if min_coord.shape != max_coord.shape:
            raise ValueError("The minimum and maximum coordinates of the bounding box must have the same shape.")
        if (min_coord > max_coord).any():
            raise ValueError("The minimum coordinates of the bounding box must be smaller than the maximum coordintes.")
        self._min = min_coord
        self._max = max_coord

    @property
    def min(self) -> numpy.ndarray:
        """
        Returns: Minimum coordinates of the bounding box.
        """
        return self._min

    @property
    def max(self) -> numpy.ndarray:
        """
        Returns: Maximum coordinates of the bounding box.
        """
        return self._max

    def center(self) -> numpy.ndarray:
        """
        Returns: Coordinates of the bounding box center.
        """
        return self._min + self.extent() / 2

    def extent(self) -> numpy.ndarray:
        """
        Returns: Extent of the bounding box along each axis.
        """
        return self._max - self._min
