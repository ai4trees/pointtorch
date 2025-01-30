"""Tests for the pointtorch.core.BoundingBox class."""

import pytest

import numpy

from pointtorch import BoundingBox


class TestBoundingBox:
    """Tests for the pointtorch.core.BoundingBox class."""

    def test_different_shapes_for_min_and_max(self):
        with pytest.raises(ValueError):
            BoundingBox(numpy.zeros(3), numpy.ones(2))

    def test_min_larger_than_max(self):
        with pytest.raises(ValueError):
            BoundingBox(numpy.ones(3), numpy.zeros(3))

    def test_min_max(self):
        min_coords = numpy.zeros(3)
        max_coords = numpy.ones(3)
        bounding_box = BoundingBox(min_coords, max_coords)

        assert (min_coords == bounding_box.min).all()
        assert (max_coords == bounding_box.max).all()

    def test_center(self):
        bounding_box = BoundingBox(numpy.zeros(3), numpy.ones(3))
        expected_center = numpy.ones(3) * 0.5

        assert (expected_center == bounding_box.center()).all()

    def test_extent(self):
        bounding_box = BoundingBox(numpy.zeros(3), numpy.ones(3))
        expected_extent = numpy.ones(3)

        assert (expected_extent == bounding_box.extent()).all()
