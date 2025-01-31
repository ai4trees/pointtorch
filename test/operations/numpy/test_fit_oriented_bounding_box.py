"""Tests for the pointtorch.operations.fit_oriented_bounding_box method."""

import numpy

import pytest

from pointtorch.operations.numpy import fit_oriented_bounding_box


class TestFitOrientedBoundingBox:
    """
    Tests for the pointtorch.operations.fit_oriented_bounding_box method.
    """

    @pytest.mark.parametrize("dim", [2, 3])
    def test_bounding_box_orientation(self, dim: int):
        coords = numpy.array([[-1, 1, 0], [1, -1, 0], [4, 2, 0], [2, 4, 0]])
        bounding_box, transformation_matrix, transformed_coords = fit_oriented_bounding_box(coords, dim)

        assert bounding_box.max[0] - bounding_box.min[0] == pytest.approx(numpy.sqrt(18))
        assert bounding_box.max[1] - bounding_box.min[1] == pytest.approx(numpy.sqrt(8))

        assert transformed_coords[0, 0] == pytest.approx(transformed_coords[1, 0])
        assert transformed_coords[2, 0] == pytest.approx(transformed_coords[3, 0])
        assert numpy.abs(transformed_coords[0, 0] - transformed_coords[2, 0]) == pytest.approx(numpy.sqrt(18))
        assert numpy.abs(transformed_coords[1, 0] - transformed_coords[3, 0]) == pytest.approx(numpy.sqrt(18))
        assert numpy.abs(transformed_coords[0, 1] - transformed_coords[1, 1]) == pytest.approx(numpy.sqrt(8))
        assert numpy.abs(transformed_coords[2, 1] - transformed_coords[3, 1]) == pytest.approx(numpy.sqrt(8))
        assert (transformed_coords[:, 2] == 0).all()

        assert (numpy.matmul(transformation_matrix, coords.T).T == transformed_coords).all()

    @pytest.mark.parametrize("dim", [0, 4])
    def test_invalid_dim(self, dim: int):
        coords = numpy.zeros((4, 3))
        with pytest.raises(ValueError):
            fit_oriented_bounding_box(coords, dim)
