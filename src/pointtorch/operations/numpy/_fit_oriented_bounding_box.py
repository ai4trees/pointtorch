"""Method to compute the oriented bounding box of a point cloud."""

__all__ = ["fit_oriented_bounding_box"]

from typing import Tuple

import numpy
from sklearn.decomposition import PCA

from pointtorch import BoundingBox


def fit_oriented_bounding_box(coords: numpy.ndarray, dim: int) -> Tuple[BoundingBox, numpy.ndarray, numpy.ndarray]:
    r"""
    Computes the oriented bounding box of a point cloud. The principal components of the point distribution are computed
    and used as a coordinate basis. The point coordinates are transformed into the coordinate system spanned by this
    basis and the axis-aligned bounding box is calculated in this coordinate system.

    Args:
        coords: Point coordinates.
        dim: Dimensionality of the bounding box. For example, a 2D bounding box is computed when setting :attr:`dim` to
            2. :attr:`dim` must be greater than 1 and smaller or equal to `D`.

    Returns:
        A tuple of three elements. The first element is the bounding box, which is represented as an axis-aligned\
        bounding box in the transformed coordinate system. The second element is the transformation matrix, which can\
        be used to transform point coordinates from the original coordinate system to the coordinate system spanned by\
        the principal components. The third element is the point coordinates in the transformed coordinate system.

    Shape:
        - | :attr:`coords`: :math:`(N, D)`
        - | Output transformation matrix: :math:`(D, D)`
        - | Output transformed point coordinates: :math:`(N, D)`
          |
          | where
          |
          | :math:`N = \text{ number of points}`
          | :math:`D = \text{ number of coordinate dimensions}`
    """

    if dim <= 0 or dim > coords.shape[1]:
        raise ValueError("Dimensionality of bounding box must be either 2 or 3.")

    coords_pca_dims = coords[:, :dim]
    centroid = coords_pca_dims.mean(axis=0, keepdims=True)
    pca = PCA(n_components=dim).fit(coords_pca_dims - centroid)
    principal_components = pca.components_.T

    if dim < coords.shape[1]:
        transformation_matrix = numpy.eye(coords.shape[1])
        transformation_matrix[:dim, :dim] = numpy.linalg.inv(principal_components)
    else:
        transformation_matrix = numpy.linalg.inv(principal_components)

    transformed_coords = numpy.matmul(transformation_matrix, (coords).T).T

    bounding_box_min = transformed_coords[:, :dim].min(axis=0)
    bounding_box_max = transformed_coords[:, :dim].max(axis=0)

    bounding_box = BoundingBox(bounding_box_min, bounding_box_max)

    return bounding_box, transformation_matrix, transformed_coords
