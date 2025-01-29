"""Tests for the pointtorch.core.PointCloud class."""

import numpy as np
import pandas as pd
import pytest

from pointtorch import PointCloud, PointCloudSeries


class TestPointCloudSeries:
    """
    Tests for the pointtorch.core.PointCloudSeries class.
    """

    @pytest.fixture
    def data(self):
        return np.array([0, 1, 2])

    @pytest.fixture
    def identifier(self):
        return "test_point_cloud"

    @pytest.fixture
    def point_cloud_series(self, data, identifier):
        return PointCloudSeries(data, identifier=identifier)

    def test_slicing(self, point_cloud_series, identifier):
        point_cloud_slice = point_cloud_series[point_cloud_series > 0]
        assert isinstance(point_cloud_slice, PointCloudSeries)
        assert identifier == point_cloud_slice.identifier

    def test_expansion(self, point_cloud_series):
        point_cloud = pd.concat([point_cloud_series, point_cloud_series], axis=1, ignore_index=True)
        assert isinstance(point_cloud, PointCloud)

    def test_identifier(self, point_cloud_series, identifier):
        assert identifier == point_cloud_series.identifier

    def test_x_max_resolution(self, data):
        x_max_resolution = 0.1
        point_cloud_series = PointCloudSeries(data, x_max_resolution=x_max_resolution)
        assert x_max_resolution == point_cloud_series.x_max_resolution

    def test_y_max_resolution(self, data):
        y_max_resolution = 0.1
        point_cloud_series = PointCloudSeries(data, y_max_resolution=y_max_resolution)
        assert y_max_resolution == point_cloud_series.y_max_resolution

    def test_z_max_resolution(self, data):
        z_max_resolution = 0.1
        point_cloud_series = PointCloudSeries(data, z_max_resolution=z_max_resolution)
        assert z_max_resolution == point_cloud_series.z_max_resolution
