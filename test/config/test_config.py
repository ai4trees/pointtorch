"""Tests for the pointtorch.config module."""

import sys
from types import ModuleType

from pointtorch.config import open3d_is_available, pytorch3d_is_available, flash_attention_is_available


class TestConfig:
    """Tests for the pointtorch.config module."""

    def test_flash_attention_is_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
        assert flash_attention_is_available()

    def test_flash_attention_not_available(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
        monkeypatch.setattr("sys.path", [])
        assert flash_attention_is_available() is False

    def test_open3d_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "open3d", ModuleType("open3d"))
        monkeypatch.setitem(sys.modules, "open3d.ml", ModuleType("open3d.ml"))
        monkeypatch.setitem(sys.modules, "open3d.ml.torch", ModuleType("open3d.ml.torch"))
        assert open3d_is_available()

    def test_open3d_not_available(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "open3d", raising=False)
        monkeypatch.setattr("sys.path", [])
        assert open3d_is_available() is False

    def test_pytorch3d_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pytorch3d", ModuleType("pytorch3d"))
        assert pytorch3d_is_available()

    def test_pytorch3d_not_available(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "pytorch3d", raising=False)
        monkeypatch.setattr("sys.path", [])
        assert pytorch3d_is_available() is False
