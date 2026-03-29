"""Tests for the unzipping tools in pointtorch.io"""

import os
import pathlib
import shutil
from typing import Union
import zipfile

import pytest

from pointtorch.io import unzip


class TestUnzip:
    """Tests for the unzipping tools in pointtorch.io"""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestUnzip"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.fixture
    def zip_file_path(self, cache_dir: str) -> str:
        zip_dir = os.path.join(cache_dir, "zip-contents")
        os.makedirs(zip_dir, exist_ok=True)
        os.makedirs(os.path.join(zip_dir, "test"), exist_ok=True)
        for idx in range(2):
            subdir = zip_dir if idx == 0 else os.path.join(zip_dir, "test")
            with open(os.path.join(subdir, f"test{idx}.txt"), "w", encoding="utf-8") as file:
                file.write(f"Test{idx}")

        zip_file_path = os.path.join(cache_dir, "test.zip")
        shutil.make_archive(zip_file_path.rstrip(".zip"), "zip", zip_dir)
        yield zip_file_path

    @staticmethod
    def _raise_not_implemented(*args, **kwargs):
        raise NotImplementedError

    @pytest.mark.parametrize("progress_bar,progress_bar_desc", [(False, None), (True, None), (True, "test")])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    @pytest.mark.parametrize("use_stream_unzip", [True, False])
    def test_valid_file(
        self,
        progress_bar: bool,
        progress_bar_desc: str,
        zip_file_path: Union[str, pathlib.Path],
        cache_dir: Union[str, pathlib.Path],
        use_pathlib: bool,
        use_stream_unzip: bool,
        monkeypatch: pytest.MonkeyPatch,
    ):

        if use_stream_unzip:
            monkeypatch.setattr(zipfile, "ZipFile", self._raise_not_implemented)

        if use_pathlib:
            zip_file_path = pathlib.Path(zip_file_path)
            cache_dir = pathlib.Path(cache_dir)

        unzip(zip_file_path, cache_dir, progress_bar=progress_bar, progress_bar_desc=progress_bar_desc)

        file_path_0 = os.path.join(cache_dir, "test0.txt")
        file_path_1 = os.path.join(cache_dir, "test/test1.txt")

        assert os.path.exists(file_path_0)
        assert os.path.exists(file_path_1)

        with open(file_path_0, "r", encoding="utf-8") as file:
            file_content = file.read()
            assert "Test0" == file_content

        with open(file_path_1, "r", encoding="utf-8") as file:
            file_content = file.read()
            assert "Test1" == file_content

    @pytest.mark.parametrize("use_stream_unzip", [True, False])
    def test_selected_items(
        self,
        use_stream_unzip: bool,
        zip_file_path: Union[str, pathlib.Path],
        cache_dir: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        if use_stream_unzip:
            monkeypatch.setattr(zipfile, "ZipFile", self._raise_not_implemented)

        cache_dir = pathlib.Path(cache_dir)

        unzip(zip_file_path, cache_dir, items=["test/test1.txt"])

        file_path_0 = cache_dir / "test0.txt"
        file_path_1 = cache_dir / "test/test1.txt"

        assert not file_path_0.exists()
        assert file_path_1.exists()

    @pytest.mark.parametrize("use_stream_unzip", [True, False])
    def test_invalid_items(
        self,
        use_stream_unzip: bool,
        zip_file_path: Union[str, pathlib.Path],
        cache_dir: str,
        monkeypatch: pytest.MonkeyPatch,
    ):

        if use_stream_unzip:
            monkeypatch.setattr(zipfile, "ZipFile", self._raise_not_implemented)

        with pytest.raises(KeyError):
            unzip(zip_file_path, cache_dir, items=["non-existing-item"])

    @pytest.mark.parametrize("use_stream_unzip", [True, False])
    def test_file_not_existing(self, use_stream_unzip: bool, cache_dir: str, monkeypatch: pytest.MonkeyPatch):
        if use_stream_unzip:
            monkeypatch.setattr(zipfile, "ZipFile", self._raise_not_implemented)

        with pytest.raises(FileNotFoundError):
            unzip(os.path.join(cache_dir, "test.zip"), cache_dir)
