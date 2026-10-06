"""Tests for the RAR extraction tools in pointtorch.io"""

import os
import pathlib
import shutil
import struct
from typing import Dict, Union
import zlib

import pytest

from pointtorch.io import unrar


class TestUnrar:
    """Tests for the RAR extraction tools in pointtorch.io"""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestUnrar"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @staticmethod
    def _rar_block(head_type: int, flags: int, body: bytes, data: bytes = b"") -> bytes:
        head = struct.pack("<BHH", head_type, flags, 7 + len(body)) + body
        return struct.pack("<H", zlib.crc32(head) & 0xFFFF) + head + data

    @staticmethod
    def _write_rar_file(rar_file_path: str, files: Dict[str, bytes], compression_method: int = 0x30) -> None:
        """
        Writes a RAR 4 archive. By default, the files are stored without compression, so that the archive can be read
        by :code:`rarfile` without any external extraction tool, which cannot be assumed to be installed in the test
        environment.
        """

        rar_data = b"Rar!\x1a\x07\x00" + TestUnrar._rar_block(0x73, 0, b"\x00" * 6)
        for file_name, file_data in files.items():
            encoded_file_name = file_name.encode("utf-8")
            file_header = (
                struct.pack(
                    "<IIBIIBBHI",
                    len(file_data),
                    len(file_data),
                    3,
                    zlib.crc32(file_data),
                    0x21,
                    20,
                    compression_method,
                    len(encoded_file_name),
                    0o100644 << 16,
                )
                + encoded_file_name
            )
            rar_data += TestUnrar._rar_block(0x74, 0x8000, file_header, file_data)
        rar_data += TestUnrar._rar_block(0x7B, 0x4000, b"")

        with open(rar_file_path, "wb") as file:
            file.write(rar_data)

    @pytest.fixture
    def rar_file_path(self, cache_dir: str) -> str:
        rar_file_path = os.path.join(cache_dir, "test.rar")
        self._write_rar_file(rar_file_path, {"test0.txt": b"Test0", "test/test1.txt": b"Test1"})
        return rar_file_path

    @pytest.mark.parametrize("progress_bar,progress_bar_desc", [(False, None), (True, None), (True, "test")])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_valid_file(
        self,
        progress_bar: bool,
        progress_bar_desc: str,
        rar_file_path: Union[str, pathlib.Path],
        cache_dir: Union[str, pathlib.Path],
        use_pathlib: bool,
    ):
        if use_pathlib:
            rar_file_path = pathlib.Path(rar_file_path)
            cache_dir = pathlib.Path(cache_dir)

        unrar(rar_file_path, cache_dir, progress_bar=progress_bar, progress_bar_desc=progress_bar_desc)

        file_path_0 = os.path.join(cache_dir, "test0.txt")
        file_path_1 = os.path.join(cache_dir, "test/test1.txt")

        assert os.path.exists(file_path_0)
        assert os.path.exists(file_path_1)

        with open(file_path_0, "r", encoding="utf-8") as file:
            assert "Test0" == file.read()

        with open(file_path_1, "r", encoding="utf-8") as file:
            assert "Test1" == file.read()

    def test_selected_items(self, rar_file_path: str, cache_dir: str):
        cache_dir_path = pathlib.Path(cache_dir)

        unrar(rar_file_path, cache_dir_path, items=["test/test1.txt"])

        assert not (cache_dir_path / "test0.txt").exists()
        assert (cache_dir_path / "test/test1.txt").exists()

    def test_invalid_items(self, rar_file_path: str, cache_dir: str):
        with pytest.raises(KeyError):
            unrar(rar_file_path, cache_dir, items=["non-existing-item"])

    def test_file_not_existing(self, cache_dir: str):
        with pytest.raises(FileNotFoundError):
            unrar(os.path.join(cache_dir, "test.rar"), cache_dir)

    def test_invalid_rar_file(self, cache_dir: str):
        rar_file_path = os.path.join(cache_dir, "test.rar")
        with open(rar_file_path, "wb") as file:
            file.write(b"Invalid")

        with pytest.raises(RuntimeError):
            unrar(rar_file_path, cache_dir)

    def test_extraction_failure(self, cache_dir: str):
        rar_file_path = os.path.join(cache_dir, "test.rar")
        self._write_rar_file(rar_file_path, {"test0.txt": b"invalid compressed data"}, compression_method=0x33)

        with pytest.raises(RuntimeError):
            unrar(rar_file_path, cache_dir, progress_bar=False)

    def test_item_outside_target_directory(self, cache_dir: str):
        rar_file_path = os.path.join(cache_dir, "test.rar")
        self._write_rar_file(rar_file_path, {"../outside.txt": b"Test"})

        with pytest.raises(RuntimeError):
            unrar(rar_file_path, os.path.join(cache_dir, "dest"), progress_bar=False)

        assert not os.path.exists(os.path.join(cache_dir, "outside.txt"))
