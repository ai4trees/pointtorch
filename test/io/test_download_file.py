"""Tests for the file download tools in pointtorch.io"""

import os
import pathlib
import shutil
from typing import Optional, Union
import zipfile

import pytest
from pytest_httpserver import HTTPServer
import werkzeug

from pointtorch.io import download_file


class TestDownloadFile:
    """Tests for the file download tools in pointtorch.io"""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestDownloadFile"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.fixture
    def zip_file_path(self, cache_dir: str):
        zip_dir = os.path.join(cache_dir, "zip-contents")
        os.makedirs(zip_dir, exist_ok=True)
        for idx in range(2):
            with open(os.path.join(zip_dir, f"test{idx}.txt"), "w", encoding="utf-8") as file:
                file.write(f"Test{idx}")

        zip_file_path = os.path.join(cache_dir, "test.zip")
        shutil.make_archive(zip_file_path.rstrip(".zip"), "zip", zip_dir)
        yield zip_file_path

    @pytest.mark.parametrize("progress_bar,progress_bar_desc", [(False, None), (True, None), (True, "test")])
    @pytest.mark.parametrize("provide_content_length_header", [True, False])
    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_valid_file(
        self,
        progress_bar: bool,
        progress_bar_desc: Optional[str],
        provide_content_length_header: bool,
        use_pathlib: bool,
        zip_file_path: str,
        cache_dir: str,
        httpserver: HTTPServer,
    ):
        def send_zip_file(request: werkzeug.Request) -> werkzeug.Response:
            response = werkzeug.utils.send_file(
                zip_file_path, request.environ, mimetype="application/zip", as_attachment=True
            )
            if not provide_content_length_header:
                response.headers.pop("Content-Length", None)

            return response

        httpserver.expect_request("/zipfile", method="GET").respond_with_handler(send_zip_file)

        file_path: Union[str, pathlib.Path] = os.path.join(cache_dir, "downloaded.zip")
        if use_pathlib:
            file_path = pathlib.Path(file_path)

        download_file(
            httpserver.url_for("/zipfile"), file_path, progress_bar=progress_bar, progress_bar_desc=progress_bar_desc
        )

        assert os.path.exists(file_path)

        unzip_dir = os.path.join(cache_dir, "unzipped")

        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(unzip_dir)

        file_path_1 = os.path.join(unzip_dir, "test1.txt")
        assert os.path.exists(file_path_1)

        with open(file_path_1, "r", encoding="utf-8") as file:
            file_content = file.read()
            assert "Test1" == file_content

    def test_download_invalid_url(self, cache_dir: str):
        with pytest.raises(RuntimeError):
            download_file("http://broken-url.", os.path.join(cache_dir, "downloaded.zip"))

    def test_not_found(self, cache_dir: str, httpserver: HTTPServer):
        httpserver.expect_request("/zipfile", method="GET").respond_with_data(
            "Not found", status=404, content_type="text/plain"
        )

        with pytest.raises(RuntimeError):
            download_file(httpserver.url_for("/zipfile"), os.path.join(cache_dir, "downloaded.zip"))
