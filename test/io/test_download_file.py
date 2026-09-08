"""Tests for the file download tools in pointtorch.io"""

import os
import pathlib
import shutil
import threading
import time
from typing import Optional, Union
import zipfile

import pytest
from pytest_httpserver import HTTPServer
import werkzeug

from pointtorch.io import download_file


def _flaky_handler(
    content: bytes,
    fail_after_bytes: int,
    stall_seconds: float = 0.0,
    always_stall: bool = False,
    ignore_range: bool = False,
):
    """
    Builds a ``pytest_httpserver`` handler that serves :attr:`content` but stops sending data after
    :attr:`fail_after_bytes` bytes (relative to the requested range) without closing the connection, simulating a
    connection that stalls, e.g., because an idle connection was silently dropped. Unless :attr:`always_stall` is
    set, this only happens on the first request for the full file; subsequent requests (including ``Range`` requests
    used to resume the download) are served in full.

    Args:
        content: The full file content to serve.
        fail_after_bytes: Number of bytes (relative to the requested range) to send before stalling.
        stall_seconds: Number of seconds to sleep before returning from a stalled request. Defaults to `0.0`, i.e.,
            the request never completes on its own and relies on the client's read timeout.
        always_stall: Whether every request (not just the first one) should stall. Defaults to `False`.
        ignore_range: Whether to ignore ``Range`` headers and always respond with the full file starting at byte 0,
            simulating a server that does not support resuming downloads. Defaults to `False`.

    Returns:
        Tuple of the handler function to pass to ``httpserver.expect_request(...).respond_with_handler(...)`` and a
        dict tracking how many requests the handler has served (under the ``"attempts"`` key).
    """

    state = {"attempts": 0}
    lock = threading.Lock()

    def handle(request: werkzeug.Request) -> werkzeug.Response:
        range_header = request.headers.get("Range")
        requested_start = int(range_header.split("=")[1].split("-")[0]) if range_header else 0
        start = 0 if ignore_range else requested_start

        with lock:
            state["attempts"] += 1
            attempt = state["attempts"]

        remaining = content[start:]
        should_stall = always_stall or (attempt == 1 and start == 0)

        def body():
            if should_stall and fail_after_bytes < len(remaining):
                yield remaining[:fail_after_bytes]
                if stall_seconds > 0:
                    time.sleep(stall_seconds)
                return
            yield remaining

        headers = {"Content-Length": str(len(remaining))}
        status = 200
        if start > 0:
            status = 206
            headers["Content-Range"] = f"bytes {start}-{len(content) - 1}/{len(content)}"

        return werkzeug.Response(body(), status=status, headers=headers, direct_passthrough=True)

    return handle, state


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

    @pytest.mark.parametrize("progress_bar", [False, True])
    def test_resumes_after_stalled_connection(self, progress_bar: bool, cache_dir: str, httpserver: HTTPServer):
        content = os.urandom(200_000)
        handler, state = _flaky_handler(content, fail_after_bytes=150_000)
        httpserver.expect_request("/file", method="GET").respond_with_handler(handler)

        file_path = os.path.join(cache_dir, "downloaded.bin")
        download_file(
            httpserver.url_for("/file"),
            file_path,
            progress_bar=progress_bar,
            timeout=2.0,
            max_retries=3,
            retry_backoff_seconds=0.1,
        )

        with open(file_path, "rb") as file:
            assert file.read() == content
        # the download must actually have been interrupted and retried, not merely succeeded on the first try
        assert state["attempts"] >= 2

    def test_restarts_from_scratch_when_server_does_not_support_range_requests(
        self, cache_dir: str, httpserver: HTTPServer
    ):
        content = os.urandom(200_000)
        handler, state = _flaky_handler(content, fail_after_bytes=150_000, ignore_range=True)
        httpserver.expect_request("/file", method="GET").respond_with_handler(handler)

        file_path = os.path.join(cache_dir, "downloaded.bin")
        download_file(
            httpserver.url_for("/file"),
            file_path,
            progress_bar=True,
            timeout=2.0,
            max_retries=3,
            retry_backoff_seconds=0.1,
        )

        with open(file_path, "rb") as file:
            assert file.read() == content
        assert state["attempts"] >= 2

    def test_completes_when_file_already_fully_downloaded(self, cache_dir: str, httpserver: HTTPServer):
        # if the file at `file_path` is already complete (e.g., from a previous run), the `Range` request for the
        # remaining bytes starts beyond the end of the file, which the server answers with HTTP 416 ("Range Not
        # Satisfiable"); this must be treated as a successful completion rather than an error
        content = os.urandom(50_000)

        def handle(request: werkzeug.Request) -> werkzeug.Response:
            range_header = request.headers.get("Range")
            start = int(range_header.split("=")[1].split("-")[0]) if range_header else 0
            if start >= len(content):
                return werkzeug.Response(status=416)
            return werkzeug.Response(content[start:], status=200, headers={"Content-Length": str(len(content))})

        httpserver.expect_request("/file", method="GET").respond_with_handler(handle)

        file_path = os.path.join(cache_dir, "downloaded.bin")
        with open(file_path, "wb") as file:
            file.write(content)

        download_file(httpserver.url_for("/file"), file_path, progress_bar=False)

        with open(file_path, "rb") as file:
            assert file.read() == content

    def test_gives_up_after_max_retries(self, cache_dir: str, httpserver: HTTPServer):
        content = os.urandom(200_000)
        # never send any data on any attempt (including retries), so that all retries are exhausted
        handler, state = _flaky_handler(content, fail_after_bytes=0, always_stall=True)
        httpserver.expect_request("/file", method="GET").respond_with_handler(handler)

        file_path = os.path.join(cache_dir, "downloaded.bin")
        start_time = time.monotonic()
        with pytest.raises(RuntimeError):
            download_file(
                httpserver.url_for("/file"),
                file_path,
                progress_bar=False,
                timeout=0.5,
                max_retries=2,
                retry_backoff_seconds=0.1,
            )
        elapsed = time.monotonic() - start_time
        # must fail once retries are exhausted instead of hanging indefinitely
        assert elapsed < 15.0
        assert state["attempts"] == 3  # the initial attempt plus 2 retries

    def test_download_invalid_url(self, cache_dir: str):
        with pytest.raises(RuntimeError):
            download_file("http://broken-url.", os.path.join(cache_dir, "downloaded.zip"))

    def test_not_found(self, cache_dir: str, httpserver: HTTPServer):
        httpserver.expect_request("/zipfile", method="GET").respond_with_data(
            "Not found", status=404, content_type="text/plain"
        )

        start_time = time.monotonic()
        with pytest.raises(RuntimeError):
            download_file(httpserver.url_for("/zipfile"), os.path.join(cache_dir, "downloaded.zip"))
        download_time = time.monotonic() - start_time

        # a 404 response is a permanent failure and must not be retried and should fail immediately
        assert download_time < 5.0
