"""Utilities for downloading files."""

__all__ = ["download_file"]

import http.client
import logging
import pathlib
import time
from typing import Dict, Optional, Union
from urllib import request, error as urllib_error

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _is_retryable(error: BaseException) -> bool:
    """
    Checks whether an error raised while downloading a file indicates a transient network issue (e.g., a stalled or
    interrupted connection) for which retrying is likely to help, or a permanent failure (e.g., an invalid URL or a
    404 response).

    Args:
        error: The error raised while downloading the file.

    Returns:
        :code:`True` if the download should be retried, :code:`False` otherwise.
    """

    if isinstance(error, (TimeoutError, ConnectionError, http.client.HTTPException)):
        return True

    # a plain URLError can wrap a transient error that occurred while establishing the connection, such as a timeout
    if isinstance(error, urllib_error.URLError) and not isinstance(error, urllib_error.HTTPError):
        return isinstance(error.reason, (TimeoutError, ConnectionError))

    return False


def _format_error(error: BaseException) -> str:
    """
    Args:
        error: The error raised while downloading the file.

    Returns:
        A human-readable description of the error.
    """

    if isinstance(error, urllib_error.HTTPError):
        return f"{error.reason}, Status: {error.code}"
    if isinstance(error, urllib_error.URLError):
        return str(error.reason)

    return str(error)


def download_file(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    url: str,
    file_path: Union[str, pathlib.Path],
    *,
    progress_bar: bool = True,
    progress_bar_desc: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 10,
    retry_backoff_seconds: float = 5.0,
    block_size: int = 1024 * 8,
) -> None:
    """
    Downloads a file via HTTP.

    A read timeout is used to detect connections that stall (e.g., because a the server silently drops an idle
    connection without closing it properly). If the connection stalls or is otherwise interrupted, the download is
    retried using an HTTP ``Range`` request so that only the missing part of the file has to be downloaded again.
    Permanent failures (e.g., an invalid URL) are not retried.

    Args:
        url: The URL of the file to download.
        file_path: Path where to save the downloaded file.
        progress_bar: Whether a progress bar should be created to show the download progress. Defaults to `True`.
        progress_bar_desc: Description of the progress bar. Only used if :attr:`progress_bar` is `True`. Defaults to
            `None`.
        timeout: Number of seconds to wait for the server to send further data before assuming that the connection
            has stalled. Defaults to `30.0`.
        max_retries: Maximum number of times to retry the download after a stalled or interrupted connection before
            giving up. Defaults to `10`.
        retry_backoff_seconds: Number of seconds to wait before retrying after a stalled or interrupted connection.
            Defaults to `5.0`.
        block_size: Number of bytes to read from the connection per chunk. Defaults to `1024 * 8`.

    Raises:
        RuntimeError: If the file download keeps failing even after :code:`max_retries` retries.
    """

    file_path = pathlib.Path(file_path)

    total_size: Optional[int] = None
    downloaded_bytes = file_path.stat().st_size if file_path.exists() else 0
    prog_bar: Optional[tqdm] = None
    progress_bar_desc_logged = False
    attempt = 0

    try:
        while True:
            headers: Dict[str, str] = {"Range": f"bytes={downloaded_bytes}-"} if downloaded_bytes > 0 else {}
            http_request = request.Request(url, headers=headers)

            try:
                with request.urlopen(http_request, timeout=timeout) as response:
                    # some servers do not support range requests and always return the full file starting at byte 0,
                    # even if a `Range` header was sent
                    resumed = downloaded_bytes > 0 and response.status == 206

                    if total_size is None:
                        content_length = response.info().get("Content-Length")
                        if content_length is not None:
                            total_size = downloaded_bytes + int(content_length) if resumed else int(content_length)
                            if progress_bar:
                                prog_bar = tqdm(
                                    desc=progress_bar_desc,
                                    total=total_size,
                                    unit="B",
                                    unit_scale=True,
                                    unit_divisor=1000,
                                )
                                prog_bar.update(downloaded_bytes)
                        elif progress_bar and not progress_bar_desc_logged:
                            # if the server does not report a content length, the total download size is unknown, so
                            # no progress bar can be shown
                            logger.info(progress_bar_desc)
                            progress_bar_desc_logged = True

                    if downloaded_bytes > 0 and not resumed:
                        downloaded_bytes = 0
                        if prog_bar is not None:
                            prog_bar.reset(total=total_size)

                    with open(file_path, "ab" if resumed else "wb") as out_file:
                        while True:
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded_bytes += len(chunk)
                            if prog_bar is not None:
                                prog_bar.update(len(chunk))

                if total_size is None or downloaded_bytes >= total_size:
                    break

                # the connection was closed before the full file was received; retry for the remaining bytes
                raise ConnectionError(
                    f"The connection was closed before the full file was downloaded ({downloaded_bytes}/{total_size} "
                    "bytes received)."
                )
            except (urllib_error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
                # HTTP 416 ("Range Not Satisfiable") indicates that the requested byte range starts beyond the end of
                # the file, i.e., that the file on disk is already complete from a previous download
                if (
                    downloaded_bytes > 0
                    and isinstance(error, urllib_error.HTTPError)
                    and error.code == 416  # pylint: disable=no-member
                ):
                    break

                if not _is_retryable(error):
                    raise RuntimeError(f"Downloading data from {url} failed ({_format_error(error)}).") from error

                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Downloading data from {url} failed after {max_retries} retries ({_format_error(error)})."
                    ) from error

                if prog_bar is not None:
                    retry_suffix = f"[attempt {attempt}/{max_retries}]"
                    prog_bar.set_description(
                        f"{progress_bar_desc} {retry_suffix}" if progress_bar_desc else retry_suffix
                    )
                else:
                    logger.warning(
                        "Download from %s stalled or was interrupted (%s). Retrying (%d/%d) in %.0f seconds.",
                        url,
                        _format_error(error),
                        attempt,
                        max_retries,
                        retry_backoff_seconds,
                    )
                time.sleep(retry_backoff_seconds)
                downloaded_bytes = file_path.stat().st_size if file_path.exists() else 0
    finally:
        if prog_bar is not None:
            prog_bar.close()
