""" Utilities for downloading files. """

__all__ = ["DownloadProgressBar", "download_file"]

import logging
import pathlib
from typing import Optional, Union
from urllib import request, error as urllib_error

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DownloadProgressBar:  # pylint: disable=too-few-public-methods
    """
    Progress bar showing the progress of file download.

    Args:
        desc: Label of the progess bar. Defaults to `None`.
    """

    def __init__(self, desc: Optional[str] = None):
        self._desc = desc
        self._progress_bar = None

    def __call__(self, block_num, block_size, total_size):
        """
        Callback to update the progress bar that should be called for each block downloaded.

        Args:
            block_num: The number of blocks already downloaded.
            block_size: Size of the downloaded blocks in bytes.
            total_size: Total size of the file to be downloaded in bytes.
        """

        if self._progress_bar is None:
            self._progress_bar = tqdm(
                desc=self._desc,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1000,
            )

        downloaded = block_num * block_size
        if downloaded < total_size:
            self._progress_bar.update(block_size)
        else:
            self._progress_bar.close()
            self._progress_bar = None


def download_file(
    url: str, file_path: Union[str, pathlib.Path], progress_bar: bool = True, progress_bar_desc: Optional[str] = None
) -> None:
    """
    Downloads a file via HTTP.

    Args:
        url: The URL of the file to download.
        file_path: Path where to save the downloaded file.
        progress_bar: Whether a progress bar should be created to show the download progress. Defaults to `True`.
        progress_bar_desc: Description of the progress bar. Only used if :attr:`progress_bar` is `True`. Defaults to
            `None`.

    Raises:
        RuntimeError: If the file download fails.
    """

    try:
        # only if the server provides a a content-length header, the file size is known in advance
        # therefore, a progress bar is only created when such a header is provided
        has_content_length_header = False
        with request.urlopen(url) as response:
            headers = response.info()
            if headers.get("Content-Length") is not None:
                has_content_length_header = True
        prog_bar = DownloadProgressBar(desc=progress_bar_desc) if (progress_bar and has_content_length_header) else None
        if progress_bar and prog_bar is None:
            logger.info(progress_bar_desc)

        file_path, _ = request.urlretrieve(url, file_path, prog_bar)
    except (urllib_error.URLError, urllib_error.HTTPError) as error:
        if isinstance(error, urllib_error.HTTPError):
            reason = f"{error.reason}, Status: {error.status}"
        else:
            reason = str(error.reason)
        raise RuntimeError(f"Downloading data from {url} failed ({reason}).") from error
