"""Utilities for extracting files from zip archives."""

__all__ = ["unzip"]

import pathlib
from shutil import copyfileobj
from typing import IO, List, Optional, Union
import zipfile_deflate64  # pylint: disable: unused-import # needed to patch zipfile
import zipfile

from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper


def unzip(
    zip_path: Union[str, pathlib.Path],
    dest_path: Union[str, pathlib.Path],
    items: Optional[List[str]] = None,
    progress_bar: bool = True,
    progress_bar_desc: Optional[str] = None,
):
    """
    Extracts all files from a zip archive.

    Args:
        zip_path: Path of the zip archive.
        dest_path: Path of the directory in which to save the extracted files.
        items: Names of the items to extract. Defaults to :code:`None`, which means that all items are extracted.
        progress_bar: Whether a progress bar should be created to show the extraction progress. Defaults to
            :code:`True`.
        progress_bar_desc: Description of the progress bar. Only used if :code:`progress_bar` is :code:`True`. Defaults
            to :code:`None`.

    Raises:
        FileNotFoundError: If the zip file does not exist.
        KeyError: If :code:`items` contains items not existing in the zip archive.
    """
    if isinstance(dest_path, str):
        dest_path = pathlib.Path(dest_path)

    with zipfile.ZipFile(zip_path) as zip_file:
        total_size = sum(getattr(item, "file_size", 0) for item in zip_file.infolist())
        if progress_bar:
            prog_bar = tqdm(desc=progress_bar_desc, unit="B", unit_scale=True, unit_divisor=1000, total=total_size)
        else:
            prog_bar = None

        if items is not None:
            valid_items = [item.filename for item in zip_file.infolist()]
            invalid_items = [item for item in items if item not in valid_items]
            if len(invalid_items) > 0:
                raise KeyError(f"The following items are not contained in the zipfile: {invalid_items}.")

        for item in zip_file.infolist():
            if items is not None and item.filename not in items:
                continue
            if not getattr(item, "file_size", 0):  # the item is a directory
                zip_file.extract(item, dest_path)
            else:
                file_path = dest_path / item.filename
                file_path.parent.mkdir(exist_ok=True, parents=True)
                with zip_file.open(item) as in_file, open(dest_path / item.filename, "wb") as out_file:
                    file_reader: Union[CallbackIOWrapper, IO[bytes]]
                    if prog_bar is not None:
                        file_reader = CallbackIOWrapper(prog_bar.update, in_file)
                    else:
                        file_reader = in_file
                    copyfileobj(file_reader, out_file)
