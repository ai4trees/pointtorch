"""Utilities for extracting files from zip archives."""

__all__ = ["unzip"]

import pathlib
from shutil import copyfileobj
from typing import BinaryIO, IO, Iterable, List, Optional, Tuple, Union
import zipfile

from stream_unzip import stream_unzip
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper


def unzip(
    zip_path: Union[str, pathlib.Path],
    dest_path: Union[str, pathlib.Path],
    items: Optional[List[str]] = None,
    progress_bar: bool = True,
    progress_bar_desc: Optional[str] = None,
) -> None:
    """
    Extract files from a zip archive.

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

    if isinstance(zip_path, str):
        zip_path = pathlib.Path(zip_path)

    try:
        _unzip_with_zipfile(zip_path, dest_path, items, progress_bar=progress_bar, progress_bar_desc=progress_bar_desc)
    except NotImplementedError:
        _unzip_with_stream_unzip(
            zip_path, dest_path, items, progress_bar=progress_bar, progress_bar_desc=progress_bar_desc
        )


def _unzip_with_zipfile(  # pylint: disable=too-many-branches
    zip_path: pathlib.Path,
    dest_path: pathlib.Path,
    items: Optional[List[str]] = None,
    progress_bar: bool = True,
    progress_bar_desc: Optional[str] = None,
) -> None:
    """
    Extracts all files from a zip archive using :code:`zipfile`.
    """

    with zipfile.ZipFile(zip_path) as zip_file:
        total_size = 0
        for item in zip_file.infolist():
            if items is None or item.filename in items:
                total_size += getattr(item, "file_size", 0)
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


def _unzip_with_stream_unzip(  # pylint: disable=too-many-branches
    zip_path: pathlib.Path,
    dest_path: pathlib.Path,
    items: Optional[List[str]] = None,
    progress_bar: bool = True,
    progress_bar_desc: Optional[str] = None,
) -> None:
    """
    Extracts all files from a zip archive using :code:`stream_unzip`.

    This implementation is used as a fallback for compression methods that are
    unsupported by the standard :mod:`zipfile` module.
    """
    remaining_items = None if items is None else set(items)
    total_size = zip_path.stat().st_size
    prog_bar = (
        tqdm(desc=progress_bar_desc, unit="B", unit_scale=True, unit_divisor=1000, total=total_size)
        if progress_bar
        else None
    )

    for item_name, _, chunks in _iter_zip_members(zip_path, progress_bar=prog_bar):
        if remaining_items is not None and item_name not in remaining_items:
            for _ in chunks:
                pass
            continue

        if remaining_items is not None:
            remaining_items.remove(item_name)

        file_path = dest_path / item_name

        if item_name.endswith("/"):
            file_path.mkdir(exist_ok=True, parents=True)
            continue

        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "wb") as out_file:
            for chunk in chunks:
                out_file.write(chunk)

    if remaining_items:
        raise KeyError(f"The following items are not contained in the zipfile: {sorted(remaining_items)}.")


def _iter_zip_members(
    zip_path: pathlib.Path, chunk_size: int = 65536, progress_bar: Optional[tqdm] = None
) -> Iterable[Tuple[str, Optional[int], Iterable[bytes]]]:
    """
    Yield archive members from a zip file.

    Args:
        zip_path: Path to the archive to read.
        chunk_size: Number of bytes to read per iteration from the underlying file object.
        progress_bar: Optional progress bar to update with compressed bytes read.

    Yields:
        Tuples of member name, optional uncompressed member size, and an iterable over the member's uncompressed byte
        chunks.
    """
    with open(zip_path, "rb") as zip_file:
        file_obj = CallbackIOWrapper(progress_bar.update, zip_file, "read") if progress_bar is not None else zip_file
        yield from (
            (file_name.decode("utf-8"), file_size, unzipped_chunks)
            for file_name, file_size, unzipped_chunks in stream_unzip(_read_chunks(file_obj, chunk_size))
        )


def _read_chunks(file_obj: Union[BinaryIO, CallbackIOWrapper], chunk_size: int) -> Iterable[bytes]:
    """
    Reads a binary file chunk-wise.

    Args:
        file_obj: Open binary file object to read from.
        chunk_size: Maximum number of bytes to return per chunk.

    Yields:
        Consecutive non-empty byte chunks from the file object.
    """
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break
        yield chunk
