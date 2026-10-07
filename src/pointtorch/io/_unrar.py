"""Utilities for extracting files from RAR archives."""

__all__ = ["unrar"]

import pathlib
from shutil import copyfileobj
from typing import IO, List, Optional, Union

import rarfile
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper


def unrar(  # pylint: disable=too-many-locals
    rar_path: Union[str, pathlib.Path],
    dest_path: Union[str, pathlib.Path],
    items: Optional[List[str]] = None,
    progress_bar: bool = True,
    progress_bar_desc: Optional[str] = None,
) -> None:
    """
    Extract files from a RAR archive.

    Since RAR uses a proprietary compression algorithm, compressed archive members can only be extracted if one of
    the tools :code:`unrar`, :code:`unar`, or :code:`7z` is installed and available on the system :code:`PATH`. Note
    that the :code:`tar.exe` built into Windows is not a usable substitute for these tools.

    Args:
        rar_path: Path of the RAR archive.
        dest_path: Path of the directory in which to save the extracted files.
        items: Names of the items to extract. Defaults to :code:`None`, which means that all items are extracted.
        progress_bar: Whether a progress bar should be created to show the extraction progress. Defaults to
            :code:`True`.
        progress_bar_desc: Description of the progress bar. Only used if :code:`progress_bar` is :code:`True`. Defaults
            to :code:`None`.

    Raises:
        FileNotFoundError: If the RAR file does not exist.
        KeyError: If :code:`items` contains items not existing in the RAR archive.
        RuntimeError: If the file is not a valid RAR archive or if the archive cannot be extracted, e.g., because none
            of the supported extraction tools is available on the system :code:`PATH`.
    """
    if isinstance(dest_path, str):
        dest_path = pathlib.Path(dest_path)

    if isinstance(rar_path, str):
        rar_path = pathlib.Path(rar_path)

    try:
        with rarfile.RarFile(rar_path) as rar_file:
            infolist = rar_file.infolist()

            if items is not None:
                valid_items = [item.filename for item in infolist]
                invalid_items = [item for item in items if item not in valid_items]
                if len(invalid_items) > 0:
                    raise KeyError(f"The following items are not contained in the RAR archive: {invalid_items}.")

            total_size = sum(item.file_size for item in infolist if items is None or item.filename in items)
            prog_bar = (
                tqdm(desc=progress_bar_desc, unit="B", unit_scale=True, unit_divisor=1000, total=total_size)
                if progress_bar
                else None
            )

            for item in infolist:
                if items is not None and item.filename not in items:
                    continue

                file_path = dest_path / item.filename
                if not file_path.resolve().is_relative_to(dest_path.resolve()):
                    raise RuntimeError(
                        f"The RAR archive contains an item outside the target directory: {item.filename}."
                    )

                if item.is_dir():
                    file_path.mkdir(exist_ok=True, parents=True)
                    continue

                file_path.parent.mkdir(exist_ok=True, parents=True)
                with rar_file.open(item) as in_file, open(file_path, "wb") as out_file:
                    file_reader: Union[CallbackIOWrapper, IO[bytes]]
                    if prog_bar is not None:
                        file_reader = CallbackIOWrapper(prog_bar.update, in_file)
                    else:
                        file_reader = in_file
                    copyfileobj(file_reader, out_file)
    except rarfile.NotRarFile as error:
        raise RuntimeError(f"{rar_path} is not a valid RAR archive.") from error
    except rarfile.Error as error:
        raise RuntimeError(
            f"Could not extract {rar_path}. This requires one of the tools 'unrar', 'unar', or '7z' to be installed "
            "and available on the system PATH (on Windows, the tar.exe built into the OS is not a usable substitute "
            "here). Please install one of these tools."
        ) from error
