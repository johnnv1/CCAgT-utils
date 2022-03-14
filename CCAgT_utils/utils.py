from __future__ import annotations

import functools
import os
import traceback
from enum import Enum
from typing import Callable
from typing import TypeVar

from CCAgT_utils.constants import FILENAME_SEP

R = TypeVar('R')


def basename(filename: str, with_extension: bool = False) -> str:
    """From a full filename get the basename with or not with the
    extension.

    Parameters
    ----------
    filename : str
        A full filename
    with_extension : bool, optional
        Flag to return the basename with extension, if True return
        the basename with the file extension, else will return just the
        basename, by default False

    Returns
    -------
    str
        The basename of the <filename> with or not the file extension
    """
    bn = os.path.basename(filename)
    if with_extension:
        return bn
    else:
        return os.path.splitext(bn)[0]


def get_traceback(f: Callable[..., R]) -> Callable[..., R]:
    """Decorator for print an error that occurs inside of some process

    Parameters
    ----------
    f : Callable
        The function that will be decorated, need to be a function called
        by a worker.

    Returns
    -------
    Callable
        The return of the function if all runs fine

    Raises
    ------
    e
        Will capture the exception from the process using the `traceback`
        print.
    """
    @functools.wraps(f)
    def wrapper(*args: object, **kwargs: object) -> R:
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper


class FILENAME_ITEM(Enum):
    slide = 0
    tile_id = 1,
    x_position_raw = 2,
    y_position_raw = 3


def items_from_filename(filename: str) -> list[str]:
    """From a full filename get the itens/infos at the basename

    Parameters
    ----------
    filename : str
        A full filename to an image or mask of CCAgT dataset

    Returns
    -------
    list
        A list with the 4 information that have at the basename
    """
    bn = basename(filename)
    items = bn.split(FILENAME_SEP)
    return items


def slide_from_filename(filename: str) -> str:
    """Based on a filename get the slide ID information

    Parameters
    ----------
    filename : str
        A full filename to an image or mask of CCAgT dataset

    Returns
    -------
    str
        The slide ID of the filename
    """
    return items_from_filename(filename)[FILENAME_ITEM.slide.value]


def find_files(dir_path: str,
               extension: str | tuple[str, ...],
               look_recursive: bool = False,
               selection: set[str] = set()) -> dict[str, str]:
    """Find all files into at the path and subdirectories

    Parameters
    ----------
    dir_path : str
        Path of the base directory to look
    extension : str | tuple[str]
        Extension of the dessired files

    Returns
    -------
    dict[str, str]
        A dict with the filename as key and the relative path for the
        file
    """
    if look_recursive:
        files = {file: os.path.join(path, file) for path, _, files in os.walk(dir_path) for file in files
                 if file.endswith(extension) and (len(selection) == 0 or file in selection)}
    else:
        files = {file: os.path.join(dir_path, file) for file in os.listdir(dir_path)
                 if file.endswith(extension) and (len(selection) == 0 or file in selection)}

    return files


def create_structure(dir_path: str, slides: set[str]) -> None:

    dir_images = os.path.join(dir_path, 'images/')
    dir_masks = os.path.join(dir_path, 'masks/')

    for slide in slides:
        os.makedirs(os.path.join(dir_images, slide), exist_ok=True)
        os.makedirs(os.path.join(dir_masks, slide), exist_ok=True)
