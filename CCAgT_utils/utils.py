from __future__ import annotations

import functools
import os
import traceback
from typing import Any
from typing import Callable
from typing import TypeVar

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


class Categories_Helper():

    def __init__(self,
                 raw_helper: list[dict[str, Any]]) -> None:

        if not isinstance(raw_helper, list):
            raise ValueError('Expected a list of dictionary that represents raw helper data!')

        self.raw_helper = raw_helper

    @property
    def min_area_by_category_id(self) -> dict[int, int]:
        return {int(x['id']): int(x['minimal_area']) for x in self.raw_helper}

    @property
    def name_by_category_id(self) -> dict[int, str]:
        return {int(x['id']): str(x['name']) for x in self.raw_helper}
