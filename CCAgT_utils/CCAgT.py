from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import TypeVar

from CCAgT_utils.utils import basename

FILENAME_SEP = '_'

LIST_OF_DICTS = TypeVar('LIST_OF_DICTS', List[Dict[str, Any]], None)


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
