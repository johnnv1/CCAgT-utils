from __future__ import annotations

import json
from enum import Enum
from typing import Any

from CCAgT_utils.errors import FileTypeError
from CCAgT_utils.visualization import colors


class Categories(Enum):
    BACKGROUND = 0
    NUCLEUS = 1
    CLUSTER = 2
    SATELLITE = 3
    NUCLEUS_OUT_OF_FOCUS = 4
    OVERLAPPED_NUCLEI = 5
    NON_VIABLE_NUCLEUS = 6
    LEUKOCYTE_NUCLEUS = 7


class Helper():

    def __init__(self,
                 raw_helper: list[dict[str, Any]]) -> None:

        if not isinstance(raw_helper, list):
            raise ValueError('Expected a list of dictionary that represents raw helper data!')

        self.raw_helper = raw_helper[:]

        self.__check_id_names()

        if all((x['id'] != 0 and x['name'].lower() != 'background') for x in self.raw_helper):
            self.raw_helper.append({
                'id': 0,
                'color': [0, 0, 0],
                'name': 'background',
                'minimal_area': 0
            })

    def __check_id_names(self) -> None:
        for id, name in self.name_by_category_id.items():
            if Categories(id).name != name.upper():
                raise ValueError(f'The category name to id does not match with the expected! For id {id} it was expected '
                                 f'{Categories(id).name} and receive {name.upper()}')

    @property
    def min_area_by_category_id(self) -> dict[int, int]:
        return {int(x['id']): int(x['minimal_area']) for x in self.raw_helper}

    @property
    def name_by_category_id(self) -> dict[int, str]:
        return {int(x['id']): str(x['name']) for x in self.raw_helper}

    @property
    def colors_by_category_id(self) -> dict[int, list[int] | list[float]]:
        def force_rgb(c: list[int] | list[float] | str) -> list[int] | list[float]:
            if isinstance(c, list):
                if len(c) in {3, 4}:
                    return c
            elif isinstance(c, str):
                if c.startswith('#'):
                    return colors.hex_to_rgb(c)

            raise TypeError('Unexpected type of color, expected color into RGB list/tuple or HEX string!')

        return {int(x['id']): force_rgb(x['color']) for x in self.raw_helper}


def read_json(filename: str, **kwargs: Any) -> Helper:
    if not filename.endswith('.json'):
        raise FileTypeError('The auxiliary file is not a JSON file.')

    with open(filename, **kwargs) as f:
        dataset_helper = json.load(f)

    categories_helpper = dataset_helper['categories']

    return Helper(categories_helpper)
