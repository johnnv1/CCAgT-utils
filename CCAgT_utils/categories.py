from __future__ import annotations

import json
from typing import Any

from CCAgT_utils.errors import FileTypeError
from CCAgT_utils.visualization import colors


class Helper():

    def __init__(self,
                 raw_helper: list[dict[str, Any]]) -> None:

        if not isinstance(raw_helper, list):
            raise ValueError('Expected a list of dictionary that represents raw helper data!')

        self.raw_helper = raw_helper[:]
        if all((x['id'] != 0 and x['name'].lower() != 'background') for x in self.raw_helper):
            self.raw_helper.append({
                'id': 0,
                'color': [0, 0, 0],
                'name': 'background',
                'minimal_area': 0
            })

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
