from __future__ import annotations

from typing import Any

from CCAgT_utils.visualization import colors


class Helper():

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

    @property
    def colors_by_category_id(self) -> dict[int, list[int] | list[float]]:
        def force_rgb(c: list | str) -> list[int] | list[float]:
            if isinstance(c, list):
                if len(c) == 3:
                    return c
            elif isinstance(c, str):
                return colors.hex_to_rgb(c)

            raise TypeError('Unexpected type of color, expected color into RGB list/tuple or HEX string!')

        return {int(x['id']): force_rgb(x['color']) for x in self.raw_helper}
