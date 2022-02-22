from __future__ import annotations

from typing import Any


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
