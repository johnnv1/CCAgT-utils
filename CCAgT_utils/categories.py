from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from CCAgT_utils.errors import FileTypeError
from CCAgT_utils.types.colors import Color
from CCAgT_utils.types.colors import random_color_from_base


class Categories(Enum):
    BACKGROUND = 0
    NUCLEUS = 1
    CLUSTER = 2
    SATELLITE = 3
    NUCLEUS_OUT_OF_FOCUS = 4
    OVERLAPPED_NUCLEI = 5
    NON_VIABLE_NUCLEUS = 6
    LEUKOCYTE_NUCLEUS = 7


CATS_COLORS = {
    Categories.BACKGROUND: Color(0, 0, 0),
    Categories.NUCLEUS: Color(21, 62, 125),
    Categories.CLUSTER: Color(114, 67, 144),
    Categories.SATELLITE: Color(254, 166, 0),
    Categories.NUCLEUS_OUT_OF_FOCUS: Color(26, 167, 238),
    Categories.OVERLAPPED_NUCLEI: Color(39, 91, 82),
    Categories.NON_VIABLE_NUCLEUS: Color(5, 207, 192),
    Categories.LEUKOCYTE_NUCLEUS: Color(255, 0, 0)
}

CATS_MIN_AREA = {
    Categories.BACKGROUND: 0,
    Categories.NUCLEUS: 500,
    Categories.CLUSTER: 40,
    Categories.SATELLITE: 30,
    Categories.NUCLEUS_OUT_OF_FOCUS: 500,
    Categories.OVERLAPPED_NUCLEI: 500,
    Categories.NON_VIABLE_NUCLEUS: 200,
    Categories.LEUKOCYTE_NUCLEUS: 200
}


@dataclass
class CategoryInfo:
    """
    id: unique id for each category
    name: name of the category
    color: the RGB values for the representations of each category
    labelbox_schemaId: The schemaID from labelbox of each category
    minimal_area: The minimal area that the category can have
    supercategory: The name of the supercategory, if the category
        belongs to a supercategory (ex.: Animal is a supercategory
        for dog)
    isthing: Defines if the categories is a stuff or a thing. A thing
        is a countable object, a category that has instance-level
        annotation. The stuff is amorphous region of similar texture,
        its a category without instance-level annotation. Specified as
        0 for stuff and 1 for things.
    """
    id: int
    name: str
    color: Color
    labelbox_schemaId: str | None = None
    minimal_area: int = 0
    supercategory: str | None = None
    isthing: int = 1


class CategoriesInfos():
    def __init__(self,
                 categories_info: list[dict[str, Any]] | None = None) -> None:
        if isinstance(categories_info, list):
            _categories_info = []
            for d in categories_info:
                if 'color' in d:
                    c = d['color']
                    d['color'] = Color(c[0], c[1], c[2])
                _categories_info.append(d)
            categories_info = _categories_info[:]

            if all((x['id'] != 0 and x['name'].lower() != 'background') for x in categories_info):
                categories_info.append({'id': 0,
                                        'color': Color(0, 0, 0),
                                        'name': 'background',
                                        'minimal_area': 0})

        elif categories_info is None:
            categories_info = []
            for cat in Categories:
                isthing = 1
                if cat == Categories.BACKGROUND:
                    isthing = 0
                categories_info.append({'color': CATS_COLORS[cat],
                                        'name': cat.name,
                                        'id': cat.value,
                                        'minimal_area': CATS_MIN_AREA[cat],
                                        'isthing': isthing})
        else:
            raise ValueError('Wrong type of categories info!')

        self._infos = [CategoryInfo(**itens) for itens in categories_info]
        self.taken_colors = {cat_info.color.rgb for cat_info in self._infos if cat_info.isthing == 0}
        self.taken_colors.add((0, 0, 0))
        self.__check_id_names()

    def __check_id_names(self) -> None:
        for id, name in self.name_by_category_id.items():
            if Categories(id).name != name.upper():
                raise ValueError(f'The category name to id does not match with the expected! For id {id} it was expected '
                                 f'{Categories(id).name} and receive {name.upper()}')

    @property
    def min_area_by_category_id(self) -> dict[int, int]:
        return {x.id: x.minimal_area for x in self._infos}

    @property
    def name_by_category_id(self) -> dict[int, str]:
        return {x.id: x.name for x in self._infos}

    @property
    def colors_by_category_id(self) -> dict[int, Color]:
        return {x.id: x.color for x in self._infos}

    # Based on https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L42
    def generate_random_color(self, category_id: int) -> Color:
        cat_info = self.cat_info_from_id(category_id)
        base_color = cat_info.color

        if cat_info.isthing == 0:
            return base_color
        elif base_color.rgb not in self.taken_colors:
            self.taken_colors.add(base_color.rgb)
            return base_color
        else:
            while True:
                color = random_color_from_base(base_color)
                if color.rgb not in self.taken_colors:
                    self.taken_colors.add(color.rgb)
                    return color

    def cat_info_from_id(self, category_id: int) -> CategoryInfo:
        for cat_info in self._infos:
            if cat_info.id == category_id:
                return cat_info
        else:
            raise KeyError('Category ID not found!')

    def get_cat_info(self, category: Categories) -> CategoryInfo:
        for cat_info in self._infos:
            if cat_info.id == category.value:
                return cat_info
        else:
            raise KeyError('Category ID not found!')

    def __iter__(self) -> CategoriesInfos:
        self._idx = 0
        return self

    def __next__(self) -> CategoryInfo:
        if self._idx < len(self._infos):
            out = self._infos[self._idx]
            self._idx += 1
            return out
        else:
            raise StopIteration


def read_json(filename: str, **kwargs: Any) -> CategoriesInfos:
    if not filename.endswith('.json'):
        raise FileTypeError('The auxiliary file is not a JSON file.')

    with open(filename, **kwargs) as f:
        dataset_helper = json.load(f)

    categories_helpper = dataset_helper['categories']

    return CategoriesInfos(categories_helpper)
