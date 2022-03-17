from __future__ import annotations

import json
from dataclasses import dataclass
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


CATS_COLORS = {
    Categories.BACKGROUND: [0, 0, 0],
    Categories.NUCLEUS: [21, 62, 125],
    Categories.CLUSTER: [114, 67, 144],
    Categories.SATELLITE: [254, 166, 0],
    Categories.NUCLEUS_OUT_OF_FOCUS: [26, 167, 238],
    Categories.OVERLAPPED_NUCLEI: [39, 91, 82],
    Categories.NON_VIABLE_NUCLEUS: [5, 207, 192],
    Categories.LEUKOCYTE_NUCLEUS: [255, 0, 0]
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
    color: list[int] | str
    labelbox_schemaId: str | None = None
    minimal_area: int = 0
    supercategory: str | None = None
    isthing: int = 1


class CategoriesInfos():
    def __init__(self,
                 categories_info: list[dict[str, Any]] | None = None) -> None:
        if isinstance(categories_info, list):
            self.__check_id_names()

            if all((x['id'] != 0 and x['name'].lower() != 'background') for x in categories_info):
                categories_info.append({'id': 0,
                                        'color': [0, 0, 0],
                                        'name': 'background',
                                        'minimal_area': 0})

        else:
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

        self._infos = [CategoryInfo(**itens) for itens in categories_info]
        self.taken_colors = {tuple(cat_info.color) for cat_info in self._infos if cat_info.isthing == 0}
        self.taken_colors.add((0, 0, 0))

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
    def colors_by_category_id(self) -> dict[int, list[int] | list[float]]:
        return {x.id: colors.force_rgb(x.color) for x in self._infos}

    # Based on https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L42
    def generate_random_color(self, category_id: int) -> list[int]:
        cat_info = self.cat_info_from_id(category_id)
        base_color = colors.force_rgb(cat_info.color)

        if cat_info.isthing == 0:
            return base_color
        elif tuple(base_color) not in self.taken_colors:
            self.taken_colors.add(tuple(base_color))
            return base_color
        else:
            while True:
                color = colors.random_color_from_base(base_color)
                if tuple(color) not in self.taken_colors:
                    self.taken_colors.add(tuple(color))
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
