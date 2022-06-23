from __future__ import annotations

import collections

import numpy as np
from PIL import Image
from PIL import ImageDraw

from CCAgT_utils.base.categories import Categories
from CCAgT_utils.formats.annotation import Annotation
from CCAgT_utils.formats.mask import Mask

# Order of categories IDs available at the metadata
DRAW_ORDER = (
    Categories.LEUKOCYTE_NUCLEUS,
    Categories.NON_VIABLE_NUCLEUS,
    Categories.OVERLAPPED_NUCLEI,
    Categories.NUCLEUS_OUT_OF_FOCUS,
    Categories.NUCLEUS,
    Categories.CLUSTER,
    Categories.SATELLITE,
)


def order_annotations_to_draw(annotations: list[Annotation]) -> list[Annotation]:
    items = collections.defaultdict(list)
    for ann in annotations:
        items[ann.category_id].append(ann)
    out = [ann for cat in DRAW_ORDER for ann in items[cat.value]]

    return out


def annotations_to_mask(
    annotations: list[Annotation],
    width: int = 1600,
    height: int = 1200,
) -> Mask:
    shape = (height, width)

    annotations_sorted = order_annotations_to_draw(annotations)

    out = Image.fromarray(np.zeros(shape, dtype=np.uint8))
    for ann in annotations_sorted:
        out = draw_annotation(out, ann, ann.category_id)

    return Mask(np.array(out))


def draw_annotation(
    target: Image.Image,
    annotation: Annotation,
    value: int | tuple[int, int, int],
) -> Image.Image:
    for geo in annotation:
        pol_x, pol_y = geo.exterior.coords.xy
        coords = list(zip(pol_x, pol_y))
        ImageDraw.Draw(target).polygon(coords, fill=value)

    return target
