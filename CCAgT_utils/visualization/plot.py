from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib import patches
from matplotlib import patheffects

from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.types.annotation import BBox
from CCAgT_utils.types.mask import Mask


def bbox(boxes: list[BBox],
         ax: plt.pyplot.Axes,
         categories_infos: CategoriesInfos,
         fontsize: int = 8,
         write_names: bool = True,
         without_text: set[int] = set({})
         ) -> plt.pyplot.Axes:

    for bb in boxes:
        ax.add_patch(patches.Rectangle((bb.x_init, bb.y_init),
                                       bb.width,
                                       bb.height,
                                       fill=False,
                                       edgecolor=categories_infos[bb.category_id].color.rgba_normalized,
                                       lw=1))
        if write_names and bb.category_id not in without_text:
            ax.text(bb.x_init,
                    bb.y_init,
                    categories_infos[bb.category_id].name,
                    verticalalignment='top',
                    color=categories_infos[bb.category_id].color.rgba_normalized,
                    clip_on=True,
                    fontsize=fontsize).set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'),
                                                        patheffects.Normal()])
    return ax


def image_with_boxes(image: np.ndarray | PIL.Image.Image,
                     boxes: list[BBox],
                     ax: plt.pyplot.Axes,
                     categories_infos: CategoriesInfos,
                     **kwargs: Any) -> plt.pyplot.Axes:
    ax.imshow(image)
    ax = bbox(boxes, ax, categories_infos, **kwargs)
    ax.set_axis_off()
    return ax


def mask_with_color(mask: Mask,
                    ax: plt.pyplot.Axes,
                    categories_infos: CategoriesInfos,
                    colorized: bool = False,
                    **kwargs: Any) -> plt.pyplot.Axes:
    if colorized:
        msk_rgb = mask.colorized(categories_infos)
        ax.imshow(msk_rgb, **kwargs)
    else:
        mask_categories = mask.unique_ids
        ax.imshow(mask.categorical,
                  cmap=mask.cmap(categories_infos),
                  vmax=max(mask_categories),
                  vmin=min(mask_categories),
                  interpolation='nearest',
                  **kwargs)

    ax.set_axis_off()
    return ax


def create_handles(categories_infos: CategoriesInfos,
                   selected_categories: list[int] = []) -> list[patches.Patch]:
    if len(selected_categories) > 0:
        categories_id = selected_categories
    else:
        categories_id = list(categories_infos.keys())

    return [patches.Patch(color=categories_infos.get_color(cat_id).rgba_normalized,
                          label=categories_infos.get_name(cat_id)) for cat_id in categories_id]
