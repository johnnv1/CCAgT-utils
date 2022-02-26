from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib import patches
from matplotlib import patheffects

from CCAgT_utils.types.annotation import BBox
from CCAgT_utils.types.mask import Mask


def bbox(boxes: list[BBox],
         ax: plt.pyplot.Axes,
         get_color: dict[int, list[int] | list[float]],
         fontsize: int = 8,
         get_categories_name: dict[int, str] | None = None,
         without_text: set[int] = set({})
         ) -> plt.pyplot.Axes:

    for bb in boxes:
        ax.add_patch(patches.Rectangle((bb.x_init, bb.y_init),
                                       bb.width,
                                       bb.height,
                                       fill=False,
                                       edgecolor=get_color[bb.category_id],
                                       lw=1))
        if get_categories_name is not None and bb.category_id not in without_text:
            ax.text(bb.x_init,
                    bb.y_init,
                    get_categories_name[bb.category_id],
                    verticalalignment='top',
                    color=get_color[bb.category_id],
                    fontsize=fontsize).set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'),
                                                        patheffects.Normal()])
    return ax


def image_with_boxes(image: np.ndarray | PIL.Image.Image,
                     boxes: list[BBox],
                     ax: plt.pyplot.Axes,
                     get_color: dict[int, list[int] | list[float]],
                     **kwargs: Any) -> plt.pyplot.Axes:
    ax.imshow(image)
    ax = bbox(boxes, ax, get_color, **kwargs)
    ax.set_axis_off()
    return ax


def mask_with_color(mask: Mask,
                    ax: plt.pyplot.Axes,
                    get_color: dict[int, list[int] | list[float]],
                    colorized: bool = False,
                    **kwargs: Any) -> plt.pyplot.Axes:
    if colorized:
        msk_rgb = mask.colorized(get_color)
        ax.imshow(msk_rgb, **kwargs)
    else:
        mask_categories = mask.unique_ids
        ax.imshow(mask.categorical,
                  cmap=mask.cmap(get_color),
                  vmax=max(mask_categories),
                  vmin=min(mask_categories),
                  interpolation='nearest',
                  **kwargs)

    ax.set_axis_off()
    return ax


def create_handles(get_color: dict[int, list[int] | list[float]],
                   get_categories_name: dict[int, str],
                   selected_categories: list[int] = []) -> list[patches.Patch]:
    if len(selected_categories) > 0:
        categories_id = selected_categories
    else:
        categories_id = list(get_color.keys())

    return [patches.Patch(color=get_color[cat_id],
                          label=get_categories_name[cat_id]) for cat_id in categories_id]
