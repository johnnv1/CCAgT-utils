from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib import patches
from matplotlib import patheffects

from CCAgT_utils.types.annotation import BBox


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
                     **kwargs) -> plt.pyplot.Axes:
    ax.imshow(image)
    ax = bbox(boxes, ax, get_color, **kwargs)
    ax.set_axis_off()
    return ax


def create_handles(get_color: dict[int, list[int] | list[float]],
                   get_categories_name: dict[int, str]) -> list[patches.Patch]:
    return [patches.Patch(color=c,
                          label=get_categories_name[cat_id]) for cat_id, c in get_color.items()]
