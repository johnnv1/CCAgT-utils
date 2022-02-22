from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import patheffects

from CCAgT_utils.types.annotation import BBox


def bbox(boxes: list[BBox],
         ax: plt.pyplot.Axes,
         get_color: dict[int, tuple[int]],
         get_categories_name: dict[int, str],
         fontsize: int = 8) -> plt.pyplot.Axes:

    for bb in boxes:
        ax.add_patch(patches.Rectangle((bb.x_init, bb.y_init),
                                       bb.width,
                                       bb.height,
                                       fill=False,
                                       edgecolor=get_color[bb.category_id],
                                       lw=1))

        ax.text(bb.x_init,
                bb.y_init,
                get_categories_name[bb.category_id],
                verticalalignment='top',
                color='red',
                fontsize=fontsize).set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'),
                                                     patheffects.Normal()])
    return ax
