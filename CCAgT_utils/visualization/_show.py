from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
from PIL import Image

from CCAgT_utils import Categories
from CCAgT_utils.converters.CCAgT import CCAgT_Annotations
from CCAgT_utils.types.annotation import bounds_to_BBox
from CCAgT_utils.visualization import plot
from CCAgT_utils.visualization.colors import rgb_to_rgba


def image_with_boxes(CCAgT_ann: CCAgT_Annotations,
                     CCAgT_helper: Categories.Helper,
                     dir_path: str,
                     images_extension: str,
                     images_names: list = [],
                     shuffle_images: bool = True) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if shuffle_images:
        random.shuffle(images_to_plot)
    get_color = {k: rgb_to_rgba(v, True) for k, v in CCAgT_helper.colors_by_category_id.items()}
    get_name = CCAgT_helper.name_by_category_id
    handles = plot.create_handles(get_color, get_name)

    for img_name in images_to_plot:
        img_path = os.path.join(dir_path, img_name + images_extension)
        if not os.path.exists(img_path):
            print(f'Not found the image {img_path}')
            continue

        image_boxes = CCAgT_ann.df[CCAgT_ann.df['image_name'] == img_name].apply(lambda r:
                                                                                 bounds_to_BBox(r['geometry'].bounds,
                                                                                                r['category_id']),
                                                                                 axis=1).to_numpy().tolist()

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        img = Image.open(img_path)
        plot.image_with_boxes(img, image_boxes, ax, get_color, get_categories_name=get_name)
        ax.legend(handles=handles)
        plt.show()
        del fig

    return 0
