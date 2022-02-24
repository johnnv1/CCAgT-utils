from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from CCAgT_utils import Categories
from CCAgT_utils.converters.CCAgT import CCAgT_Annotations
from CCAgT_utils.types.annotation import bounds_to_BBox
from CCAgT_utils.types.annotation import count_BBox_categories
from CCAgT_utils.visualization import plot
from CCAgT_utils.visualization.colors import rgb_to_rgba


def image_with_boxes(CCAgT_ann: CCAgT_Annotations,
                     CCAgT_helper: Categories.Helper,
                     dir_path: str,
                     images_extension: str,
                     images_names: list = [],
                     shuffle_images: bool = True,
                     look_recursive: bool = True) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if look_recursive:
        print(f'Finding all files into the directory {dir_path}...')
        all_files = {file: os.path.join(path, file) for path, _, files in os.walk(dir_path) for file in files}
        print(f'Find a total of {len(all_files)} files.')

    if shuffle_images:
        random.shuffle(images_to_plot)
    get_color = {k: rgb_to_rgba(v, True) for k, v in CCAgT_helper.colors_by_category_id.items()}
    get_name = CCAgT_helper.name_by_category_id
    get_id = {v: k for (k, v) in get_name.items()}

    for img_name in tqdm(images_to_plot):

        if look_recursive:
            img_path = all_files[img_name + images_extension]
        else:
            img_path = os.path.join(dir_path, img_name + images_extension)
            if not os.path.exists(img_path):
                print(f'Not found the image {img_path}')
                continue

        image_boxes = CCAgT_ann.df[CCAgT_ann.df['image_name'] == img_name].apply(lambda r:
                                                                                 bounds_to_BBox(r['geometry'].bounds,
                                                                                                r['category_id']),
                                                                                 axis=1).to_numpy().tolist()

        counter = count_BBox_categories(image_boxes, get_name)
        text_counter = ' | '.join([f'{key}:: {value}' for key, value in counter.items()])
        selected_categories = [get_id[cat_name] for cat_name in counter]
        handles = plot.create_handles(get_color, get_name, selected_categories)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        img = Image.open(img_path)
        plot.image_with_boxes(img, image_boxes, ax, get_color, get_categories_name=get_name)
        ax.legend(handles=handles)
        ax.set_title(img_name)
        plt.figtext(0.5, 0.01, text_counter,
                    fontsize=10,
                    wrap=True,
                    horizontalalignment='center',
                    bbox={'facecolor': 'grey',
                          'alpha': 0.3, 'pad': 5})
        plt.show()
        del fig

    return 0
