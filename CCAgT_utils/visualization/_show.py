from __future__ import annotations

import os
import random
import sys

import matplotlib.pyplot as plt
from PIL import Image

from CCAgT_utils import categories
from CCAgT_utils.converters.CCAgT import CCAgT
from CCAgT_utils.types.annotation import bounds_to_BBox
from CCAgT_utils.types.annotation import count_BBox_categories
from CCAgT_utils.types.mask import Mask
from CCAgT_utils.visualization import plot


def __search_all_files(dir_path: str) -> dict[str, str]:
    print(f'Finding all files into the directory {dir_path}...')
    all_files = {file: os.path.join(path, file) for path, _, files in os.walk(dir_path) for file in files}
    print(f'Find a total of {len(all_files)} files.')
    return all_files


def image_with_boxes(
    CCAgT_ann: CCAgT,
    categories_infos: categories.CategoriesInfos,
    dir_path: str,
    images_extension: str,
    images_names: list[str] = [],
    shuffle_images: bool = True,
    look_recursive: bool = True,
) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if look_recursive:
        all_files = __search_all_files(dir_path)

    if shuffle_images:
        random.shuffle(images_to_plot)

    get_id = {cat_info.name: cat_info.id for cat_info in categories_infos}

    for img_name in images_to_plot:
        if look_recursive:
            img_path = all_files[img_name + images_extension]
        else:
            img_path = os.path.join(dir_path, img_name + images_extension)
            if not os.path.exists(img_path):
                print(f'Not found the image {img_path}', file=sys.stderr)
                continue

        image_boxes = CCAgT_ann.df[CCAgT_ann.df['image_name'] == img_name].apply(
            lambda r:
            bounds_to_BBox(
                r['geometry'].bounds,
                r['category_id'],
            ),
            axis=1,
        ).to_numpy().tolist()

        counter = count_BBox_categories(image_boxes, categories_infos)
        text_counter = ' | '.join([f'{key}:: {value}' for key, value in counter.items()])
        selected_categories = {get_id[cat_name] for cat_name in counter}
        handles = plot.create_handles(categories_infos, selected_categories)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        img = Image.open(img_path)
        plot.image_with_boxes(img, image_boxes, ax, categories_infos, write_names=True)
        ax.legend(handles=handles)
        ax.set_title(img_name)
        plt.figtext(
            0.5, 0.01, text_counter,
            fontsize=10,
            wrap=True,
            horizontalalignment='center',
            bbox={
                'facecolor': 'grey',
                'alpha': 0.3, 'pad': 5,
            },
        )
        plt.show()
        del fig

    return 0


def image_and_mask(
    CCAgT_ann: CCAgT,
    categories_infos: categories.CategoriesInfos,
    dir_path: str,
    dir_mask_path: str,
    images_extension: str,
    masks_extension: str,
    images_names: list[str] = [],
    shuffle_images: bool = True,
    look_recursive: bool = True,
) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if look_recursive:
        all_files = __search_all_files(dir_path)
        all_masks_files = __search_all_files(dir_mask_path)

    if shuffle_images:
        random.shuffle(images_to_plot)

    for img_name in images_to_plot:

        if look_recursive:
            img_path = all_files[img_name + images_extension]
            msk_path = all_masks_files[img_name + masks_extension]
        else:
            img_path = os.path.join(dir_path, img_name + images_extension)
            msk_path = os.path.join(dir_mask_path, img_name + masks_extension)

            if not os.path.exists(img_path):
                print(f'Not found the image {img_path}', file=sys.stderr)
                continue

            if not os.path.exists(msk_path):
                print(f'Not found the mask {msk_path}', file=sys.stderr)
                continue

        img = Image.open(img_path)
        msk_p = Image.open(msk_path).convert('L')
        msk = Mask(msk_p)

        fig = plt.figure(figsize=(32, 18))

        ax1 = fig.add_subplot(2, 2, (1, 2))
        ax1.imshow(img)
        ax1.set_axis_off()

        ax2 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
        plot.mask_with_color(msk, ax2, categories_infos, colorized=True)

        handles = plot.create_handles(categories_infos, msk.unique_ids)
        ax2.legend(handles=handles)
        ax2.set_title('Mask')

        ax3 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
        ax3.imshow(img)
        plot.mask_with_color(msk, ax3, categories_infos, colorized=True, alpha=0.4)
        ax3.set_title('Image with mask')

        fig.suptitle(img_name)
        plt.tight_layout()
        plt.show()

    return 0


def image_with_boxes_and_mask(
    CCAgT_ann: CCAgT,
    categories_infos: categories.CategoriesInfos,
    dir_path: str,
    dir_mask_path: str,
    images_extension: str,
    masks_extension: str,
    images_names: list[str] = [],
    shuffle_images: bool = True,
    look_recursive: bool = True,
) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if look_recursive:
        all_files = __search_all_files(dir_path)
        all_masks_files = __search_all_files(dir_mask_path)

    if shuffle_images:
        random.shuffle(images_to_plot)

    for img_name in images_to_plot:

        if look_recursive:
            img_path = all_files[img_name + images_extension]
            msk_path = all_masks_files[img_name + masks_extension]
        else:
            img_path = os.path.join(dir_path, img_name + images_extension)
            msk_path = os.path.join(dir_mask_path, img_name + masks_extension)

            if not os.path.exists(img_path):
                print(f'Not found the image {img_path}', file=sys.stderr)
                continue

            if not os.path.exists(msk_path):
                print(f'Not found the mask {msk_path}', file=sys.stderr)
                continue

        img = Image.open(img_path)
        msk_p = Image.open(msk_path).convert('L')
        msk = Mask(msk_p)

        image_boxes = CCAgT_ann.df[CCAgT_ann.df['image_name'] == img_name].apply(
            lambda r:
            bounds_to_BBox(
                r['geometry'].bounds,
                r['category_id'],
            ),
            axis=1,
        ).to_numpy().tolist()

        counter = count_BBox_categories(image_boxes, categories_infos)
        text_counter = ' | '.join([f'{key}:: {value}' for key, value in counter.items()])
        handles = plot.create_handles(categories_infos, msk.unique_ids)

        fig = plt.figure(figsize=(32, 18))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.set_axis_off()

        ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        plot.image_with_boxes(img, image_boxes, ax2, categories_infos, write_names=True)
        plot.mask_with_color(msk, ax2, categories_infos, colorized=True, alpha=0.4, vmin=1)

        ax2.legend(handles=handles)

        plt.figtext(
            0.5, 0.01, text_counter,
            fontsize=10,
            wrap=True,
            horizontalalignment='center',
            bbox={
                'facecolor': 'grey',
                'alpha': 0.3, 'pad': 5,
            },
        )
        fig.suptitle(img_name)
        plt.tight_layout()
        plt.show()

    return 0
