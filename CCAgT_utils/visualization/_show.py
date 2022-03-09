from __future__ import annotations

import os
import random
import sys
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from CCAgT_utils import categories
from CCAgT_utils.converters.CCAgT import CCAgT
from CCAgT_utils.types.annotation import bounds_to_BBox
from CCAgT_utils.types.annotation import count_BBox_categories
from CCAgT_utils.types.mask import Mask
from CCAgT_utils.utils import basename
from CCAgT_utils.visualization import plot
from CCAgT_utils.visualization.colors import rgb_to_rgba


def search_all_files(dir_path: str) -> dict[str, str]:
    # TODO: ADD A EXTENSION SELECTION
    print(f'Finding all files into the directory {dir_path}...')
    all_files = {file: os.path.join(path, file) for path, _, files in os.walk(dir_path) for file in files}
    print(f'Find a total of {len(all_files)} files.')
    return all_files


def image_with_boxes(CCAgT_ann: CCAgT,
                     CCAgT_helper: categories.Helper,
                     dir_path: str,
                     images_extension: str,
                     images_names: list[str] = [],
                     shuffle_images: bool = True,
                     look_recursive: bool = True) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if look_recursive:
        all_files = search_all_files(dir_path)

    if shuffle_images:
        random.shuffle(images_to_plot)

    get_color = {int(k): rgb_to_rgba(v, True) for k, v in CCAgT_helper.colors_by_category_id.items()}
    get_name = CCAgT_helper.name_by_category_id
    get_id = {v: k for (k, v) in get_name.items()}
    for img_name in tqdm(images_to_plot):
        if look_recursive:
            img_path = all_files[img_name + images_extension]
        else:
            img_path = os.path.join(dir_path, img_name + images_extension)
            if not os.path.exists(img_path):
                print(f'Not found the image {img_path}', file=sys.stderr)
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


def image_and_mask(CCAgT_ann: CCAgT,
                   CCAgT_helper: categories.Helper,
                   dir_path: str,
                   dir_mask_path: str,
                   images_extension: str,
                   masks_extension: str,
                   images_names: list[str] = [],
                   shuffle_images: bool = True,
                   look_recursive: bool = True) -> int:

    images_to_plot = CCAgT_ann.df['image_name'].unique()
    if len(images_names) > 0:
        images_to_plot = [i for i in images_to_plot if i in images_names]

    if look_recursive:
        all_files = search_all_files(dir_path)
        all_masks_files = search_all_files(dir_mask_path)

    if shuffle_images:
        random.shuffle(images_to_plot)

    get_color = {int(k): rgb_to_rgba(v, True) for k, v in CCAgT_helper.colors_by_category_id.items()}
    get_name = CCAgT_helper.name_by_category_id

    for img_name in tqdm(images_to_plot):

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
        plot.mask_with_color(msk, ax2, CCAgT_helper.colors_by_category_id, colorized=True)

        handles = plot.create_handles(get_color, get_name, msk.unique_ids)
        ax2.legend(handles=handles)
        ax2.set_title('Mask')

        ax3 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
        ax3.imshow(img)
        plot.mask_with_color(msk, ax3, CCAgT_helper.colors_by_category_id, colorized=True, alpha=0.4)
        ax3.set_title('Image with mask')

        fig.suptitle(img_name)
        plt.tight_layout()
        plt.show()

    return 0


class GetImagesAndMasks:
    paths: str
    selected_images: list[str]

    def __init__(self,
                 images: list[str],
                 selected_images: list[str],
                 dir_images: str,
                 dir_masks: str,
                 images_extension: str,
                 masks_extension: str,
                 shuffle_images: bool,
                 look_recursive: bool):

        if len(selected_images) > 0:
            images = [i for i in images if i in selected_images]

        if look_recursive:
            self.all_images_files = search_all_files(dir_images)
            self.all_masks_files = search_all_files(dir_masks)

        if shuffle_images:
            random.shuffle(images)

        self.images_names = images
        self.look_recursive = look_recursive
        self._images_extension = images_extension
        self._masks_extension = masks_extension
        self._dir_images = dir_images
        self._dir_masks = dir_masks

    def full_path(self, idx: int) -> tuple[str, str] | tuple[None, None]:
        bn = basename(self.images_names[idx])
        image_name = bn + self._images_extension
        mask_name = bn + self._masks_extension

        image_not_found = True
        mask_not_found = True
        if self.look_recursive:
            if image_name in self.all_images_files:
                image_path = self.all_images_files[image_name]
                image_not_found = False

            if mask_name in self.all_masks_files:
                mask_path = self.all_masks_files[mask_name]
                mask_not_found = False
        else:
            image_path = os.path.join(self._dir_images, image_name)
            mask_path = os.path.join(self._dir_masks, mask_name)
            image_not_found = not os.path.exists(image_path)
            mask_not_found = not os.path.exists(mask_path)

        if image_not_found:
            print(f'Not found the image! `{image_name}` at `{self._dir_images}`', file=sys.stderr)
            return (None, None)

        if mask_not_found:
            print(f'Not found the mask! `{mask_name}` at `{self._dir_masks}`', file=sys.stderr)
            return (None, None)

        return (image_path, mask_path)

    @staticmethod
    def load_image_and_mask(image_path: str, mask_path: str) -> Any:
        image = Image.open(image_path)
        mask = Mask(Image.open(mask_path).convert('L'))
        return image, mask

    def __iter__(self) -> GetImagesAndMasks:
        self._idx = 0
        self.current = self.full_path(self._idx)

        return self

    def __next__(self) -> Any:
        if self._idx < len(self.images_names):
            image_path = None
            mask_path = None
            while image_path is None and mask_path is None:
                self.current_image_name = self.images_names[self._idx]
                self.current_image_path = self.full_path(self._idx)
                self._idx += 1

                image_path, mask_path = self.current_image_path

            return self.load_image_and_mask(image_path, mask_path)
        else:
            raise StopIteration


def image_with_boxes_and_mask(CCAgT_ann: CCAgT,
                              CCAgT_helper: categories.Helper,
                              dir_path: str,
                              dir_mask_path: str,
                              images_extension: str,
                              masks_extension: str,
                              images_names: list[str] = [],
                              shuffle_images: bool = True,
                              look_recursive: bool = True) -> int:

    get_color = {int(k): rgb_to_rgba(v, True) for k, v in CCAgT_helper.colors_by_category_id.items()}
    get_color_rgb = CCAgT_helper.colors_by_category_id.copy()
    get_name = CCAgT_helper.name_by_category_id

    get_img_msk = GetImagesAndMasks(images=CCAgT_ann.df['image_name'].unique(),
                                    selected_images=images_names,
                                    dir_images=dir_path,
                                    dir_masks=dir_mask_path,
                                    images_extension=images_extension,
                                    masks_extension=masks_extension,
                                    shuffle_images=shuffle_images,
                                    look_recursive=look_recursive)

    iter_img_msk = iter(get_img_msk)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 18), sharex=True, sharey=True)
    image, mask = next(iter_img_msk)
    ax1_im = ax1.imshow(image)
    ax1.set_axis_off()

    image_boxes = CCAgT_ann.df[CCAgT_ann.df['image_name'] ==
                               get_img_msk.current_image_name].apply(lambda r:
                                                                     bounds_to_BBox(
                                                                         r['geometry'].bounds,
                                                                         r['category_id']),
                                                                     axis=1).to_numpy().tolist()

    counter = count_BBox_categories(image_boxes, get_name)
    text_counter = ' | '.join([f'{key}:: {value}' for key, value in counter.items()])
    handles = plot.create_handles(get_color, get_name, mask.unique_ids)
    ax2.legend(handles=handles)

    ax2 = plot.image_with_boxes(image, image_boxes, ax2, get_color, get_categories_name=get_name)
    ax2 = plot.mask_with_color(mask, ax2, get_color_rgb, colorized=True, alpha=0.4, vmin=1)

    text_ex = plt.figtext(0.5, 0.01, text_counter,
                          fontsize=10,
                          wrap=True,
                          horizontalalignment='center',
                          bbox={'facecolor': 'grey',
                                'alpha': 0.3, 'pad': 5})
    fig.suptitle(get_img_msk.current_image_name)
    plt.tight_layout()

    def on_key(event: Any) -> None:
        try:
            if event.key in ['a', 'A', 'left']:
                get_img_msk._idx -= 2

            if event.key in ['a', 'A', 'd', 'D', 'left', 'right']:
                image, mask = next(iter_img_msk)

                ax1_im.set_data(image)

                ax2.cla()

                image_boxes = CCAgT_ann.df[CCAgT_ann.df['image_name'] ==
                                           get_img_msk.current_image_name].apply(lambda r:
                                                                                 bounds_to_BBox(r['geometry'].bounds,
                                                                                                r['category_id']),
                                                                                 axis=1).to_numpy().tolist()

                plot.image_with_boxes(image, image_boxes, ax2, get_color, get_categories_name=get_name)
                plot.mask_with_color(mask, ax2, get_color_rgb, colorized=True, alpha=0.4, vmin=1)
                handles = plot.create_handles(get_color, get_name, mask.unique_ids)
                counter = count_BBox_categories(image_boxes, get_name)
                text_counter = ' | '.join([f'{key}:: {value}' for key, value in counter.items()])
                ax2.legend(handles=handles)
                text_ex.set_text(text_counter)
                fig.suptitle(get_img_msk.current_image_name)
            fig.canvas.draw()
        except StopIteration:
            plt.close()
            print('Already see all images!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return 0
