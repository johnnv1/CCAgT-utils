from __future__ import annotations

import os

from CCAgT_utils.checkers import masks_that_has
from CCAgT_utils.utils import basename
from CCAgT_utils.utils import find_files


def clean_images_and_masks(dir_images: str,
                           dir_masks: str,
                           categories: set[int],
                           extension: str | tuple[str, ...] = ('.png', '.jpg'),
                           look_recursive: bool = True) -> None:

    basenames_matches = masks_that_has(dir_masks, categories, extension, look_recursive)

    image_filenames_to_remove = {v for k, v in find_files(dir_images, extension, look_recursive).items()
                                 if basename(k) not in basenames_matches}

    mask_filenames_to_remove = {v for k, v in find_files(dir_masks, extension, look_recursive).items()
                                if basename(k) not in basenames_matches}

    print('Deleting images files...')
    for filename in image_filenames_to_remove:
        os.remove(filename)

    print('Deleting masks files...')
    for filename in mask_filenames_to_remove:
        os.remove(filename)


"""
TODO: Entry point

CCAgT-utils create-subdataset --name <name> (--slice-images [(horizontal), (vertical)] | --extract <category id>)--original
<path to original dataset with images/ and masks/ dirs> --remove-images-without Sequence[categories ids] --output-path <path
to local to create the new subdataset> --check-if-all-have-at-least-one-of Sequence[categories ids]

D2
CCAgT-utils create-subdataset --name Dataset2 --slice-images [4, 4] --original <2.1.1/original/> --remove-images-without
[1,2,3,4,5,6,7] --output-path <2.1.1/>

D3
CCAgT-utils create-subdataset --name Dataset3 --original <2.1.1/original/> --remove-images-without [1] --output-path <2.1.1/>
--check-if-all-have-at-least-one-of {2, 3}

D4
CCAgT-utils create-subdataset --name Dataset4 --slice-images [4, 4] --original <2.1.1/original/> --remove-images-without [1]
--output-path <2.1.1/> --check-if-all-have-at-least-one-of {2, 3}

D5
CCAgT-utils create-subdataset --name Dataset5 --extract 1 --check-if-all-have-at-least-one-of {2, 3}
"""
