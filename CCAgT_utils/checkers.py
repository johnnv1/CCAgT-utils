from __future__ import annotations

import multiprocessing
from typing import Any

import numpy as np
from PIL import Image

from CCAgT_utils.utils import basename
from CCAgT_utils.utils import find_files
from CCAgT_utils.utils import get_traceback


def has_all_into_dir(dir_images: str, filenames: list[str], **kwargs: Any) -> bool:
    files = find_files(dir_images, **kwargs)

    return all(filename in files for filename in filenames)


@get_traceback
def single_core_mask_has(filenames: set[str],
                         categories: set[int]) -> set[str]:
    return {basename(filename) for filename in filenames
            if any(v in categories for v in np.unique(Image.open(filename).convert('L')))}


def masks_that_has(dir_masks: str,
                   categories: set[int],
                   extension: str | tuple[str, ...] = '.png',
                   look_recursive: bool = True) -> set[str]:

    files = find_files(dir_masks, extension, look_recursive)
    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)

    filenames_splitted = np.array_split(list(files), cpu_num)
    print(f'Start the checker if the masks have at least one of the categories ({categories}) using {cpu_num} cores with '
          f'{len(filenames_splitted[0])} masks per core...')
    processes = []
    for filenames in filenames_splitted:
        msk_filenames = {files[f] for f in filenames}

        p = workers.apply_async(single_core_mask_has, (msk_filenames,
                                                       categories))
        processes.append(p)

    masks_with_categories: set[str] = set()
    for p in processes:
        masks_with_categories = masks_with_categories.union(p.get())

    return masks_with_categories
