from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Tuple
from typing import Union

import numpy as np
from PIL import Image

from CCAgT_utils.utils import find_files
from CCAgT_utils.utils import get_traceback

R = Union[float, Tuple[float, float, float]]


@dataclass
class Statistics:
    mean: R = 0.
    std: R = 0.
    max: R = 0.
    min: R = 0.
    count: int = 0

    def join_stats(self, results: Statistics) -> None:
        if self.count == 0:
            self.mean = results.mean
            self.std = results.std
            self.max = results.max
            self.min = results.min
        else:
            self.mean = np.mean([self.mean, results.mean], axis=0)
            self.std = np.mean([self.std, results.std], axis=0)
            self.max = np.max([self.max, results.max], axis=0)
            self.min = np.min([self.min, results.min], axis=0)
        self.count += results.count


def from_array(array: np.ndarray) -> Statistics:
    axis = (0, 1)
    _mean = np.mean(array, axis=axis)
    _std = np.std(array, axis=axis)
    _max = np.max(array, axis=axis)
    _min = np.min(array, axis=axis)

    return Statistics(_mean, _std, _max, _min, count=1)


@get_traceback
def single_core_from_image_files(filenames: list[str]) -> Statistics:
    if len(filenames) == 0:
        raise ValueError('It was expected a list of filenames with at least one value.')

    out_stats = Statistics()
    for filename in filenames:
        out_stats.join_stats(
            from_array(
                np.asarray(
                    Image.open(filename)
                )
            ))

    return out_stats


def from_image_files(images_dir: str,
                     extensions: str | tuple[str, ...] = '.jpg',
                     selection: set[str] = set()
                     ) -> Statistics:
    """From a directory path with images, will generate the stats of all
    images. The statistics generated are: mean, std, max, and min.

    Parameters
    ----------
    images_dir : str
        Path for the directories that contains the images of interest.

    extensions : str | tuple[str, ...], optional
        The extensions of the images files, by default '.jpg'

    selection : set[str], optional
        The images basenames (with extension) of selected to compute
        the statistics, by default set([]) (all images will be used)

    Returns
    -------
    dict[str, float | tuple[float, ...]]
        Will a dict where the key is the name of the statistics and the
        value is the computed statistic.
    """

    all_images = find_files(images_dir, extensions, True, selection)

    all_filenames = list(all_images.values())

    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)

    filenames_splitted = np.array_split(all_filenames, cpu_num)
    print(f'Start compute Statistics for {len(all_filenames)} ({extensions}) files using {cpu_num} cores with '
          f'{len(filenames_splitted[0])} files per core...')

    processes = []
    for filenames in filenames_splitted:
        if len(filenames) == 0:
            continue

        p = workers.apply_async(single_core_from_image_files, (filenames.tolist(), ))
        processes.append(p)

    out_stats = Statistics()
    for p in processes:
        out_stats.join_stats(p.get())

    print(f'Successfully computed the statstics of {out_stats.count} files with {len(processes)} processes!')
    return out_stats

# TODO: add describe for CCAgT Annotations
