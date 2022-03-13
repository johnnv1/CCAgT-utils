from __future__ import annotations

from typing import Tuple
from typing import Union

import numpy as np
from PIL import Image

from CCAgT_utils.utils import find_files

R = Union[float, Tuple[float, ...]]


def array(array: np.ndarray) -> tuple[R, R, R, R]:
    axis = (0, 1)
    _mean = np.mean(array, axis=axis)
    _std = np.std(array, axis=axis)
    _max = np.max(array, axis=axis)
    _min = np.min(array, axis=axis)

    return (_mean, _std, _max, _min)


def image_files(images_dir: str,
                extensions: str | tuple[str, ...] = '.jpg',
                selection: set[str] = set()
                ) -> dict[str, R]:
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

    filenames = list(all_images.values())

    out = {}
    _mean, _std, _max, _min = array(np.asarray(Image.open(filenames[0])))
    out['mean'] = _mean
    out['std'] = _std
    out['max'] = _max
    out['min'] = _min

    for filename in filenames[1:]:
        _mean, _std, _max, _min = array(np.asarray(Image.open(filename)))

        out['mean'] = np.mean([out['mean'], _mean], axis=0)
        out['std'] = np.mean([out['std'], _std], axis=0)
        out['max'] = np.max([out['max'], _max], axis=0)
        out['min'] = np.min([out['min'], _min], axis=0)

    return out
