from __future__ import annotations

import multiprocessing
import os
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image

from CCAgT_utils.categories import Categories
from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.constants import STRUCTURE
from CCAgT_utils.converters.CCAgT import CCAgT
from CCAgT_utils.converters.CCAgT import read_parquet
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

    def to_dict(self) -> dict[str, R | int]:
        return asdict(self)

    def __str__(self) -> str:
        _mean = f'Mean: {self.mean:.2f}'
        _std = f'std: {self.std:.2f}'
        _max = f'Max: {self.max:.2f}'
        _min = f'Min: {self.min:.2f}'
        _count = f'Quantity: {self.count}'

        return f'{_count} | {_mean} | {_std}  | {_max} | {_min}'


def from_list(itens: list[int | float]) -> Statistics:
    _mean = np.mean(itens)
    _std = np.std(itens)
    _max = np.max(itens)
    _min = np.min(itens)
    return Statistics(_mean, _std, _max, _min, count=len(itens))


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
                    Image.open(filename),
                ),
            ),
        )

    return out_stats


def from_image_files(
    images_dir: str,
    extensions: str | tuple[str, ...] = '.jpg',
    selection: set[str] = set(),
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
    print(
        f'Start compute Statistics for {len(all_filenames)} ({extensions}) files using {cpu_num} cores with '
        f'{len(filenames_splitted[0])} files per core...',
    )

    processes = []
    for filenames in filenames_splitted:
        if len(filenames) == 0:
            continue  # pragma: no cover

        p = workers.apply_async(single_core_from_image_files, (filenames.tolist(),))
        processes.append(p)

    out_stats = Statistics()
    for p in processes:
        out_stats.join_stats(p.get())

    print(f'Successfully computed the statstics of {out_stats.count} files with {len(processes)} processes!')
    return out_stats


def annotations_per_image(
    ccagt: CCAgT,
    categories_infos: CategoriesInfos,
) -> pd.DataFrame:
    df = ccagt.df
    df_describe_images = df.groupby(['image_id', 'category_id']).size().reset_index().rename(columns={0: 'count'})
    df_describe_images = df_describe_images.pivot(columns=['category_id'], index='image_id')
    df_describe_images = df_describe_images.rename({c.id: c.name.upper() for c in categories_infos}, axis=1)
    df_describe_images['qtd_annotations'] = df_describe_images.sum(axis=1)
    df_describe_images = df_describe_images.fillna(0)
    df_describe_images['NORs'] = df_describe_images[
        'count',
        Categories.CLUSTER.name,
    ] + df_describe_images[
        'count',
        Categories.SATELLITE.name,
    ]

    return df_describe_images


def ccagt_annotations(
    ccagt: CCAgT,
    categories_infos: CategoriesInfos,
) -> dict[str, Any]:
    df = ccagt.df
    ann_count = {cat.name: df.loc[df['category_id'] == cat.id, 'area'].shape[0] for cat in categories_infos}
    qtd_ann = df.shape[0]
    ann_dist = {cat_name: qtd_cat / qtd_ann for cat_name, qtd_cat in ann_count.items()}
    area_stats = {
        cat.name: from_list(df.loc[df['category_id'] == cat.id, 'area'].tolist())
        for cat in categories_infos if ann_count[cat.name] > 0
    }

    qtd_images = df['image_id'].nunique()
    qtd_slides = df['slide_id'].nunique()
    return {
        'qtd_images': qtd_images,
        'qtd_slide': qtd_slides,
        'qtd_annotations': qtd_ann,
        'qtd_annotations_categorical': ann_count,
        'dist_annotations': ann_dist,
        'area_stats': area_stats,
    }


def tvt_annotations_as_df(
    train: dict[str, Any],
    valid: dict[str, Any],
    test: dict[str, Any],
) -> tuple[pd.DataFrame, ...]:
    out = {}
    out['train'] = train
    out['validation'] = valid
    out['test'] = test
    folds = out.keys()

    df_qtd = pd.DataFrame({
        'fold': folds,
        'images': [out[f]['qtd_images'] for f in folds],
        'slides': [out[f]['qtd_slide'] for f in folds],
        'annotations': [out[f]['qtd_annotations'] for f in folds],
    })
    df_qtd_categorical = pd.DataFrame([
        {
            'fold': f,
            **{
                k: v for k, v in out[f]['qtd_annotations_categorical'].items()
                if k != Categories.BACKGROUND.name
            },
        }
        for f in folds
    ])
    df_qtd = pd.merge(df_qtd, df_qtd_categorical, on='fold')
    df_qtd.loc['total'] = df_qtd.sum()
    df_qtd.loc[df_qtd.index == 'total', 'fold'] = 'total'

    total_images = df_qtd.loc[df_qtd['fold'] == 'total', 'images'].tolist()[0]
    total_ann = df_qtd.loc[df_qtd['fold'] == 'total', 'annotations'].tolist()[0]
    df_dist = pd.DataFrame({
        'fold': folds,
        '% images': [out[f]['qtd_images'] / total_images for f in folds],
        '% annotations': [out[f]['qtd_annotations'] / total_ann for f in folds],
    })
    df_dist_categorical = pd.DataFrame([
        {
            'fold': f,
            **{
                f'% {k}': v / out[f]['qtd_annotations']
                for k, v in out[f]['qtd_annotations_categorical'].items()
                if k != Categories.BACKGROUND.name
            },
        }
        for f in folds
    ])
    df_dist = pd.merge(df_dist, df_dist_categorical, on='fold')

    df_area = pd.DataFrame()
    for f in folds:
        _df = pd.DataFrame([{'category': k, **v.to_dict()} for k, v in out[f]['area_stats'].items()])
        _df = _df.set_index('category').transpose()
        _df['fold'] = f
        df_area = pd.concat([df_area, _df])

    return df_qtd, df_dist, df_area


def dataset(
    ccagt_path: str,
    categories_infos: CategoriesInfos,
    dataset_dir: str,
    extensions: tuple[str, ...] = ('.jpg', '.png'),
) -> None:
    ccagt = read_parquet(ccagt_path)

    name = os.path.basename(os.path.normpath(dataset_dir))
    images_dir = os.path.join(dataset_dir, STRUCTURE['i'])
    masks_dir = os.path.join(dataset_dir, STRUCTURE['m'])

    print(f'Dataset name: `{name}` | Location: `{dataset_dir}`')
    print(f'From the annotations file ({ccagt_path}) -')
    if ccagt.df.shape[0] == 0:
        print('Do not have any annotation!')
    else:
        desc = ccagt_annotations(ccagt, categories_infos)
        print(f'Quantity of images: {desc["qtd_images"]}')
        print(f'Quantity of slides: {desc["qtd_slide"]}')
        print(f'Quantity of annotations: {desc["qtd_annotations"]}')
        for cat_name, qtd in desc['qtd_annotations_categorical'].items():
            dist = desc['dist_annotations'][cat_name]
            print(f' > Quantity of annotations for {cat_name}: {qtd} - {dist*100:.2f}%')
        print('Statistics of the area of each category...')
        for cat_name, area_stats in desc['area_stats'].items():
            print(f' > Statistics of area for {cat_name}: {area_stats}')

        images_quantity = len(find_files(images_dir, extensions, True))
        masks_quantity = len(find_files(masks_dir, extensions, True))
        print('On disk data -')
        print(f'Total of images: {images_quantity} - at `{images_dir}`')
        print(f'Total of masks: {masks_quantity} - at `{masks_dir}`')


def categorical_mask(mask: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(mask, return_counts=True)
    return dict(zip(unique, counts))


@get_traceback
def single_core_from_mask_files(
    filenames: list[str],
) -> dict[int, int]:
    if len(filenames) == 0:
        raise ValueError('It was expected a list of filenames with at least one value.')

    out = {cat.value: 0 for cat in Categories}
    for filename in filenames:
        counts = categorical_mask(
            np.asarray(
                Image.open(filename).convert('L'),
            ),
        )
        out = {k: v + counts[k] if k in counts else v for k, v in out.items()}

    return out


def from_mask_files(
    masks_dir: str,
    extensions: str | tuple[str, ...] = '.png',
    selection: set[str] = set(),
) -> dict[str, int]:
    all_masks = find_files(masks_dir, extensions, True, selection)

    all_filenames = list(all_masks.values())

    cpu_num = multiprocessing.cpu_count()
    workers = multiprocessing.Pool(processes=cpu_num)

    filenames_splitted = np.array_split(all_filenames, cpu_num)
    print(
        f'Start count pixels quantity for {len(all_filenames)} ({extensions}) files using {cpu_num} cores with '
        f'{len(filenames_splitted[0])} files per core...',
    )

    processes = []
    for filenames in filenames_splitted:
        if len(filenames) == 0:
            continue  # pragma: no cover

        p = workers.apply_async(single_core_from_mask_files, (filenames.tolist(),))
        processes.append(p)

    out = {cat.value: 0 for cat in Categories}
    for p in processes:
        counts = p.get()
        out = {k: v + counts[k] if k in counts else v for k, v in out.items()}

    n_files = len(all_masks)
    print(f'Successfully computed pixels quantity of each category from {n_files} files with {len(processes)} processes!')

    out_by_names = {str(Categories(k).name): int(v) for k, v in out.items()}
    return out_by_names
