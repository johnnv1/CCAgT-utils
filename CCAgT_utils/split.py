from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.converters.CCAgT import CCAgT
from CCAgT_utils.describe import annotations_per_image


def tvt(ids: list[int], tvt_size: tuple[float, float, float], seed: int = 1609) -> tuple[list[int], list[int], list[int]]:
    """TODO

    Based on `github.com/scikit-learn/scikit-learn/blob/
    37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/
    model_selection/_split.py#L2321`

    Parameters
    ----------
    ids : list[int]
        _description_
    tvt_size : tuple[float, float, float]
        _description_
    seed : int, optional
        _description_, by default 1609

    Returns
    -------
    dict[str, list[int]]
        _description_
    """
    n_samples = len(ids)
    qtd = {'valid': ceil(n_samples * tvt_size[1]),
           'test': ceil(n_samples * tvt_size[2])}
    qtd['train'] = int(n_samples - qtd['valid'] - qtd['test'])
    rng = np.random.RandomState(seed)
    permutatation = rng.permutation(ids)

    out = {'train': list(permutatation[:qtd['train']]),
           'valid': list(permutatation[qtd['train']:qtd['train'] + qtd['valid']]),
           'test': list(permutatation[qtd['train'] + qtd['valid']:])}

    for k in ['train', 'valid', 'test']:
        if len(out[k]) != qtd[k]:
            print(f'At {k} have { len(out[k])} and as expected {qtd[k]}')

    return out['train'], out['valid'], out['test']


def tvt_by_nors(ccagt: CCAgT,
                categories_infos: CategoriesInfos,
                tvt_size: tuple[float, float, float] = (.7, .15, .15),
                **kwargs: Any) -> tuple[list[int], list[int], list[int]]:
    if sum(tvt_size) != 1:
        raise ValueError('The sum of `tvt_size` need to be equals 1!')

    df_describe_imgs = annotations_per_image(ccagt, categories_infos)

    img_ids = {}
    img_ids['low_nors'] = df_describe_imgs.loc[(df_describe_imgs['NORs'] < 2)].index
    img_ids['medium_nors'] = df_describe_imgs[(df_describe_imgs['NORs'] >= 2) * (df_describe_imgs['NORs'] <= 7)].index
    img_ids['high_nors'] = df_describe_imgs[(df_describe_imgs['NORs'] > 7)].index

    train_ids = []
    valid_ids = []
    test_ids = []

    for k, ids in img_ids.items():
        print(f'Splitting {len(ids)} images with {k} quantity...')
        if len(ids) == 0:
            continue
        _train, _valid, _test = tvt(ids, tvt_size, **kwargs)
        print(f'>T: {len(_train)} V: {len(_valid)} T: {len(_test)}')
        train_ids.extend(_train)
        valid_ids.extend(_valid)
        test_ids.extend(_test)

    _used = train_ids + valid_ids + test_ids
    if len(_used) != df_describe_imgs.shape[0]:
        # Have some case that trigger this?
        ids_not_used = df_describe_imgs[~df_describe_imgs.index.isin(_used)].index.to_list()
        test_ids.extend(ids_not_used)

    return train_ids, valid_ids, test_ids
