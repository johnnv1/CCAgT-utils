from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.converters.CCAgT import CCAgT
from CCAgT_utils.describe import annotations_per_image


def tvt(ids: list[int],
        tvt_size: tuple[float, float, float],
        seed: int = 1609
        ) -> tuple[list[int], list[int], list[int]]:
    """From a list of indexes/ids (int) will generate the
    train-validation-test data.


    Based on `github.com/scikit-learn/scikit-learn/blob/
    37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/
    model_selection/_split.py#L2321`

    Parameters
    ----------
    ids : list[int]
        a list of indexes/ids
    tvt_size : tuple[float, float, float]
        The size of each fold (train, validation, test)
        In general the train size, will be ignored because,
        `train size = n samples - validation size - test size`

    seed : int, optional
        The seed for the random state, by default 1609

    Returns
    -------
    dict[str, list[int]]
        A dict, where the key is tha name of the respectively fold and
        the value is the list of indexes/ids selected.
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
        # Have some case that trigger this?
        if len(out[k]) != qtd[k]:
            print(f'At {k} have { len(out[k])} and as expected {qtd[k]}')

    return out['train'], out['valid'], out['test']


def tvt_by_nors(ccagt: CCAgT,
                categories_infos: CategoriesInfos,
                tvt_size: tuple[float, float, float] = (.7, .15, .15),
                **kwargs: Any
                ) -> tuple[list[int], list[int], list[int]]:
    """This will split the CCAgT annotations based on the number of NORs
    into each image. With a silly separation, first will split
    between each fold images with one or less NORs, after will split
    images with the amount of NORs is between 2 and 7, and at least will
    split images that have more than 7 NORs.

    Parameters
    ----------
    ccagt : CCAgT
        The annotations of the dataset
    categories_infos : CategoriesInfos
        The auxiliary information's for each category at the dataset
    tvt_size : tuple[float, float, float], optional
        The desired size of each fold (train, validation, test),
        by default (.7, .15, .15)

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        A tuple with values for each fold and the value is the list of
        indexes/ids selected. At the sequence of train, validation, test.

    Raises
    ------
    ValueError
        If the sum of tvt_size be different of 1 will raise
    """
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
