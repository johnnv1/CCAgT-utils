from __future__ import annotations

import pandas as pd
import pytest

from CCAgT_utils import split
from CCAgT_utils.formats import CCAgT


def test_tvt():
    items = list(range(10))

    out = split.tvt(items, (0.6, 0.2, 0.2), seed=6547)

    assert out[0] == {5, 0, 9, 2, 1, 7}
    assert out[1] == {4, 3}
    assert out[2] == {6, 8}


def test_tvt_by_nors(ccagt_df_multi, categories_infos):

    df = ccagt_df_multi.copy()
    df_dummy = ccagt_df_multi.copy()
    for n in range(8):
        df_dummy['image_name'] = f'C_xxx{n}'
        df = pd.concat([df, df_dummy])

    df['image_id'] = CCAgT.generate_ids(df['image_name'])
    df['slide_id'] = CCAgT.slides_ids(df)
    out = split.tvt_by_nors(df, categories_infos, (0.7, 0.15, 0.15), seed=6547)

    assert out[0] == {1, 2, 4, 7, 9, 11}
    assert out[1] == {5, 6}
    assert out[2] == {8, 10, 3}


def test_tvt_by_nors_wrong_size(ccagt_df_multi, categories_infos):
    with pytest.raises(ValueError):
        split.tvt_by_nors(ccagt_df_multi, categories_infos, (1, 0.2, 0.2))
