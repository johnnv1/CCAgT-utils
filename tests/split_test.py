from __future__ import annotations

import pandas as pd
import pytest

from CCAgT_utils import split


def test_tvt():
    items = list(range(10))

    out = split.tvt(items, (0.6, 0.2, 0.2), seed=6547)

    assert out[0] == {5, 0, 9, 2, 1, 7}
    assert out[1] == {4, 3}
    assert out[2] == {6, 8}


def test_tvt_by_nors(ccagt_ann_multi, ccagt_df_multi, categories_infos):

    for n in range(8):
        ccagt_df_multi['image_name'] = f'C_xxx{n}'
        ccagt_ann_multi.df = pd.concat([ccagt_ann_multi.df, ccagt_df_multi])

    ccagt_ann_multi.df['image_id'] = ccagt_ann_multi.generate_ids(ccagt_ann_multi.df['image_name'])
    ccagt_ann_multi.df['slide_id'] = ccagt_ann_multi.get_slide_id()
    out = split.tvt_by_nors(ccagt_ann_multi, categories_infos, (0.6, 0.2, 0.2), seed=6547)

    assert out[0] == {8, 2, 5, 6}
    assert out[1] == {1, 4}
    assert out[2] == {3, 7}


def test_tvt_by_nors_wrong_size(ccagt_ann_multi, categories_infos):
    with pytest.raises(ValueError):
        split.tvt_by_nors(ccagt_ann_multi, categories_infos, (1, 0.2, 0.2))
