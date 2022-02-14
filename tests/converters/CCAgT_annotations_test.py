from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Point

from CCAgT_utils.converters.CCAgT import CCAgT_Annotations


def test_init_class():
    df = pd.DataFrame([{'image_name': 'A', 'geometry': Point(1, 1), 'category_id': 1}])
    ccagt_ann = CCAgT_Annotations(df)

    assert ccagt_ann.df.equals(df)


def test_init_wrong_columns():
    with pytest.raises(KeyError):
        df = pd.DataFrame([{'image_name': 'A', 'category_id': 1}])
        CCAgT_Annotations(df)

    with pytest.raises(KeyError):
        df = pd.DataFrame([{'category_id': 1}])
        CCAgT_Annotations(df)

    with pytest.raises(KeyError):
        df = pd.DataFrame([{'image_name': 'A', 'geometry': Point(1, 1)}])
        CCAgT_Annotations(df)


def test_init_wrong_data():
    with pytest.raises(TypeError):
        CCAgT_Annotations([{'image_name': 'A', 'geometry': Point(1, 1)}])
