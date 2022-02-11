from __future__ import annotations

import pandas as pd

from CCAgT_utils.converters.LabelBox import LabelBox_Annotations


def test_labelbox_to_dataFrame():
    raw_data = [{'col A': 0, 'col B': 1}, {'col A': 2, 'col B': 3}]
    raw_df = pd.DataFrame(raw_data)
    lbb_ann = LabelBox_Annotations(raw_data)

    assert lbb_ann.raw_dataframe.equals(raw_df)
