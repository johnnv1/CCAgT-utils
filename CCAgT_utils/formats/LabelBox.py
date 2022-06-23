from __future__ import annotations

from typing import Any

import pandas as pd

from CCAgT_utils.base.utils import open_and_read_json


def load(
    filename: str,
) -> pd.DataFrame:
    raw = open_and_read_json(filename)

    validate(raw)

    return pd.DataFrame(raw)


def validate(raw: list[dict[str, Any]]) -> bool:
    if not isinstance(raw, list):
        raise ValueError('Expected a list of dictionary that represents raw LabelBox data!')

    expected_data = set({'ID', 'External ID', 'Reviews', 'Label'})
    for it in raw:
        if not all(i in it for i in expected_data):
            if 'Skipped' not in it:
                raise KeyError(f'Not found expected values need to have `Skipped` or {expected_data}')

    return True
