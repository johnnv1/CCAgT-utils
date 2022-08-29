from __future__ import annotations

import pytest

from CCAgT_utils.formats.labelbox import LabelBox
from CCAgT_utils.formats.labelbox import load
from CCAgT_utils.formats.labelbox import validate


def test_load(lbox_sample_complete, lbox_raw_sample_complete):
    raw_df = load(lbox_sample_complete)

    assert raw_df._constructor == LabelBox
    assert LabelBox(lbox_raw_sample_complete).equals(raw_df)


def test_validate():
    data = [{'ID': ..., 'External ID': ..., 'Reviews': ..., 'Label': ...}]
    assert validate(data)

    data.append({'Skipped': ...})
    assert validate(data)

    with pytest.raises(ValueError):
        validate({'ID': ...})

    data.append({'ID': ..., 'Reviews': ..., 'Label': ...})
    with pytest.raises(KeyError):
        validate(data)
