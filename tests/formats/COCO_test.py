from __future__ import annotations

from CCAgT_utils.base.colors import Color
from CCAgT_utils.formats.coco import build_description
from CCAgT_utils.formats.coco import color_to_id
from CCAgT_utils.formats.coco import save


def test_color_to_id():
    c = Color(Red=10, Green=20, Blue=15)
    expected = 1314570
    out = color_to_id(c)
    assert out == expected


def test_save(tmpdir):
    outfilename = tmpdir.join('example.json')
    save({'name': 'test'}, str(outfilename))

    assert outfilename.isfile()


def test_build_description(ccagt_df_single_nucleus_complete):
    df = ccagt_df_single_nucleus_complete
    df['slide_id'] = 1
    template = 'a={number_of_images} | b={number_of_annotations}'
    out = build_description(template, df)
    assert out == 'a=1 | b=1'
