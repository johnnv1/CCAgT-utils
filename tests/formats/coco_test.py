from __future__ import annotations

import pytest

from CCAgT_utils.base.colors import Color
from CCAgT_utils.formats.coco import build_description
from CCAgT_utils.formats.coco import color_to_id
from CCAgT_utils.formats.coco import save
from CCAgT_utils.formats.coco import validate


@pytest.fixture
def coco_instance_example():
    return {
        'images': [{'id': 0, 'file_name': 'a', 'width': 1, 'height': 1}],
        'annotations': [
            {
                'id': 0, 'image_id': 0, 'category_id': 0,
                'segmentation': [], 'bbox': [], 'iscrowd': 0,
                'area': 1,
            },
        ],
        'categories': [{'id': 0, 'name': '', 'supercategory': ''}],
    }


@pytest.fixture
def coco_segmentation_example():
    return {
        'images': [{'id': 0, 'file_name': 'a', 'width': 1, 'height': 1}],
        'annotations': [
            {
                'image_id': 0, 'file_name': '',
                'segments_info': [
                    {
                        'id': 0, 'category_id': 0, 'area': 1,
                        'bbox': [], 'iscrowd': 0,
                    },
                ],
            },
        ],
        'categories': [
            {
                'id': 0, 'name': '', 'supercategory': '',
                'isthing': 0, 'color': [0, 0, 0],
            },
        ],
    }


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


def test_validate(coco_instance_example):
    out = validate(coco_instance_example, 'instance')
    assert out


def test_validate_notimplemented(coco_segmentation_example):
    with pytest.raises(NotImplementedError):
        validate(coco_segmentation_example, 'panoptic')


def test_validate_wrong_data_type():
    with pytest.raises(TypeError) as er:
        validate([], '')
    assert str(er.value)[:30] == 'The coco data should be a dict'


def test_validate_without_required_fields():
    with pytest.raises(ValueError) as er:
        validate({}, '')
    assert str(er.value)[:68] == (
        'The dict should have at least the fields `images` and `annotations`.'
    )


def test_validate_without_valid_data_for_required_fields(
    coco_instance_example,
):
    with pytest.raises(TypeError) as er:
        validate({'images': {}, 'annotations': [{}]}, '')
    assert str(er.value) == 'The `image` field should be a list of dicts.'

    with pytest.raises(TypeError)as er:
        validate({'images': [{}], 'annotations': {}}, '')
    assert str(er.value) == (
        'The `annotations` field should be a list of dicts.'
    )

    images = coco_instance_example['images'][:]
    ann = coco_instance_example['annotations'][:]
    cats = coco_instance_example['categories'][:]
    with pytest.raises(ValueError) as er:
        validate(
            {'images': [{}], 'annotations': ann}, 'instance',
        )
    assert str(er.value)[:62] == (
        'The data at `images` field should have always have the info of'
    )

    with pytest.raises(ValueError) as er:
        validate(
            {'images': images, 'annotations': [{}]},
            'instance',
        )
    assert str(er.value)[:62] == (
        'The data should have the field `categories` for coco instances'
    )

    with pytest.raises(TypeError)as er:
        validate(
            {
                'images': images,
                'annotations': ann,
                'categories': [[]],
            },
            'instance',
        )
    assert str(er.value) == 'The `categories` field should be a list of dicts.'

    with pytest.raises(ValueError)as er:
        validate(
            {
                'images': images,
                'annotations': [{}],
                'categories': cats,
            },
            'instance',
        )
    assert str(er.value)[:59] == (
        'The data at `annotations` field should have always have the'
    )

    with pytest.raises(ValueError)as er:
        validate(
            {
                'images': images,
                'annotations': ann,
                'categories': [{}],
            },
            'instance',
        )
    assert str(er.value)[:58] == (
        'The data at `categories` field should have always have the'
    )
