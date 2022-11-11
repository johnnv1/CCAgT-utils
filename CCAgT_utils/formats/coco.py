from __future__ import annotations

import json
from typing import Any
from typing import Sequence

from typing_extensions import TypeGuard

from CCAgT_utils.base.colors import Color
from CCAgT_utils.formats.ccagt import CCAgT

# def load():
# TODO:


def save(
        coco: dict[str, Any],
        filename: str,
) -> None:
    with open(filename, 'w') as outfile:
        json.dump(coco, outfile)


def _is_list_of_dict(
    data: Sequence[object] | dict[Any, Any],
) -> TypeGuard[list[dict[Any, Any]]]:
    return isinstance(data, list) and all(isinstance(i, dict) for i in data)


def validate(
    data: dict[str, Any],
    format: str,
) -> bool:
    if not isinstance(data, dict):
        raise TypeError(f'The coco data should be a dict, gotcha {type(data)}')

    if not ('images' in data and 'annotations' in data):
        raise ValueError(
            'The dict should have at least the fields `images` and '
            f'`annotations`. Gotcha {data.keys()}',
        )

    if not _is_list_of_dict(data['images']):
        raise TypeError('The `image` field should be a list of dicts.')

    if not _is_list_of_dict(data['annotations']):
        raise TypeError('The `annotations` field should be a list of dicts.')

    image_required_fields = {'id', 'file_name', 'width', 'height'}
    if not all(
        field in img_info
        for img_info in data['images']
        for field in image_required_fields
    ):
        raise ValueError(
            'The data at `images` field should have always have the '
            f'info of {image_required_fields}',
        )

    if format == 'instance':
        if not ('categories' in data):
            raise ValueError(
                'The data should have the field `categories` for coco '
                f'instances format. Gotcha {data.keys()}',
            )

        if not _is_list_of_dict(data['categories']):
            raise TypeError(
                'The `categories` field should be a list of dicts.',
            )

        annotations_required_fields = {
            'id', 'image_id', 'category_id', 'segmentation', 'bbox', 'iscrowd',
            'area',
        }
        if not all(
            field in annotation
            for annotation in data['annotations']
            for field in annotations_required_fields
        ):
            print(data['annotations'])
            raise ValueError(
                'The data at `annotations` field should have always have the '
                f'info of {annotations_required_fields}',
            )

        categories_required_fields = {'id', 'name', 'supercategory'}
        if not all(
            field in cat_info
            for cat_info in data['categories']
            for field in categories_required_fields
        ):
            raise ValueError(
                'The data at `categories` field should have always have the '
                f'info of {categories_required_fields}',
            )
    else:
        raise NotImplementedError

    return True


def build_description(template: str, ccagt_df: CCAgT) -> str:
    img_quantity = len(ccagt_df['image_id'].unique())
    slide_quantity = len(ccagt_df['slide_id'].unique())
    annotations_quantity = ccagt_df.shape[0]

    output = template.replace('{number_of_images}', str(img_quantity))
    output = output.replace('{number_of_slides}', str(slide_quantity))
    output = output.replace(
        '{number_of_annotations}',
        str(annotations_quantity),
    )

    return output


# noqa: E501 - Copied from github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L73
def color_to_id(color: Color) -> int:
    """Encode the color into a ID using:
    ID = R * 256 * G + 256 * 256 + B.


    Parameters
    ----------
    color : Color
        A RGB color

    Returns
    -------
    int
        An ID based on the color
    """
    return int(color.Red + 256 * color.Blue + 256 * 256 * color.Green)
