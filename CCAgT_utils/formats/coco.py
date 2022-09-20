from __future__ import annotations

from CCAgT_utils.base.colors import Color


# def load():
# TODO:


# def validate():
# TODO:
# Based on https://github.com/cocodataset/panopticapi/blob/master/converters/detection2panoptic_coco_format.py
"""
    COCO_OD
        annotation{
            "id": int,
            "image_id": int,
            "category_id": int,
            "bbox": [x,y,width,height],
            "segments_info": [segment_info],
            "area": float,
            "iscrowd": 0 or 1,
        }
    COCO_PS
        annotation{
            "image_id": int,
            "file_name": str,
            "segments_info": [segment_info],
        }
        segment_info{
            "id": int,
            "category_id": int,
            "area": int,
            "bbox": [x,y,width,height],
            "iscrowd": 0 or 1,
        }
    """


# Copied from github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L73
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