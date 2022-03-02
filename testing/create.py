from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
from PIL import Image


def row_CCAgT(obj: Any, cat: int, name: str) -> dict[str, Any]:
    return {'image_name': name,
            'geometry': obj,
            'category_id': cat}


def mask_categorical(shape: tuple[int, int]) -> np.ndarray:
    h = int(shape[0])
    w = int(shape[1])
    out = np.zeros(shape, dtype=np.uint8)
    out[:h // 2, :w // 2] = 1
    out[:h // 2, w // 2:] = 2
    out[h // 2:, :w // 2] = 3
    out[h // 2:, w // 2:] = 4
    return np.array(out, dtype=np.uint8)


def mask_colorized(shape: tuple[int, int]) -> np.ndarray:
    h = int(shape[0])
    w = int(shape[1])
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:h // 2, :w // 2] = [21, 62, 125]
    out[:h // 2, w // 2:] = [114, 67, 144]
    out[h // 2:, :w // 2] = [255, 166, 0]
    out[h // 2:, w // 2:] = [26, 167, 238]
    return np.array(out, dtype=np.uint8)


class RawAuxFiles():
    def __init__(self, raw_data: list[dict[str, Any]], aux_data: list[dict[str, Any]]) -> None:
        self.raw_data = raw_data
        self.aux_data = aux_data
        self.tmp_dir = tempfile.TemporaryDirectory('_RawAuxFiles', 'CCAgTutils_')
        self.raw_path = os.path.join(self.tmp_dir.name, 'raw_file.json')
        self.aux_path = os.path.join(self.tmp_dir.name, 'aux_file.json')

    def __enter__(self) -> tuple[str, str, str]:
        with open(self.raw_path, 'w') as f:
            json.dump(self.raw_data, f)

        with open(self.aux_path, 'w') as f:
            json.dump(self.aux_data, f)

        return (self.tmp_dir.name, self.raw_path, self.aux_path)

    def __exit__(self, *args: Any) -> None:
        self.tmp_dir.cleanup()


class ImageMaskFiles():
    def __init__(self, height: int = 1000, width: int = 1000, names: list[str] = ['example']) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory('_ImageMaskFiles', 'CCAgTutils_')
        self.mask_dir = os.path.join(self.tmp_dir.name, 'masks/')
        self.image_dir = os.path.join(self.tmp_dir.name, 'images/')
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        self.__mask = mask_categorical((height, width))
        self.__image = mask_colorized((height, width))
        img = Image.fromarray(self.__image)
        msk = Image.fromarray(self.__mask)
        for name in names:
            self.mask_path = os.path.join(self.mask_dir, f'{name}.png')
            self.image_path = os.path.join(self.image_dir, f'{name}.jpg')
            img.save(self.image_path)
            msk.save(self.mask_path)

    def __enter__(self) -> tuple[str, str, str]:

        return (self.tmp_dir.name, self.mask_dir, self.image_dir)

    def __exit__(self, *args: Any) -> None:
        self.tmp_dir.cleanup()
