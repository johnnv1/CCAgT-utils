from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image


def row_CCAgT(obj: Any, cat: int, name: str, **kwargs: Any) -> dict[str, Any]:
    return {
        'image_name': name,
        'geometry': obj,
        'category_id': cat,
        **kwargs,
    }


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


class ImageMaskFiles():
    def __init__(
        self,
        tmpdir: str,
        height: int = 1000,
        width: int = 1000,
        names: list[str] = ['example'],
        create_image: bool = True,
        create_mask: bool = True,
    ) -> None:
        self.tmpdir = tmpdir
        self.image_dir = os.path.join(tmpdir, 'images/')
        self.mask_dir = os.path.join(tmpdir, 'masks/')

        if create_mask:
            os.makedirs(self.mask_dir, exist_ok=True)
            self.__mask = mask_categorical((height, width))
            msk = Image.fromarray(self.__mask)

        if create_image:
            os.makedirs(self.image_dir, exist_ok=True)
            self.__image = mask_colorized((height, width))
            img = Image.fromarray(self.__image)

        for name in names:
            if create_mask:
                self.mask_path = os.path.join(self.mask_dir, f'{name}.png')
                msk.save(self.mask_path)

            if create_image:
                self.image_path = os.path.join(self.image_dir, f'{name}.jpg')
                img.save(self.image_path)

    def __enter__(self) -> tuple[str, str, str]:

        return (self.tmpdir, self.mask_dir, self.image_dir)

    def __exit__(self, *args: Any) -> None:
        pass
