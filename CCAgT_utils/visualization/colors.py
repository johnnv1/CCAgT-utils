from __future__ import annotations

import numpy as np


def rgb_to_rgba(color: list[int] | list[float],
                normalize: bool = False,
                bytes_precision: int = 8,
                alpha_value: int = 255) -> list[int] | list[float]:
    max_value = 2 ** bytes_precision - 1
    if len(color) == 4 and isinstance(color[0], float):
        return color
    elif len(color) == 3:
        if isinstance(color[0], float):
            return list(color) + [alpha_value / max_value]
        color = list(color) + [alpha_value]
    else:
        raise ValueError('Expected a RGB or a RGBA value!')

    if normalize:
        return [float(x / max_value) for x in color]

    return color


def hex_to_rgb(hex: str,
               bytes_precision: int = 16) -> list[int]:
    hex = hex.lstrip('#')
    hlen = len(hex)

    return [int(hex[i:i + hlen // 3], bytes_precision) for i in range(0, hlen, hlen // 3)]


# Based on https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L43
def random_color_from_base(base: list[int], max_dist: int = 30) -> list[int]:
    new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
    new_color = np.maximum(0, np.minimum(255, new_color))
    return list(int(i) for i in new_color)


def force_rgb(c: list[int] | str) -> list[int]:
    if isinstance(c, list):
        if len(c) == 3:
            return c
    elif isinstance(c, str):
        if c.startswith('#'):
            return hex_to_rgb(c)

    raise TypeError('Unexpected type of color, expected color into RGB list/tuple or HEX string!')
