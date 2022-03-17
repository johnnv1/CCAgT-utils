from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Color:
    Red: int
    Green: int
    Blue: int
    Alpha: int = 255
    size: int = 8

    def __post_init__(self) -> None:
        self._max_value = 2 ** self.size - 1

        _error_suf = f'value exceeds the max value for the color size! (bits size={self.size} -> max_value={self._max_value})'
        if self.Red > self._max_value:
            raise ValueError(f'The Red {_error_suf}')
        if self.Green > self._max_value:
            raise ValueError(f'The Green {_error_suf}')
        if self.Blue > self._max_value:
            raise ValueError(f'The Blue {_error_suf}')
        if self.Alpha > self._max_value:
            raise ValueError(f'The Alpha {_error_suf}')

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.Red, self.Green, self.Blue)

    @property
    def rgba(self) -> tuple[int, int, int, int]:
        return (self.Red, self.Green, self.Blue, self.Alpha)

    @property
    def rgb_normalized(self) -> tuple[float, float, float]:
        return (self.Red / self._max_value,
                self.Green / self._max_value,
                self.Blue / self._max_value)

    @property
    def rgba_normalized(self) -> tuple[float, float, float, float]:
        return (self.Red / self._max_value,
                self.Green / self._max_value,
                self.Blue / self._max_value,
                self.Alpha / self._max_value)


def from_tuple(t: tuple[int, int, int], alpha: int = 255, size: int = 8) -> Color:
    return Color(t[0], t[1], t[2], alpha, size)


def hex_to_rgb(hex: str) -> tuple[int, ...]:
    hex = hex.lstrip('#')
    hlen = len(hex)

    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


# Based on https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L43
def random_color_from_base(base: Color, max_dist: int = 30) -> Color:
    base_rgb = base.rgb
    new_color = base_rgb + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
    new_color = np.maximum(0, np.minimum(base._max_value, new_color))
    return from_tuple((int(new_color[0]), int(new_color[1]), int(new_color[2])), size=base.size)
