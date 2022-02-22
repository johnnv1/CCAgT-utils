from __future__ import annotations


def rgb_to_rgba(color: list[int] | list[float],
                normalize: bool = False,
                bytes_precision: int = 8) -> list[int] | list[float]:

    max_value = 2 ** bytes_precision - 1
    o = list(color) + [max_value]

    if normalize:
        return [float(x / max_value) for x in o]

    return o


def hex_to_rgb(hex: str,
               bytes_precision: int = 16) -> list[int]:
    hex = hex.lstrip('#')
    hlen = len(hex)

    return [int(hex[i:i + hlen // 3], bytes_precision) for i in range(0, hlen, hlen // 3)]
