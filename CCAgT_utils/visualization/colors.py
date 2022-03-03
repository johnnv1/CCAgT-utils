from __future__ import annotations


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
