from __future__ import annotations

from typing import Any


def row_CCAgT(obj: Any, cat: int, name: str) -> dict[str, Any]:
    return {'image_name': name,
            'geometry': obj,
            'category_id': cat}
