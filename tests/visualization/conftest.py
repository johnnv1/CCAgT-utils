from __future__ import annotations

import pytest

from CCAgT_utils.types.annotation import BBox


@pytest.fixture
def boxes():
    return [BBox(10, 200, 50, 125, 1),
            BBox(20, 290, 10, 10, 2),
            BBox(50, 250, 8, 9, 3)]
