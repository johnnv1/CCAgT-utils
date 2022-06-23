from __future__ import annotations

import numpy as np

from CCAgT_utils.converters import masks
from CCAgT_utils.formats.annotation import Annotation


def test_order_annotations_to_draw(cluster_ex, satellite_ex, nucleus_ex):
    anns = [Annotation(cluster_ex, 2), Annotation(satellite_ex, 3), Annotation(nucleus_ex, 1)]
    anns = masks.order_annotations_to_draw(anns)

    categories_ids = [ann.category_id for ann in anns]
    assert categories_ids == [1, 2, 3]


def test_annotations_to_mask(cluster_ex, cluster_mask_ex):
    shape = cluster_mask_ex.shape
    msk = masks.annotations_to_mask([Annotation(cluster_ex, 2)], shape[0], shape[1])
    assert np.array_equal(cluster_mask_ex, msk.categorical)
