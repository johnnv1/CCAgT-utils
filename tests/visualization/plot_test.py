from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison

from CCAgT_utils.visualization import plot


@pytest.mark.slow
@image_comparison(baseline_images=['boxes_plot'], extensions=['png'])
def test_bbox(boxes, categories_infos, shape):
    plt.close('all')

    _, ax = plt.subplots()
    ax.set_xlim([0, shape[1]])
    ax.set_ylim([shape[0], 0])
    ax.set_axis_off()
    plot.bbox(
        boxes, ax, categories_infos, write_names=True, without_text=set({3}),
    )


@pytest.mark.slow
@image_comparison(
    baseline_images=['image_with_boxes_plot'],
    extensions=['png'],
)
def test_image_with_boxes(rgb_image, boxes, categories_infos):
    plt.close('all')
    _, ax = plt.subplots()
    plot.image_with_boxes(
        rgb_image, boxes, ax, categories_infos, write_names=False,
    )


@pytest.mark.slow
@image_comparison(baseline_images=['mask_with_color_plot'], extensions=['png'])
def test_mask_with_color_cmap(mask, categories_infos):
    plt.close('all')
    _, ax = plt.subplots(figsize=(16, 9), dpi=300)
    plot.mask_with_color(mask, ax, categories_infos)


@pytest.mark.slow
@image_comparison(baseline_images=['mask_with_color_plot'], extensions=['png'])
def test_mask_with_color_colorized(mask, categories_infos):
    plt.close('all')
    _, ax = plt.subplots(figsize=(16, 9), dpi=300)
    plot.mask_with_color(mask, ax, categories_infos, colorized=True)


def test_create_handles(categories_infos):
    handles1 = plot.create_handles(categories_infos)

    assert len(handles1) == len(categories_infos)
    assert handles1[0].get_label() == categories_infos[0].name
    expected = categories_infos[0].color.rgba_normalized
    assert tuple(handles1[0].get_edgecolor()) == expected

    selected_categories = [2, 3]
    handles2 = plot.create_handles(categories_infos, selected_categories)

    assert len(handles2) == len(selected_categories)
    assert handles2[0].get_label() == categories_infos[2].name
    expected = categories_infos[3].color.rgba_normalized
    assert tuple(handles2[1].get_edgecolor()) == expected
