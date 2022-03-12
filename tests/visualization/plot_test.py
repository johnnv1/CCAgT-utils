from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison

from CCAgT_utils.visualization import plot


@pytest.mark.slow
@image_comparison(baseline_images=['boxes_plot'], extensions=['png'])
def test_bbox(boxes, get_color_rgba_norm, get_categories_name, shape):
    plt.close('all')

    _, ax = plt.subplots()
    ax.set_xlim([0, shape[1]])
    ax.set_ylim([shape[0], 0])
    ax.set_axis_off()
    plot.bbox(boxes, ax, get_color_rgba_norm,
              get_categories_name=get_categories_name, without_text=set({3}))


@pytest.mark.slow
@image_comparison(baseline_images=['image_with_boxes_plot'], extensions=['png'])
def test_image_with_boxes(rgb_image, boxes, get_color_rgba_norm):
    plt.close('all')
    _, ax = plt.subplots()
    plot.image_with_boxes(rgb_image, boxes, ax, get_color_rgba_norm)


@pytest.mark.slow
@image_comparison(baseline_images=['mask_with_color_plot'], extensions=['png'])
def test_mask_with_color_cmap(mask, get_color_rgba_norm):
    plt.close('all')
    _, ax = plt.subplots(figsize=(16, 9), dpi=300)
    plot.mask_with_color(mask, ax, get_color_rgba_norm)


@pytest.mark.slow
@image_comparison(baseline_images=['mask_with_color_plot'], extensions=['png'])
def test_mask_with_color_colorized(mask, get_color_rgb):
    plt.close('all')
    _, ax = plt.subplots(figsize=(16, 9), dpi=300)
    plot.mask_with_color(mask, ax, get_color_rgb, colorized=True)


def test_create_handles(get_color_rgba_norm, get_categories_name):
    handles1 = plot.create_handles(get_color_rgba_norm, get_categories_name)

    assert len(handles1) == len(get_color_rgba_norm)
    assert handles1[0].get_label() == get_categories_name[1]
    assert list(handles1[0].get_edgecolor()) == get_color_rgba_norm[1]

    selected_categories = [2, 3]
    handles2 = plot.create_handles(get_color_rgba_norm, get_categories_name, selected_categories)

    assert len(handles2) == len(selected_categories)
    assert handles2[0].get_label() == get_categories_name[2]
    assert list(handles2[1].get_edgecolor()) == get_color_rgba_norm[3]
