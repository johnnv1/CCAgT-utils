from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from CCAgT_utils import describe
from CCAgT_utils.base.categories import Categories
from CCAgT_utils.formats import ccagt
from testing import create


def test_statistics():
    a = describe.Statistics()
    b = describe.Statistics(1., 1., 1., -1., 1)

    a.join_stats(b)

    assert a.count == 1
    assert a.mean == 1.
    a.join_stats(b)

    assert a.count == 2
    assert a.mean == 1.

    assert a.to_dict() == {
        'mean': 1.0,
        'std': 1.0,
        'max': 1.0,
        'min': -1.0,
        'count': 2,
    }

    assert str(a) == 'Quantity: 2 | Mean: 1.00 | std: 1.00  | Max: 1.00 | Min: -1.00'


def test_describe_from_list():
    data = [10, 10, 10, 20, 20, 20, 1, 2, 3]
    stats = describe.from_list(data)

    assert stats.mean == pytest.approx(10.66, 0.01)
    assert stats.std == pytest.approx(7.378, 0.01)
    assert stats.max == 20
    assert stats.min == 1
    assert stats.count == 9


def test_describe_from_array():
    arr = np.array([[10, 10, 10], [20, 20, 20], [1, 2, 3]])
    stats = describe.from_array(arr)

    assert stats.mean == pytest.approx(10.66, 0.01)
    assert stats.std == pytest.approx(7.378, 0.01)
    assert stats.max == 20
    assert stats.min == 1
    assert stats.count == 1


def test_single_core_from_image_files_empty():
    with pytest.raises(ValueError):
        describe.single_core_from_image_files([])


def test_single_core_from_image_files(shape, tmpdir):
    image = create.mask_colorized(shape)
    filename = os.path.join(tmpdir, 'test.png')
    Image.fromarray(image).save(filename)

    stats = describe.single_core_from_image_files([filename])
    assert np.array_equal(stats.mean, np.mean(image, axis=(0, 1)))
    assert np.array_equal(stats.std, np.std(image, axis=(0, 1)))
    assert np.array_equal(stats.max, np.max(image, axis=(0, 1)))
    assert np.array_equal(stats.min, np.min(image, axis=(0, 1)))
    assert stats.count == 1


@pytest.mark.slow
def test_from_image_files(shape, tmpdir):
    image = create.mask_colorized(shape)
    for i in range(2):
        Image.fromarray(image).save(os.path.join(tmpdir, f'test{i}.png'))

    stats = describe.from_image_files(tmpdir, '.png')
    assert np.array_equal(stats.mean, np.mean(image, axis=(0, 1)))
    assert np.array_equal(stats.std, np.std(image, axis=(0, 1)))
    assert np.array_equal(stats.max, np.max(image, axis=(0, 1)))
    assert np.array_equal(stats.min, np.min(image, axis=(0, 1)))
    assert stats.count == 2


def test_from_image_files_empty(tmpdir):
    stats = describe.from_image_files(tmpdir, '.WrongExtension')
    assert stats.count == 0


def test_annotations_per_image(ccagt_df_multi, categories_infos):
    df = ccagt_df_multi.copy()
    df['image_id'] = ccagt.generate_ids(df['image_name'])

    df = describe.annotations_per_image(df, categories_infos)

    assert df['NORs'].tolist() == [2., 4., 0.]
    assert df['count', 'NUCLEUS'].tolist() == [1., 2., 2.]


def test_ccagt_annotations(ccagt_df_multi, categories_infos):
    df = ccagt_df_multi.copy()
    df['image_id'] = ccagt.generate_ids(df['image_name'])
    df['slide_id'] = ccagt.slides_ids(df)
    df['area'] = ccagt.geometries_area(df)

    out = describe.ccagt_annotations(df, categories_infos)

    assert out['qtd_images'] == 3
    assert out['qtd_slide'] == 2
    assert out['qtd_annotations'] == df.shape[0]
    assert out['qtd_annotations_categorical'] == {
        'Nucleus': 5, 'Cluster': 3, 'Satellite': 3, 'Nucleus_out_of_focus': 0,
        'Overlapped_Nuclei': 0, 'non_viable_nucleus': 0, 'Leukocyte_Nucleus': 0,
        'background': 0,
    }
    assert out['dist_annotations'] == {
        'Nucleus': pytest.approx(0.454, 0.01), 'Cluster': pytest.approx(0.273, 0.01),
        'Satellite': pytest.approx(0.273, 0.01), 'Nucleus_out_of_focus': 0.0,
        'Overlapped_Nuclei': 0.0, 'non_viable_nucleus': 0.0, 'Leukocyte_Nucleus': 0.0,
        'background': 0.0,
    }
    assert len(out['area_stats']) == 3


def test_tvt_annotations_as_df(ccagt_df_multi, categories_infos):
    df = ccagt_df_multi.copy()
    df['image_id'] = ccagt.generate_ids(df['image_name'])
    df['slide_id'] = ccagt.slides_ids(df)
    df['area'] = ccagt.geometries_area(df)

    out = describe.ccagt_annotations(df, categories_infos)

    df_qtd, df_dist, df_area = describe.tvt_annotations_as_df(out, out, out)

    assert df_qtd.shape[0] == 4
    assert df_qtd.shape[1] == len(categories_infos) + 4
    assert df_qtd.loc[df_qtd['fold'] == 'total'].shape[0] == 1
    assert df_qtd['images'].tolist() == [3, 3, 3, 9]
    assert df_qtd['annotations'].tolist() == [11, 11, 11, 33]

    assert df_dist.shape[0] == 3
    assert df_dist.shape[1] == len(categories_infos) + 3
    assert df_dist.iloc[0]['% annotations'] == pytest.approx(0.3333, 0.01)
    assert df_dist.iloc[2]['% images'] == pytest.approx(0.3333, 0.01)

    assert df_dist.shape[1] == len(categories_infos) + 3

    cats_with_data = sum(True for v in out['qtd_annotations_categorical'].values() if v > 0)
    assert df_area.shape == (15, 1 + cats_with_data)
    assert df_area.loc[df_area['fold'] == 'train', 'Nucleus'].tolist() == [900., 0., 900., 900., 5.]


def test_count_pixels(shape):
    msk = create.mask_categorical(shape)
    total = shape[0] * shape[1]
    qtd_expected = int(total * 0.25)

    cats = np.unique(msk)
    expected = {cat.value: qtd_expected for cat in Categories if cat.value in cats}

    counts = describe.categorical_mask(msk)
    assert counts == expected
    assert sum(counts.values()) == total


def test_single_core_from_mask_files(shape, tmpdir):
    msk = create.mask_categorical(shape)

    filename = os.path.join(tmpdir, 'test.png')
    Image.fromarray(msk).save(filename)

    out = describe.single_core_from_mask_files([filename])

    total = shape[0] * shape[1]
    qtd_expected = int(total * 0.25)
    cats = np.unique(msk)
    expected = {cat.value: qtd_expected if cat.value in cats else 0 for cat in Categories}

    assert out == expected
    assert sum(out.values()) == total


def test_single_core_from_mask_files_empty():
    with pytest.raises(ValueError):
        describe.single_core_from_mask_files([])


@pytest.mark.slow
def test_from_mask_files(shape, tmpdir):
    msk = create.mask_categorical(shape)
    for i in range(2):
        Image.fromarray(msk).save(os.path.join(tmpdir, f'test{i}.png'))

    out = describe.from_mask_files(tmpdir, '.png')

    total = shape[0] * shape[1]
    qtd_expected = int(total * 0.25)
    cats = np.unique(msk)
    expected = {cat.name: qtd_expected * 2 if cat.value in cats else 0 for cat in Categories}

    assert out == expected
    assert sum(out.values()) == total * 2


def test_describe_dataset(capsys, ccagt_ann_multi_path, categories_infos, tmpdir):
    assert describe.dataset(ccagt_ann_multi_path, categories_infos, tmpdir) is None
    msg, _ = capsys.readouterr()

    assert 'Dataset name: `test_describe_dataset0` |' == msg[:40]

    tmpfilename = tmpdir.join('test.parquet.gzip')

    pd.DataFrame().to_parquet(str(tmpfilename))
    assert describe.dataset(str(tmpfilename), categories_infos, tmpdir) is None
    msg, _ = capsys.readouterr()
    assert msg[-28:] == 'Do not have any annotation!\n'
