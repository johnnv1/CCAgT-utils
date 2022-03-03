from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon

from CCAgT_utils.converters import CCAgT
from CCAgT_utils.converters.LabelBox import LabelBox_Annotations
from CCAgT_utils.types.mask import Mask
from CCAgT_utils.visualization import colors
from testing import create


@pytest.fixture
def get_color_rgb():
    # TODO: Maybe load from the json file?
    return {1: [21, 62, 125], 2: [114, 67, 144], 3: [255, 166, 0], 4: [26, 167, 238],
            5: [39, 91, 82], 6: [5, 207, 192], 7: [255, 0, 0]}


@pytest.fixture
def get_color_rgba_norm(get_color_rgb):
    return {k: colors.rgb_to_rgba(v, True) for k, v in get_color_rgb.items()}


@pytest.fixture
def get_categories_name():
    # TODO: Maybe load from the json file?
    return {1: 'Nucleus', 2: 'Cluster', 3: 'Satellite', 4: 'Nucleus_out_of_focus', 5: 'Overlapped_Nuclei',
            6: 'non-viable_nucleus', 7: 'Leukocyte_Nucleus'}


@pytest.fixture
def get_min_area():
    # TODO: Maybe load from the json file?
    return {1: 500, 2: 40, 3: 30, 4: 500, 5: 500, 6: 200, 7: 200}


@pytest.fixture
def categories_aux_data(get_color_rgb, get_categories_name, get_min_area):
    return [{'color': c,
             'id': cat_id,
             'name': get_categories_name[cat_id],
             'labelbox_schemaId': f'<Unique ID for category {get_categories_name[cat_id]}>',
             'minimal_area': get_min_area[cat_id],
             'supercategory': ''} for cat_id, c in get_color_rgb.items()]


@pytest.fixture
def ccagt_metadata():
    template = '''
qtd of images = {number_of_images}
qtd of slides = {number_of_slides}
qtd of annotations = {number_of_annotations}
'''

    return {'url': 'https://127.0.0.1:5000/tests/CCAgT-utils',
            'version': 'test version',
            'description_template': template,
            'university': {
                'name': 'university testname',
                'link': 'https:///127.0.0.1:5000/tests/CCAgT-utils/university',
                'short_name': 'UTEST'
            },
            'contributors': 'Contrib A, Contrib B, Contrib tests'}


@pytest.fixture
def ccagt_aux_data(categories_aux_data, ccagt_metadata):
    return {'categories': categories_aux_data, 'metadata': ccagt_metadata}


@pytest.fixture
def shape():
    return (1200, 1600)


@pytest.fixture
def gray_image(shape):
    return np.zeros(shape, dtype=np.uint8)


@pytest.fixture
def rgb_image(gray_image):
    return np.stack([gray_image, gray_image, gray_image], axis=-1)


@pytest.fixture
def mask_categorical(shape):
    return create.mask_categorical(shape)


@pytest.fixture
def mask_colorized(shape):
    return create.mask_colorized(shape)


@pytest.fixture
def mask(mask_categorical):
    return Mask(mask_categorical)


@pytest.fixture
def nucleus_ex():
    return Polygon([(0, 0), (0, 30), (30, 30), (30, 0)])


@pytest.fixture
def cluster_ex():
    return Polygon([(10, 10), (10, 15), (15, 15), (15, 10)])


@pytest.fixture
def satellite_ex():
    return Point(1, 1)


@pytest.fixture
def lbb_raw_single_satellite(satellite_ex):
    return {'ID': 'a1', 'External ID': 'tmp/A_xxx.png', 'Skipped': False, 'Reviews': [{'score': 1,
                                                                                       'labelId': 'a1'}],
            'Label': {'objects': [{
                'featureId': '<ID for this annotation - 26>',
                'schemaId': '<Unique ID for category Satellite>',
                'color': '#FF4A46',
                'title': 'satellite',
                'value': 'satellite',
                'point': {
                    'x': satellite_ex.x,
                    'y': satellite_ex.y
                },
                'instanceURI': '<URL for this annotation>'}]}
            }


@pytest.fixture
def lbb_raw_single_nucleus(nucleus_ex):
    return {'ID': 'a1', 'External ID': 'tmp/A_xxx.png', 'Skipped': False, 'Reviews': [{'score': 1,
                                                                                       'labelId': 'a1'}],
            'Label': {'objects': [{
                'featureId': '<ID for this annotation - 26>',
                'schemaId': '<Unique ID for category Nucleus>',
                'color': '#1CE6FF',
                'title': 'nucleus',
                'value': 'nucleus',
                'polygon': [{'x': x, 'y': y} for x, y in zip(*nucleus_ex.exterior.xy)],
                'instanceURI': '<URL for this annotation>'}]}
            }


@pytest.fixture
def lbb_raw_single_wrong_nucleus():
    return {'ID': 'a1', 'External ID': 'tmp/A_xxx.png', 'Skipped': False, 'Reviews': [{'score': 1,
                                                                                       'labelId': 'a1'}],
            'Label': {'objects': [{
                'featureId': '<ID for this annotation - 1>',
                'schemaId': '<Unique ID for category Nucleus>',
                'color': '#1CE6FF',
                'title': 'nucleus',
                'value': 'nucleus',
                'polygon': [{
                            'x': 450,
                            'y': 450
                            }, {
                            'x': 460,
                            'y': 460
                            }, {
                            'x': 470,
                            'y': 460}],
                'instanceURI': '<URL for this annotation>'}]}
            }


@pytest.fixture
def lbb_raw_sample_complete(lbb_raw_single_satellite, lbb_raw_single_nucleus):
    o = [lbb_raw_single_satellite,
         lbb_raw_single_nucleus,
         {'ID': 'a2', 'External ID': 'tmp/B_xxx', 'Skipped': True}]
    return o


@pytest.fixture
def lbb_ann(categories_aux_data, lbb_raw_sample_complete):
    return LabelBox_Annotations(lbb_raw_sample_complete, categories_aux_data)


@pytest.fixture
def lbb_raw_expected_ccagt_df(satellite_ex, nucleus_ex):
    d = [create.row_CCAgT(satellite_ex, 3, 'A_xxx'),
         create.row_CCAgT(nucleus_ex, 1, 'A_xxx')]
    return pd.DataFrame(d)


@pytest.fixture
def ccagt_df_multi(nucleus_ex, cluster_ex, satellite_ex):
    # Using a dict with a list for each collum, will raise a warning for Points because of pandas cast type
    d = [create.row_CCAgT(satellite_ex, 3, 'A_xx1'),
         create.row_CCAgT(nucleus_ex, 1, 'A_xx1'),
         create.row_CCAgT(cluster_ex, 2, 'A_xx1'),
         create.row_CCAgT(nucleus_ex, 1, 'B_xx1'),
         create.row_CCAgT(nucleus_ex, 1, 'B_xx1')]
    return pd.DataFrame(d)


@pytest.fixture
def ccagt_ann_single_nucleus(nucleus_ex):
    d = pd.DataFrame([create.row_CCAgT(nucleus_ex, 1, 'C_xx1')])
    return CCAgT.CCAgT_Annotations(d)


@pytest.fixture
def ccagt_ann_multi(ccagt_df_multi):
    return CCAgT.CCAgT_Annotations(ccagt_df_multi)


@pytest.fixture
def coco_ann_single_nucleus(nucleus_ex):
    min_x, min_y, max_x, max_y = nucleus_ex.bounds
    x, y = nucleus_ex.exterior.xy
    pol = [float(val) for pair in zip(x, y) for val in pair]
    return [{'id': 1,
             'image_id': 1,
             'category_id': 1,
             'bbox': [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)],
             'segmentation': [pol],
             'area': float(float(max_x - min_x) * float(max_y - min_y)),
             'iscrowd': 0}]


# def pytest_addoption(parser):
#     parser.addoption("--skip-slow", action="store_true", default=False, help="this will skip the slow tests")


# def pytest_configure(config):
#     config.addinivalue_line("markers", "slow: mark test as slow to run")


# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--skip-slow"):
#         skip_slow = pytest.mark.skip(reason="--skip-slow option activated")
#         for item in items:
#             if "slow" in item.keywords:
#                 item.add_marker(skip_slow)
