from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from CCAgT_utils import Categories
from CCAgT_utils.converters.CCAgT import CCAgT_Annotations
from CCAgT_utils.converters.LabelBox import LabelBox_Annotations
from CCAgT_utils.errors import FileTypeError


def labelbox_to_COCO(target: str,
                     raw_path: str,
                     aux_path: str,
                     out_path: str,
                     image_extension: str,
                     decimals: int) -> int:

    if not raw_path.endswith('.json'):
        raise FileTypeError('The raw file is not a JSON file.')
    if not aux_path.endswith('.json'):
        raise FileTypeError('The auxiliary file is not a JSON file.')
    if not out_path.endswith('.json'):
        raise FileTypeError('The output file is not a JSON file.')

    print('Starting the convertion from labelbox to COCO...')
    print(f'Target = {target}')
    print(f'Raw path = {raw_path}')
    print(f'Aux path = {aux_path}')
    print(f'Out path = {out_path}')

    target = target.upper()

    if target in ['OBJECT-DETECTION', 'OD']:
        return labelbox_to_OD_COCO(raw_path, aux_path, out_path, image_extension, decimals)
    elif target in ['INSTANCE-SEGMENTATION', 'IS']:
        raise NotImplementedError
    elif target in ['PANOPTIC-SEGMENTATION', 'PS']:
        raise NotImplementedError

    return 0


def open_and_read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def __build_description(template: str, df: pd.Dataframe) -> str:
    img_quantity = len(df['image_id'].unique())
    slide_quantity = len(df['slide_id'].unique())
    annotations_quantity = df.shape[0]

    o = template.replace('{number_of_images}', str(img_quantity))
    o = o.replace('{number_of_slides}', str(slide_quantity))
    o = o.replace('{number_of_annotations}', str(annotations_quantity))

    return o


def __prepare_data(CCAgT_ann: CCAgT_Annotations,
                   categories_helper_raw: list[dict[str, Any]],
                   image_extension: str) -> None:
    print('\tSearching overlapping and joining labels for overlapping annotations (category id = 5)...')
    overlapping_annotations = CCAgT_ann.find_overlapping_annotations(categories_id={5})
    df = CCAgT_ann.union_geometries(overlapping_annotations)

    print('\tDefine the geometry type and transform Satellite (category id = 3) points into Polygons...')
    df['geo_type'] = CCAgT_ann.geometries_type()
    sat_series = df.loc[(df['category_id'] == 3) & (df['geo_type'] == 'Point'), 'geometry']

    df.loc[(df['category_id'] == 3) &
           (df['geo_type'] == 'Point'), 'geometry'] = CCAgT_ann.satellite_point_to_polygon(sat_series)

    df['geo_type'] = CCAgT_ann.geometries_type()

    print('\tVerify and fit the geometries into the images boundary...')
    df['geometry'] = CCAgT_ann.fit_geometries_to_image_boundary()
    df.dropna(axis=0, subset=['geometry'], inplace=True)

    if len(image_extension) > 0:
        print(f'\tSetting {image_extension} as the images extensions')
        df['image_name'] = df['image_name'].apply(lambda x: x + image_extension)

    print('\tComputing the annotations area and the images IDs...')
    df['area'] = CCAgT_ann.geometries_area()
    df['image_id'] = CCAgT_ann.generate_ids(df['image_name'])
    df['slide_id'] = CCAgT_ann.get_slide_id()

    print('\tDeleting annotations based on the minimal area settet at auxiliary file.')
    ccagt_helper = Categories.Helper(categories_helper_raw)
    min_area = ccagt_helper.min_area_by_category_id
    df = CCAgT_ann.delete_by_area(min_area)

    print('\tSearching intersections of nuclei with NORs labels (category id in [1] and [2, 3])...')
    df_base_intersects_target = CCAgT_ann.verify_if_intersects(base_categories_id={1}, target_categories_id={2, 3})
    if not df_base_intersects_target.empty:
        index_to_drop = df_base_intersects_target[~df_base_intersects_target['has_intersecting']].index.to_numpy()
        print(f'\t\tA total of {len(index_to_drop)} of nuclei (category id = 1) will be deleted.')
        df.drop(index_to_drop, inplace=True)

    print('\tSearching intersections of NORs with nuclei (normal and overlapped) labels (category id in [2, 3] and [1, 5])..')
    df_base_intersects_target = CCAgT_ann.verify_if_intersects(base_categories_id={2, 3}, target_categories_id={1, 5})
    if not df_base_intersects_target.empty:
        index_to_drop = df_base_intersects_target[~df_base_intersects_target['has_intersecting']].index.to_numpy()
        print(f'\t\tA total of {len(index_to_drop)} of NORs (category id = [2, 3]) will be deleted.')
        df.drop(index_to_drop, inplace=True)


def labelbox_to_OD_COCO(raw_path: str,
                        aux_path: str,
                        out_path: str,
                        image_extension: str,
                        decimals: int) -> int:
    print('Starting the conversion from label box to COCO Object Detection format...')

    print('\tLoading raw data...')
    labelbox_raw = open_and_read_json(raw_path)
    print('\tLoading auxiliary data...')
    dataset_helper = open_and_read_json(aux_path)
    categories_helpper = dataset_helper['categories']

    lb_ann = LabelBox_Annotations(labelbox_raw, categories_helpper)

    print('\tConverting RAW data into CCAgT formart...')
    CCAgT_ann = lb_ann.to_CCAgT()

    __prepare_data(CCAgT_ann, categories_helpper, image_extension)

    print('\tTransforming annotations from CCAgT format to COCO Object Detection Format...')
    annotations_coco = CCAgT_ann.to_OD_COCO(decimals=decimals)

    desc = __build_description(dataset_helper['metadata']['description_template'], CCAgT_ann.df)

    info_coco = {'year': datetime.now().strftime('%Y'),
                 'version': dataset_helper['metadata']['version'],
                 'description': desc,
                 'contributor': dataset_helper['metadata']['contributors'],
                 'url': dataset_helper['metadata']['url'],
                 'date_created': datetime.now().strftime('%Y-%m-%d')}

    categories_coco = [{'supercategory': None if it['supercategory'] == '' else it['supercategory'],
                        'name': it['name'],
                        'id': it['id']} for it in categories_helpper]

    image_names = CCAgT_ann.df['image_name'].unique().tolist()

    images_coco = [{'file_name': img_name,
                    'height': CCAgT_ann.IMAGE_HEIGHT,
                    'width': CCAgT_ann.IMAGE_WIDTH,
                    'id': CCAgT_ann.image_id_by_name(img_name)} for img_name in image_names]

    CCAgT_coco = {'info': info_coco,
                  'categories': categories_coco,
                  'images': images_coco,
                  'annotations': annotations_coco}

    with open(out_path, 'w') as outfile:
        json.dump(CCAgT_coco, outfile)

    return 0


def labelbox_to_CCAgT(raw_path: str,
                      aux_path: str,
                      out_path: str,
                      image_extension: str,
                      preprocess: bool = False) -> int:
    if not raw_path.endswith('.json'):
        raise FileTypeError('The raw file is not a JSON file.')
    if not aux_path.endswith('.json'):
        raise FileTypeError('The auxiliary file is not a JSON file.')
    if out_path.endswith('.parquet'):
        out_path += '.gzip'
    else:
        if not out_path.endswith('.parquet.gzip'):
            raise FileTypeError('The output file is not a parquet.gzip file.')

    print('Starting the convertion from labelbox to COCO...')
    print(f'Raw path = {raw_path}')
    print(f'Aux path = {aux_path}')
    print(f'Out path = {out_path}')

    print('Starting the conversion from label box to CCAgT format...')

    print('\tLoading raw data...')
    labelbox_raw = open_and_read_json(raw_path)
    print('\tLoading auxiliary data...')
    dataset_helper = open_and_read_json(aux_path)
    categories_helpper = dataset_helper['categories']

    lb_ann = LabelBox_Annotations(labelbox_raw, categories_helpper)

    print('\tConverting RAW data into CCAgT formart...')
    CCAgT_ann = lb_ann.to_CCAgT()

    if preprocess:
        __prepare_data(CCAgT_ann, categories_helpper, image_extension)

    print('\tSaving the CCAgT to a parquet file...')
    CCAgT_ann.to_parquet(out_path)

    return 0
