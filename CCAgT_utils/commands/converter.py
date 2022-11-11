from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any
from typing import Sequence

from typing_extensions import TypeAlias

from CCAgT_utils.base.categories import Categories
from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.base.utils import open_and_read_json
from CCAgT_utils.converter import from_labelbox
from CCAgT_utils.converter import to_coco_instances
from CCAgT_utils.converter import to_mask
from CCAgT_utils.formats import ccagt
from CCAgT_utils.formats import coco
from CCAgT_utils.formats.labelbox import LabelBox
from CCAgT_utils.prepare import ccagt_dataset

SUBPARSER_T: TypeAlias = 'argparse._SubParsersAction[argparse.ArgumentParser]'


def converter_command_parser(
        subparsers: SUBPARSER_T | None = None,
) -> argparse.ArgumentParser:

    if subparsers is not None:
        parser = subparsers.add_parser('convert')
    else:
        parser = argparse.ArgumentParser(
            prog='CCAgT-utils dataset converter command',
        )

    group_output = parser.add_argument_group(
        'Define the output format desired',
    )

    mexg_output = group_output.add_mutually_exclusive_group(required=True)
    mexg_output.add_argument(
        '--to-ccagt',
        action='store_true',
        help=(
            'Convert the input data into the CCAgT dataset format. This '
            'conversion is allowed from LabelBox format.'
        ),
    )

    mexg_output.add_argument(
        '--to-coco',
        action='store_true',
        help=(
            'Convert the input data into the COCO dataset format. This '
            'conversion is allowed from CCAgT format.'
        ),
    )

    mexg_output.add_argument(
        '--to-masks',
        action='store_true',
        help=(
            'Convert the input data into semantic segmentation masks. This '
            'conversion is allowed from CCAgT format.'
        ),
    )

    converter_group = parser.add_argument_group(
        'General parameters for the converters command',
    )

    converter_group.add_argument(
        '-i',
        '--in-path',
        help='Path for the input file.',
        required=True,
        metavar='INPUT_PATH',
    )

    converter_group.add_argument(
        '-o',
        '--out-path',
        help='Path for the output dir.',
        required=True,
        metavar='OUTPUT_DIR',
    )

    converter_group.add_argument(
        '-f',
        '--out-filename',
        help='Filename for the output file.',
        required=False,
        metavar='OUTPUT_FILENAME',
    )

    ccagt_group = parser.add_argument_group(
        'Parameters to work with CCAgT file',
    )
    ccagt_group.add_argument(
        '-p',
        '--preprocess',
        help='Flag to define if want to run the preprocessing pipeline',
        action='store_true',
    )

    labelbox_group = parser.add_argument_group(
        'Parameters to work with LabelBox files',
    )

    labelbox_group.add_argument(
        '-a',
        '--aux-path',
        help='Path for the auxiliary file for the categories.',
        required=False,
        metavar='AUX_PATH',
    )

    coco_group = parser.add_argument_group(
        'Parameters to work with COCO files',
    )

    coco_group.add_argument(
        '--target',
        help='Define the target of the COCO format.',
        choices=[
            'object-detection', 'OD', 'panoptic-segmentation', 'PS',
            'instance-segmentation', 'IS',
        ],
        required=False,
    )

    coco_group.add_argument(
        '--precision',
        default=2,
        help='Define the quantity of decimals to be saved at coco file.',
        required=False,
    )

    masks_group = parser.add_argument_group(
        'Parameters for the masks',
    )

    masks_group.add_argument(
        '--split-by-slide',
        help='To save the masks into subdirectories for each slide',
        action='store_true',
    )

    if subparsers is not None:
        parser.set_defaults(func=converter_command)
        parser.set_defaults(checker=check_arguments)

    return parser


def check_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.to_coco and (args.target is None):
        parser.error('--to-coco requires --target')
    elif args.to_ccagt and (args.aux_path is None):
        parser.error('--to-ccagt requires --aux-path')


def to_ccagt(
    in_path: str,
    out_path: str,
    aux_path: str,
    preprocess: bool,
) -> int:
    if not out_path.endswith('.parquet.gzip'):
        raise FileTypeError('The output file is not a parquet.gzip file.')

    print('Starting the conversion from labelbox to CCAgT...')
    print(f'Input path = {in_path}')
    print(f'Auxiliary path = {aux_path}')
    print(f'Output path = {out_path}')

    print('\tLoading input data...')
    labelbox_raw = open_and_read_json(in_path)
    labelbox_df = LabelBox(labelbox_raw)

    print('\tLoading auxiliary data...')
    auxiliary_raw = open_and_read_json(aux_path)
    categories_infos = CategoriesInfos(auxiliary_raw['categories'])

    print('\tConverting the LabelBox to CCAgT...')
    ccagt_df = from_labelbox(labelbox_df, categories_infos)

    if preprocess:
        ccagt_df = ccagt_dataset(ccagt_df, categories_infos)

    print('\tSaving the CCAgT...')
    ccagt.save(ccagt_df, out_path)

    return 0


def to_coco_instances_pipeline(
    ccagt_df: ccagt.CCAgT,
    categories_info: CategoriesInfos,
    out_path: str,
    info_coco: dict[str, Any],
    precision: int,
) -> int:
    print(
        '>Setting all overlapped nuclei as iscrowd (1), and others as 0'
        '(false for the iscrowd)',
    )
    ccagt_df['iscrowd'] = 0

    ccagt_df.loc[
        ccagt_df['category_id'] == Categories.OVERLAPPED_NUCLEI.value,
        'iscrowd',
    ] = 1

    print(
        '>Generating annotations from CCAgT to COCO instance detection/'
        'segmentation...',
    )
    detection_records = to_coco_instances(ccagt_df, precision)

    print('>Building COCO `categories`!')
    categories_coco = [
        {
            'supercategory': it.supercategory,
            'name': it.name,
            'id': it.id,
        } for it in categories_info
    ]

    print('>Building COCO `images`!')
    images_coco = [
        {
            'file_name': str(df['image_name'].unique()[0]),
            'height': int(df['image_height'].unique()[0]),
            'width': int(df['image_width'].unique()[0]),
            'id': int(img_id),
        } for img_id, df in ccagt_df.groupby('image_id')
    ]

    print('>Building COCO Object Detection file!')
    CCAgT_coco = {
        'info': info_coco,
        'categories': categories_coco,
        'images': images_coco,
        'annotations': detection_records,
    }

    coco.validate(CCAgT_coco, 'instance')
    coco.save(CCAgT_coco, out_path)

    return 0


def to_coco(
    target: str,
    in_path: str,
    out_path: str,
    aux_path: str | None,
    split_by_slide: bool = False,
    precision: int = 2,
) -> int:

    ni_target = {'PANOPTIC-SEGMENTATION', 'PS'}
    if target in ni_target:
        raise NotImplementedError

    print('Starting the conversion from labelbox to CCAgT...')

    if os.path.isdir(out_path):
        out_filename = os.path.join(out_path, f'CCAgT_COCO_{target}.json')
        out_dir = out_path
    else:
        out_filename = out_path
        out_dir = os.path.dirname(out_path)

    print(f'Input path = {in_path}')
    print(
        f'Auxiliary path = {aux_path} (If None, the default CategoriesInfo'
        'will be used!)',
    )
    print(f'Output file = {out_filename}')
    print(f'Output directory = {out_dir}')

    print(f'Loading CCAgT annotations from {in_path}')
    ccagt_df = ccagt.load(in_path)

    print('Computing the annotations area and the images IDs...')
    ccagt_df['area'] = ccagt.geometries_area(ccagt_df)
    ccagt_df['image_id'] = ccagt.generate_ids(ccagt_df['image_name'])
    ccagt_df['slide_id'] = ccagt.slides_ids(ccagt_df)

    info_coco = {
        'year': datetime.now().strftime('%Y'),
        'date_created': datetime.now().strftime('%Y-%m-%d'),
    }

    if aux_path is None:
        categories_infos = CategoriesInfos()

    else:
        print('\tLoading auxiliary data...')
        dataset_helper = open_and_read_json(aux_path)
        categories_infos = CategoriesInfos(dataset_helper['categories'])

        desc = coco.build_description(
            dataset_helper['metadata']['description_template'],
            ccagt_df,
        )
        info_coco['version'] = dataset_helper['metadata']['version']
        info_coco['description'] = desc
        info_coco['contributor'] = dataset_helper['metadata']['contributors']
        info_coco['url'] = dataset_helper['metadata']['url']

    return to_coco_instances_pipeline(
        ccagt_df,
        categories_infos,
        out_filename,
        info_coco,
        precision,
    )


def to_masks(
    in_path: str,
    out_path: str,
    split_by_slide: bool,
) -> int:

    print(f'Loading CCAgT annotations from {in_path}')
    ccagt_df = ccagt.load(in_path)

    print('Generating the masks...')
    to_mask(ccagt_df, out_path, split_by_slide)

    return 0


def converter_command(
    args: argparse.Namespace | None = None,
) -> int:
    if args is None:
        return 1

    _fn = args.out_filename
    if args.to_ccagt:
        filename = 'CCAgT.parquet.gzip' if _fn is None else _fn
        out_path = os.path.join(args.out_path, filename)
        return to_ccagt(
            in_path=os.path.abspath(args.in_path),
            out_path=os.path.abspath(out_path),
            aux_path=os.path.abspath(args.aux_path),
            preprocess=args.preprocess,
        )
    elif args.to_masks:
        return to_masks(
            in_path=os.path.abspath(args.in_path),
            out_path=os.path.abspath(args.out_path),
            split_by_slide=args.split_by_slide,
        )

    _ap = args.aux_path
    aux_path = os.path.abspath(_ap) if isinstance(_ap, str) else None
    _op = args.out_path
    out_path = os.path.join(_op, _fn) if isinstance(_fn, str) else _op

    return to_coco(
        target=args.target.upper(),
        in_path=os.path.abspath(args.in_path),
        out_path=os.path.abspath(out_path),
        aux_path=aux_path,
        split_by_slide=args.split_by_slide,
        precision=int(args.precision),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = converter_command_parser()
    args = parser.parse_args(argv)
    return converter_command(args)


if __name__ == '__main__':
    raise SystemExit(main())
