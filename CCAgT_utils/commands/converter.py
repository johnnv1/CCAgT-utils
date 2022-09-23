from __future__ import annotations

import argparse
import os
from typing import Sequence

from typing_extensions import TypeAlias

from CCAgT_utils.base.categories import CategoriesInfos
from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.base.utils import open_and_read_json
from CCAgT_utils.converter import from_labelbox
from CCAgT_utils.formats import ccagt
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


def to_coco() -> int:
    raise NotImplementedError
    # return 0


def converter_command(
    args: argparse.Namespace | None = None,
) -> int:
    if args is None:
        return 1

    if args.to_ccagt:
        _fn = args.out_filename
        filename = 'CCAgT.parquet.gzip' if _fn is None else _fn
        out_path = os.path.join(args.out_path, filename)

        return to_ccagt(
            in_path=os.path.abspath(args.in_path),
            out_path=os.path.abspath(out_path),
            aux_path=os.path.abspath(args.aux_path),
            preprocess=args.preprocess,
        )

    return to_coco()


def main(argv: Sequence[str] | None = None) -> int:
    parser = converter_command_parser()
    args = parser.parse_args(argv)
    return converter_command(args)


if __name__ == '__main__':
    raise SystemExit(main())
