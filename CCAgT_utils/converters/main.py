from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from CCAgT_utils.constants import VERSION
from CCAgT_utils.converters.utils import ccagt_generate_masks
from CCAgT_utils.converters.utils import CCAgT_to_COCO
from CCAgT_utils.converters.utils import labelbox_to_CCAgT
from CCAgT_utils.converters.utils import labelbox_to_COCO


def _add_converter_default_options(parser: argparse.ArgumentParser,
                                   have_target: bool = True) -> None:
    if have_target:
        parser.add_argument('-t',
                            '--target',
                            help=('Define the target of the COCO format. Expected `object-detection` or `OD`,'
                                  ' `panoptic-segmentation` or `PD`, `instance-segmentation` or `IS`.'),
                            required=True)
    parser.add_argument('-r',
                        '--raw-file',
                        help='Path for the labelbox raw file. A JSON file is expected.',
                        required=True,
                        metavar='RAW_FILE_LABELBOX_PATH')
    parser.add_argument('-a',
                        '--aux-file',
                        help=('Path for the categories auxiliary/helper file. A JSON file is expected.'),
                        required=True,
                        metavar='HELPER_FILE_PATH')
    parser.add_argument('-e',
                        '--images-extension',
                        help='The extension of the filenames at COCO file. Example `.jpg`',
                        default='')


def _add_mask_generator_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-l',
                        '--labels-path',
                        help='Path for the CCAgT file. A parquet file is expected.',
                        required=True,
                        metavar='CCAgT_FILE_PATH')
    parser.add_argument('-o',
                        '--output-dir',
                        help=('Path for the directory where the masks should be saved.'),
                        required=True)
    parser.add_argument('--split-by-slide',
                        help='To save the masks into subdirectories for each slide',
                        action='store_true')
    # TODO: colorized option
    # parser.add_argument('--colorized',
    #                     help='Flag to generate RGB masks if desired',
    #                     default=False)


def _add_ccagt_to_coco_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-t',
                        '--target',
                        help='Define the target of the COCO format.',
                        choices=['object-detection', 'OD', 'panoptic-segmentation', 'PS', 'instance-segmentation', 'IS'],
                        required=True)
    parser.add_argument('-l',
                        '--labels-path',
                        help='Path for the CCAgT file. A parquet file is expected.',
                        required=True,
                        metavar='CCAgT_FILE_PATH')
    parser.add_argument('-o',
                        '--output-dir',
                        help=('Path for the directory where the masks should be saved.'),
                        required=True)
    parser.add_argument('--out-file',
                        help=('Path for the output file. A JSON file is expected. By default will be '
                              '<output-dir>/CCAgT_COCO_format_<target>.json'),
                        metavar='OUTPUT_PATH')
    parser.add_argument('-a',
                        '--aux-file',
                        help=('Path for the categories auxiliary/helper file. A JSON file is expected.'),
                        metavar='HELPER_FILE_PATH')
    parser.add_argument('--split-by-slide',
                        help='To save the masks into subdirectories for each slide',
                        action='store_true')


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog='CCAgT_converter')

    # https://stackoverflow.com/a/8521644/812183
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {VERSION}')

    subparsers = parser.add_subparsers(dest='command')

    lb_to_coco_parser = subparsers.add_parser('labelbox_to_COCO', help='Converter from Raw labelbox file to a COCO formart')
    _add_converter_default_options(lb_to_coco_parser)
    lb_to_coco_parser.add_argument('-o',
                                   '--out-file',
                                   help='Path for the output file. A JSON file is expected.',
                                   default='CCAgT_COCO_format.json',
                                   metavar='OUTPUT_PATH')
    lb_to_coco_parser.add_argument('-p',
                                   '--out-precision',
                                   help='The number of digits (decimals), for the coords at output file',
                                   default=2,
                                   metavar='OUTPUT_PRECISION')

    lb_to_ccagt_parser = subparsers.add_parser('labelbox_to_CCAgT',
                                               help='Converter from Raw labelbox file to a CCAgT format.')
    _add_converter_default_options(lb_to_ccagt_parser, have_target=False)
    lb_to_ccagt_parser.add_argument('-o',
                                    '--out-file',
                                    help='Path for the output file. A parquet file is expected.',
                                    default='CCAgT.parquet.gzip',
                                    metavar='OUTPUT_PATH')
    lb_to_ccagt_parser.add_argument('-p',
                                    '--preprocess',
                                    help='Flag to define if want to run teh preprocessing steps',
                                    default=False)

    ccagt_to_mask = subparsers.add_parser('generate_masks', help='Converter from CCAgT file to masks')
    _add_mask_generator_options(ccagt_to_mask)

    ccagt_to_coco_parser = subparsers.add_parser('CCAgT_to_COCO',
                                                 help='Converter from CCAgT annotations to COCO annotations')
    _add_ccagt_to_coco_options(ccagt_to_coco_parser)

    help = subparsers.add_parser('help', help='Show help for a specific command.')
    help.add_argument('help_cmd', nargs='?', help='Command to show help for.')

    if len(argv) == 0:
        argv = ['help']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    if args.command == 'labelbox_to_COCO':
        return labelbox_to_COCO(target=args.target,
                                raw_path=os.path.abspath(args.raw_file),
                                aux_path=os.path.abspath(args.aux_file),
                                out_path=os.path.abspath(args.out_file),
                                image_extension=args.images_extension,
                                decimals=int(args.out_precision))
    elif args.command == 'labelbox_to_CCAgT':
        return labelbox_to_CCAgT(raw_path=os.path.abspath(args.raw_file),
                                 aux_path=os.path.abspath(args.aux_file),
                                 out_path=os.path.abspath(args.out_file),
                                 image_extension=args.images_extension,
                                 preprocess=args.preprocess)
    elif args.command == 'CCAgT_to_COCO':
        return CCAgT_to_COCO(target=args.target,
                             ccagt_path=os.path.abspath(args.labels_path),
                             aux_path=args.aux_file,
                             out_dir=os.path.abspath(args.output_dir),
                             out_file=args.out_file,
                             split_by_slide=args.split_by_slide)

    elif args.command == 'generate_masks' and args.labels_path != '':
        return ccagt_generate_masks(os.path.abspath(args.labels_path),
                                    os.path.abspath(args.output_dir),
                                    args.split_by_slide)

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
