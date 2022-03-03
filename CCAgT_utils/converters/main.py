from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from CCAgT_utils.constants import VERSION
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

    help = subparsers.add_parser('help', help='Show help for a specific command.')
    help.add_argument('help_cmd', nargs='?', help='Command to show help for.')

    if len(argv) == 0:
        argv = ['help']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    if args.command == 'labelbox_to_COCO' and args.raw_file != '':
        return labelbox_to_COCO(target=args.target,
                                raw_path=os.path.abspath(args.raw_file),
                                aux_path=os.path.abspath(args.aux_file),
                                out_path=os.path.abspath(args.out_file),
                                image_extension=args.images_extension,
                                decimals=int(args.out_precision))
    elif args.command == 'labelbox_to_CCAgT' and args.raw_file != '':
        return labelbox_to_CCAgT(raw_path=os.path.abspath(args.raw_file),
                                 aux_path=os.path.abspath(args.aux_file),
                                 out_path=os.path.abspath(args.out_file),
                                 image_extension=args.images_extension,
                                 preprocess=args.preprocess)

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
