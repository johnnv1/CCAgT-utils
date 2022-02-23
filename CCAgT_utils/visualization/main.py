from __future__ import annotations

import argparse
import sys
from typing import Sequence

from CCAgT_utils import Categories
from CCAgT_utils.converters import CCAgT
from CCAgT_utils.visualization import _show


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog='CCAgT_visualization')

    subparsers = parser.add_subparsers(dest='command')
    show_parser = subparsers.add_parser('show', help='To show the image with the boundary boxes.')
    show_parser.add_argument('-l',
                             '--labels-file',
                             help='Path for the CCAgT file with the labels.',
                             required=True)
    show_parser.add_argument('-a',
                             '--aux-file',
                             help='Path for the categories auxiliary/helper file. A JSON file is expected.',
                             required=True,
                             metavar='HELPER_FILE_PATH')
    show_parser.add_argument('-t',
                             '--type',
                             help='The type of plots desired.',
                             default='image-with-boxes',
                             choices=['image-with-boxes'])
    show_parser.add_argument('-i',
                             '--images-names',
                             help='Filenames of the images to plot. If nothing be passed, all images will be plotted',
                             default=[],
                             nargs='+')
    show_parser.add_argument('-d',
                             '--dir-path',
                             help='Path for a directory that have the images.',
                             default='./')
    show_parser.add_argument('-s',
                             '--shuffle-images',
                             help='To shuffle the images order',
                             default=True)
    show_parser.add_argument('-e',
                             '--image-extension',
                             help='Define the extension file of the images.',
                             default='.jpg')

    help = subparsers.add_parser('help', help='Show help for a specific command.')
    help.add_argument('help_cmd', nargs='?', help='Command to show help for.')

    if len(argv) == 0:
        argv = ['show']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    if args.command == 'show':
        CCAgT_ann = CCAgT.read_parquet(args.labels_file)
        CCAgT_helper = Categories.read_json(args.aux_file)
        if args.type == 'image-with-boxes':
            return _show.image_with_boxes(CCAgT_ann,
                                          CCAgT_helper,
                                          args.dir_path,
                                          args.image_extension,
                                          args.images_names,
                                          args.shuffle_images)

    return 1
