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
                             '--plot-type',
                             help='The type of plots desired.',
                             default='image-with-boxes',
                             choices=['image-with-boxes', 'image-and-mask'])
    show_parser.add_argument('-i',
                             '--images-names',
                             help='Filenames of the images to plot. If nothing be passed, all images will be plotted',
                             default=[],
                             nargs='+')
    show_parser.add_argument('-d',
                             '--dir-path',
                             help='Path for a directory that have the images.',
                             default='./')
    show_parser.add_argument('-m',
                             '--dir-masks-path',
                             help='Path for a directory that have the masks.',
                             default='./')
    show_parser.add_argument('-s',
                             '--shuffle-images',
                             help='To shuffle the images order',
                             default=True)
    show_parser.add_argument('-e',
                             '--image-extension',
                             help='Define the extension file of the images.',
                             default='.jpg')
    show_parser.add_argument('--mask-extension',
                             help='Define the extension file of the masks.',
                             default='.png')
    show_parser.add_argument('-r',
                             '--look-recursive',
                             help='Define if needs to look into the subdirectories of the --dir-path for find the images.',
                             default=True)

    help = subparsers.add_parser('help', help='Show help for a specific command.')
    help.add_argument('help_cmd', nargs='?', help='Command to show help for.')

    if len(argv) == 0:
        argv = ['show']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    if args.command == 'show' and (args.labels_file != '' and args.aux_file != ''):
        CCAgT_ann = CCAgT.read_parquet(args.labels_file)
        CCAgT_helper = Categories.read_json(args.aux_file)
        if args.plot_type == 'image-with-boxes' and args.dir_path != '':
            return _show.image_with_boxes(CCAgT_ann,
                                          CCAgT_helper,
                                          args.dir_path,
                                          args.image_extension,
                                          args.images_names,
                                          args.shuffle_images,
                                          args.look_recursive)
        elif args.plot_type == 'image-and-mask' and args.dir_path != '':
            return _show.image_and_mask(CCAgT_ann,
                                        CCAgT_helper,
                                        args.dir_path,
                                        args.dir_masks_path,
                                        args.image_extension,
                                        args.mask_extension,
                                        args.images_names,
                                        args.shuffle_images,
                                        args.look_recursive)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
