from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
from typing import Sequence

from CCAgT_utils import slice
from CCAgT_utils.checkers import masks_that_has
from CCAgT_utils.constants import VERSION
from CCAgT_utils.prepare import clean_images_and_masks
from CCAgT_utils.prepare import extract_image_and_mask_by_category
from CCAgT_utils.utils import basename
from CCAgT_utils.utils import find_files


def _add_create_subdataset_options(parser: argparse.ArgumentParser) -> None:

    parser.add_argument('--name',
                        metavar='SUBDATASET NAME',
                        required=True,
                        help='The name of the subdataset that will be generated')

    parser.add_argument('--original',
                        metavar='PATH_FOR_ORIGINAL_DATASET',
                        required=True,
                        help=('Path for the original dataset. It is expected that this directory has the subdirectories '
                              '`images/` and `masks/`'))
    parser.add_argument('--output',
                        metavar='PATH_TO_WRITE_THE_SUBDATASETS',
                        required=True,
                        help=('Path to write the new subdataset. It will create a directory with this subdataset name, and '
                              'subdirectories `images/` and `masks/`'))

    group_ex = parser.add_mutually_exclusive_group(required=False)
    group_ex.add_argument('--slice-images',
                          nargs=2,
                          metavar='HORIZONTAL VERTICAL',
                          type=int,
                          help=('Define that wants slice the images into smaller parts. Needs to pass the amount of slice '
                                'desired for the horizontal and vertical split of the images.'))
    group_extract = group_ex.add_argument_group('Extract options')

    group_extract.add_argument('--extract',
                               nargs='*',
                               type=int,
                               metavar='CATEGORIES IDs',
                               help=('Define that wants extract based on one category. Will generate one image for each '
                                     'instance of the desired category.'))
    group_extract.add_argument('--labels',
                               help='Path for the CCAgT file with the labels. Just works with --extract')

    group_extract.add_argument('--paddings',
                               default=0,
                               help=('In percent (float values) or pixels (integer values) select, the size of paddings to '
                                     'apply. Just works with --extract'))

    parser.add_argument('--remove-images-without',
                        nargs='*',
                        type=int,
                        metavar='CATEGORIES IDs',
                        help=('Define that you wants remove of this subdataset the images that does not have the categories '
                              'passed as argument.'))

    group = parser.add_argument_group('Check if all images have at least one of the desired categories.')
    group.add_argument('--check-if-all-have-at-least-one-of',
                       nargs='*',
                       metavar='CATEGORIES IDs',
                       type=int,
                       help=('Define that you wants check if all images have at least one of the categories passed as '
                             'argument.'))
    group.add_argument('--delete',
                       action='store_true',
                       help='To remove all images that do not have at least one of desired categories.')

    parser.add_argument('--extensions',
                        nargs='*',
                        default=('.jpg', '.png'),
                        help='The extensions of the images and masks.')


def create_subdataset(name: str,
                      original_dir: str,
                      output_base: str,
                      slice_images: tuple[int, ...] | None,
                      extract: set[int] | None,
                      categories_to_keep: set[int] | None,
                      check_if_all_have_one: set[int] | None,
                      delete: bool,
                      CCAgT_path: str | None,
                      paddings: str,
                      extensions: tuple[str, ...] = ('.jpg', '.png')) -> int:
    output_dir = os.path.join(output_base, name)
    output_images_dir = os.path.join(output_dir, 'images/')
    output_masks_dir = os.path.join(output_dir, 'masks/')

    if os.path.isdir(output_images_dir) or os.path.isdir(output_masks_dir):
        print(f'Already exist a dataset with name={name} at {output_base}!', file=sys.stderr)
        return 1

    input_images_dir = os.path.join(original_dir, 'images/')
    input_masks_dir = os.path.join(original_dir, 'masks/')

    if not os.path.isdir(input_images_dir) or not os.path.isdir(input_masks_dir):
        print(f'Do not found the original dataset at {input_images_dir} or {input_masks_dir}!', file=sys.stderr)
        return 1

    print('\n\n------------------------')
    if slice_images:
        slice_images = tuple(slice_images)
        print(f'Create images and masks splitting then into {slice_images} (horizontal, vertical) parts')
        slice.images_and_masks(input_images_dir,
                               input_masks_dir,
                               output_dir,
                               slice_images[0], slice_images[1],
                               extension=extensions,
                               look_recursive=True)
    elif extract:
        if CCAgT_path:
            print(f'Create images and masks for each instance of the categories {extract}')
            extract_image_and_mask_by_category(input_images_dir,
                                               input_masks_dir,
                                               output_dir,
                                               extract,
                                               CCAgT_path,
                                               ast.literal_eval(paddings),
                                               extensions,
                                               True)
        else:
            print('When using `--extract`, please specify the labels files with `--labels` argument', file=sys.stderr)
            return 1
    else:
        print('The images and masks of the subdataset will be copied from the original dataset!')
        print('Coping images files...')
        shutil.copytree(input_images_dir, output_images_dir)
        print('Coping masks files...')
        shutil.copytree(input_masks_dir, output_masks_dir)

    if categories_to_keep:
        print('------------------------')
        print(f'Delete images that do not have the categories: {categories_to_keep} ')
        clean_images_and_masks(output_images_dir, output_masks_dir, set(categories_to_keep), extensions)

    if check_if_all_have_one:
        print('------------------------')
        check_if_all_have_one = set(check_if_all_have_one)

        masks_with = masks_that_has(output_masks_dir, check_if_all_have_one, extensions, True)
        all_masks = {basename(k): v for k, v in find_files(output_masks_dir, extensions, True).items()}
        diff = set(all_masks.keys()) - masks_with

        if len(diff) > 0:
            print(f'A total of {len(diff)} files there is not at least one of the categories {check_if_all_have_one}',
                  file=sys.stderr)

            if delete:
                all_images = {basename(k): v for k, v in find_files(output_images_dir, extensions, True).items()}
                print('Deleting images and masks that do not have at least one of the selected categories...')
                for bn in diff:
                    os.remove(all_masks[bn])
                    os.remove(all_images[bn])
        else:
            print(f'Everything is ok, having 0 files that do not have any of the categories {check_if_all_have_one}')
    print('\n------------------------')
    print('Creation of the subdataset finished!')
    print(f'Dataset name: `{name}` | Location: `{output_dir}`')
    images_quantity = len(find_files(output_images_dir, extensions, True))
    masks_quantity = len(find_files(output_masks_dir, extensions, True))
    print(f'Total of images: {images_quantity}')
    print(f'Total of images: {masks_quantity}')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog='CCAgT_converter')

    # https://stackoverflow.com/a/8521644/812183
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {VERSION}')

    subparsers = parser.add_subparsers(dest='command')

    create_subdataset_parser = subparsers.add_parser('create-subdataset',
                                                     help=('Based on the original dataset create a personalized version of '
                                                           'the dataset with the desired modifications. Examples: slice the '
                                                           'images into smaller parts, select just images with some category,'
                                                           ' images of a specific category.'))
    _add_create_subdataset_options(create_subdataset_parser)

    help = subparsers.add_parser('help', help='Show help for a specific command.')
    help.add_argument('help_cmd', nargs='?', help='Command to show help for.')

    if len(argv) == 0:
        argv = ['help']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    if args.command == 'create-subdataset' and args.name != '':
        return create_subdataset(str(args.name),
                                 str(args.original),
                                 str(args.output),
                                 args.slice_images,
                                 args.extract,
                                 args.remove_images_without,
                                 args.check_if_all_have_at_least_one_of,
                                 args.delete,
                                 args.labels,
                                 args.paddings,
                                 tuple(args.extensions))

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
