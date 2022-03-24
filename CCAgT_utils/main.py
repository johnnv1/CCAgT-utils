from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
from typing import Sequence

from CCAgT_utils import describe
from CCAgT_utils import slice
from CCAgT_utils.categories import CategoriesInfos
from CCAgT_utils.constants import VERSION
from CCAgT_utils.converters.CCAgT import read_parquet
from CCAgT_utils.converters.utils import ccagt_generate_masks
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
                              '`images/` and `masks/`, and `CCAgT.parquet.gzip`'))
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
                               metavar='CATEGORY ID',
                               help=('Define that wants extract based on one category. Will generate one image for each '
                                     'instance of the desired category.'))

    group_extract.add_argument('--paddings',
                               default=0,
                               help=('In percent (float values) or pixels (integer values) select, the size of paddings to '
                                     'apply. Just works with --extract'))

    parser.add_argument('--remove-images-without',
                        nargs='*',
                        type=int,
                        metavar='CATEGORY ID',
                        help=('Define that you wants remove of this subdataset the images that does not have the categories '
                              'passed as argument.'))

    group = parser.add_argument_group('Check if all images have at least one of the desired categories.')
    group.add_argument('--check-if-all-have-at-least-one-of',
                       nargs='*',
                       metavar='CATEGORY ID',
                       type=int,
                       help=('Define that you wants check if all images have at least one of the categories passed as '
                             'argument.'))
    group.add_argument('--delete',
                       action='store_true',
                       help='To remove all images that do not have at least one of desired categories.')

    group.add_argument('--generate-masks',
                       action='store_true',
                       help='To generate the masks for semantic segmentation')

    parser.add_argument('--labels',
                        help='Path for the CCAgT file with the labels. By default will look at PATH_FOR_ORIGINAL_DATASET')

    parser.add_argument('--extensions',
                        nargs='*',
                        default=('.jpg', '.png'),
                        help='The extensions of the images and masks.')


STRUCTURE = {'i': 'images/', 'm': 'masks/', 'l': 'CCAgT.parquet.gzip'}


def copy_original_dataset(original_images: str,
                          original_annotations: str,
                          output_images: str,
                          output_annotations: str) -> None:

    print('The images and masks of the subdataset will be copied from the original dataset!')
    print('Coping images files...')
    shutil.copytree(original_images, output_images)

    print('Coping CCAgT annotations file...')
    shutil.copy(original_annotations, output_annotations)


def create_subdataset(name: str,
                      original_dir: str,
                      output_base: str,
                      slice_images: tuple[int, ...] | None,
                      extract: set[int] | None,
                      categories_to_keep: set[int] | None,
                      check_if_all_have_one: set[int] | None,
                      delete: bool,
                      generate_masks: bool,
                      CCAgT_path: str | None,
                      paddings: str,
                      extensions: tuple[str, ...] = ('.jpg', '.png')) -> int:
    output_dir = os.path.join(output_base, name)

    output_images_dir = os.path.join(output_dir, STRUCTURE['i'])
    output_masks_dir = os.path.join(output_dir, STRUCTURE['m'])
    output_annotations_path = os.path.join(output_dir, STRUCTURE['l'])

    if os.path.isdir(output_images_dir) or os.path.isdir(output_masks_dir):
        print(f'Already exist a dataset with name={name} at {output_base}!', file=sys.stderr)
        return 1

    if CCAgT_path is None:
        CCAgT_path = os.path.join(original_dir, STRUCTURE['l'])

    input_images_dir = os.path.join(original_dir, STRUCTURE['i'])
    input_masks_dir = os.path.join(original_dir, STRUCTURE['m'])

    if not os.path.isfile(CCAgT_path):
        print(f'Not found the annotations file at `{CCAgT_path}`!', file=sys.stderr)
        return 1

    if not os.path.isdir(input_images_dir):
        print(f'Not found the original data at `{input_images_dir}`!', file=sys.stderr)
        return 1
    # elif not os.path.isdir(input_masks_dir):
    #     print(f'Not found the original data masks at `{input_masks_dir}`!', file=sys.stderr)
    #     return 1

    print('\n\n------------------------')
    if slice_images is not None:
        # TODO
        slice_images = tuple(slice_images)
        print(f'Create images and masks splitting then into {slice_images} (horizontal, vertical) parts')
        slice.images_and_masks(input_images_dir,
                               input_masks_dir,
                               output_dir,
                               slice_images[0], slice_images[1],
                               extension=extensions,
                               look_recursive=True)
    elif extract is not None:
        # TODO
        if CCAgT_path is not None:
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
        copy_original_dataset(input_images_dir, CCAgT_path, output_images_dir, output_annotations_path)

    ccagt_annotations = read_parquet(output_annotations_path)

    if categories_to_keep is not None:
        categories_to_keep = set(categories_to_keep)
        print('------------------------')
        print(f'Delete annotations that with categories different from {categories_to_keep} ')
        ccagt_annotations.df = ccagt_annotations.df[ccagt_annotations.df['category_id'].isin(categories_to_keep)]

    if check_if_all_have_one is not None:
        print('------------------------')
        check_if_all_have_one = set(check_if_all_have_one)
        if isinstance(categories_to_keep, set):
            cats_to_have = check_if_all_have_one.union(categories_to_keep)

        images_names = set(ccagt_annotations.df['image_name'].unique())
        images_names_filtered = set(ccagt_annotations.df.loc[ccagt_annotations.df['category_id'].isin(cats_to_have),
                                                             'image_name'].unique())

        diff = images_names.difference(images_names_filtered)
        if len(diff) > 0:
            print(f'A total of {len(diff)} files there is not at least one of the categories {cats_to_have}',
                  file=sys.stderr)

            if delete:
                all_images = {basename(k): v for k, v in find_files(output_images_dir, extensions, True).items()}
                print('Deleting images that do not have at least one of the selected categories...')
                for bn in diff:
                    os.remove(all_images[bn])
                print('Deleting this images from the annotations...')
                ccagt_annotations.df = ccagt_annotations.df[ccagt_annotations.df['category_id'].isin(cats_to_have)]
        else:
            print(f'Everything is ok, having 0 files that do not have any of the categories {check_if_all_have_one}')

    print('------------------------')
    print('Checking if have any image without annotation...')
    all_images = {basename(k): v for k, v in find_files(output_images_dir, extensions, True).items()}
    images_with_annotation = set(ccagt_annotations.df['image_name'].unique())
    diff = set(all_images.keys()).difference(images_with_annotation)
    if len(diff) > 0:
        print(f'Deleting total of {len(diff)} images without annotation...')
        for bn in diff:
            os.remove(all_images[bn])

    print('------------------------')
    print(f'Saving the annotations to {output_annotations_path}...')
    ccagt_annotations.to_parquet(output_annotations_path)

    if generate_masks:
        print('------------------------')
        print('Generating masks for semantic segmentation...')
        ccagt_generate_masks(output_annotations_path, output_masks_dir, True)

    print('\n------------------------')
    print('Creation of the subdataset finished!')
    describe.dataset(output_annotations_path, CategoriesInfos(), output_dir, extensions)
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
                                 args.generate_masks,
                                 args.labels,
                                 args.paddings,
                                 tuple(args.extensions))

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
