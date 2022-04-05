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
from CCAgT_utils.categories import read_json
from CCAgT_utils.constants import STRUCTURE
from CCAgT_utils.constants import VERSION
from CCAgT_utils.converters.CCAgT import read_parquet
from CCAgT_utils.converters.utils import ccagt_generate_masks
from CCAgT_utils.prepare import ccagt_dataset
from CCAgT_utils.prepare import extract_image_and_annotations_by_category
from CCAgT_utils.utils import basename
from CCAgT_utils.utils import find_files


def _add_create_subdataset_options(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        '--name',
        metavar='SUBDATASET_NAME',
        required=True,
        help='The name of the subdataset that will be generated',
    )

    parser.add_argument(
        '--original',
        metavar='PATH_FOR_ORIGINAL_DATASET',
        required=True,
        help=(
            'Path for the original dataset. It is expected that this directory has the subdirectories '
            '`images/` and `masks/`, and `CCAgT.parquet.gzip`'
        ),
    )
    parser.add_argument(
        '--output',
        metavar='PATH_TO_WRITE_THE_SUBDATASETS',
        required=True,
        help=(
            'Path to write the new subdataset. It will create a directory with this subdataset name, and '
            'subdirectories `images/` and `masks/`'
        ),
    )

    group_edit_image = parser.add_argument_group(
        'Define the type of images wanted at the end. If none of these be chosen,'
        ' will just copy the original data!',
    )
    group_ex = group_edit_image.add_mutually_exclusive_group(required=False)
    group_ex.add_argument(
        '--slice-images',
        nargs=2,
        metavar='HORIZONTA_VERTICAL',
        type=int,
        help=(
            'Define that wants slice the images into smaller parts. Needs to pass the amount of slice '
            'desired for the horizontal and vertical split of the images.'
        ),
    )

    group_ex.add_argument(
        '--extract',
        nargs='*',
        type=int,
        metavar='CATEGORY_ID',
        help=(
            'Define that wants extract based on one category. Will generate one image for each '
            'instance of the desired category.'
        ),
    )

    group_extract = parser.add_argument_group('Arguments to use with `--extract` option')
    group_extract.add_argument(
        '--paddings',
        default=0,
        help=(
            'In percent (float values) or pixels (integer values) select, the size of paddings to '
            'apply. Just works with --extract'
        ),
    )

    group_clean = parser.add_argument_group('To clear the images and/or annotations for the subdataset')
    group_ex_clean = group_clean.add_mutually_exclusive_group(required=False)
    group_ex_clean.add_argument(
        '--remove-images-without',
        nargs='*',
        type=int,
        metavar='CATEGORY_ID',
        help=(
            'Define that you wants remove of this subdataset the images that does not have the '
            'categories passed as argument.'
        ),
    )

    group_ex_clean.add_argument(
        '--remove-annotations-different',
        nargs='*',
        type=int,
        metavar='CATEGORY_ID',
        help=(
            'Define that you wants remove of this subdataset the annotations that have different '
            'categories then the passed as argument.'
        ),
    )

    check_group = parser.add_argument_group('Process of checking the annotation and images.')
    check_group.add_argument(
        '--check-if-all-have-at-least-one-of',
        nargs='*',
        metavar='CATEGORY_ID',
        type=int,
        help=(
            'Define that you wants check if all images have at least one of the categories passed as '
            'argument. Will print a information about.'
        ),
    )
    check_group.add_argument(
        '--no-check-images',
        action='store_false',
        help=(
            'Will not check if all images at the new dataset have at least one annotation. By default '
            'the process will check if all images have at least one annotation.'
        ),
    )
    check_group.add_argument(
        '--delete',
        action='store_true',
        help='Will delete images without annotation, or without the chosen categories',
    )

    group_kwargs = parser.add_argument_group('Extra arguments if desired')
    group_kwargs.add_argument(
        '--generate-masks',
        action='store_true',
        help='To generate the masks for semantic segmentation based on the new annotations',
    )
    group_kwargs.add_argument(
        '--labels',
        help=(
            'Path for the CCAgT file with the labels. By default will look at '
            '`PATH_FOR_ORIGINAL_DATASET/CCAgT.parquet.gzip`'
        ),
    )
    group_kwargs.add_argument(
        '--aux-file',
        help='Path for the CCAgT auxiliary file, by default will use default Categories Infomation!',
    )
    group_kwargs.add_argument(
        '--extensions',
        nargs='*',
        default=('.jpg', '.png'),
        help='The extensions to look for when search the images and masks.',
    )
    group_kwargs.add_argument(
        '--overwrite',
        action='store_true',
        help='With this option if already exist a dataset if this name will overwrite this!',
    )


def create_subdataset(
    *,
    name: str,
    original_dir: str,
    output_base: str,
    slice_images: tuple[int, ...] | None,
    extract: set[int] | None,
    categories_to_keep: tuple[int, set[int]] | None,
    categories_to_check: set[int] | None,
    delete: bool,
    generate_masks: bool,
    CCAgT_path: str | None,
    paddings: str,
    check_if_images_have_annotations: bool,
    extensions: tuple[str, ...],
    aux_file_path: str | None,
    overwrite: bool,
) -> int:

    output_dir = os.path.join(output_base, name)

    output_images_dir = os.path.join(output_dir, STRUCTURE['i'])
    output_masks_dir = os.path.join(output_dir, STRUCTURE['m'])
    output_annotations_path = os.path.join(output_dir, STRUCTURE['l'])

    if overwrite and os.path.isdir(output_dir):
        print(f'Deleting all outdated data from `{output_dir}`...')
        shutil.rmtree(output_dir)
    elif (os.path.isdir(output_images_dir) or os.path.isdir(output_masks_dir)):
        print(f'Already exist a dataset with name={name} at {output_base}!', file=sys.stderr)
        return 1

    if CCAgT_path is None:
        CCAgT_path = os.path.join(original_dir, STRUCTURE['l'])

    input_images_dir = os.path.join(original_dir, STRUCTURE['i'])

    if not os.path.isfile(CCAgT_path):
        print(f'Not found the annotations file at `{CCAgT_path}`!', file=sys.stderr)
        return 1

    if not os.path.isdir(input_images_dir):
        print(f'Not found the original data at `{input_images_dir}`!', file=sys.stderr)
        return 1

    if isinstance(aux_file_path, str):
        categories_infos = read_json(aux_file_path)
    else:
        categories_infos = CategoriesInfos()

    print('------------------------')
    print(f'Loading the original annotations file from `{CCAgT_path}`...')
    ccagt_annotations = read_parquet(CCAgT_path)

    if isinstance(categories_to_keep, tuple):
        _choice_to_delete, _cats_to_keep = categories_to_keep

        if _choice_to_delete == 0:
            # --remove-images-without
            print(f'Delete images where not have at least one annotation with the categories: {_cats_to_keep}')
            _idx_with_categories = ccagt_annotations.df['category_id'].isin(_cats_to_keep)
            images_with_categories = set(ccagt_annotations.df.loc[_idx_with_categories, 'image_id'].unique())
            ccagt_annotations.df = ccagt_annotations.df[ccagt_annotations.df['image_id'].isin(images_with_categories)]

        elif _choice_to_delete == 1:
            # --remove-annotations-without
            print(f'Delete annotations where the categories is not in: {_cats_to_keep} ')
            ccagt_annotations.df = ccagt_annotations.df[ccagt_annotations.df['category_id'].isin(_cats_to_keep)]
        else:
            print('Unexpected choice for the type of removal proccess.', file=sys.stderr)
            return 1
    else:
        print('No process of remove chosen, just skiped.')

    if ccagt_annotations.df.shape[0] == 0:
        print('The annotations file has none annotation, just finishing the process!', file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving the annotations to `{output_annotations_path}`...')
    ccagt_annotations.to_parquet(output_annotations_path)

    print('------------------------')
    if isinstance(slice_images, tuple):
        # --slice-images
        print(f'Generate images and annotations splitting the originals into {slice_images} (horizontal, vertical) parts...')
        slice.images_and_annotations(
            input_images_dir,
            output_annotations_path,
            output_dir,
            output_annotations_path,
            slice_images[0], slice_images[1],
            extension=extensions,
            look_recursive=True,
        )
    elif extract is not None:
        # --extract
        print(f'Create images and annotations for each instance of the categories {extract}')
        print('  > If have the categories `Nucleus` or `Overlapped nuclei` will also keep the NORs annotations.')
        extract_image_and_annotations_by_category(
            input_images_dir,
            output_dir,
            extract,
            output_annotations_path,
            ast.literal_eval(paddings),
            extensions,
            True,
        )
    else:
        # if not choice between --slice-images and --extract, will just copy
        # TODO: copy with multiprocess
        print('The images and masks of the subdataset will be copied from the original dataset!')
        print('Coping images files...')
        shutil.copytree(input_images_dir, output_images_dir)

    print('------------------------')
    print(f'Loading the annotations file from `{output_annotations_path}`...')
    ccagt_annotations = read_parquet(output_annotations_path)

    print('------------------------')
    ccagt_annotations = ccagt_dataset(ccagt_annotations, categories_infos, do_fit_geometries=False)

    print('------------------------')
    images_without_the_categories: set[str] = set({})
    if isinstance(categories_to_check, set):
        # --check-if-all-have-at-least-one-of
        print(f'Checking all images have at least one of the categories {categories_to_check}...')
        images_names = set(ccagt_annotations.df['image_name'].unique())
        images_names_filtered = set(
            ccagt_annotations.df.loc[
                ccagt_annotations.df['category_id'].isin(categories_to_check),
                'image_name',
            ].unique(),
        )

        images_without_the_categories = images_names.difference(images_names_filtered)
        if len(images_without_the_categories) > 0:
            print(
                (
                    f'A total of {len(images_without_the_categories)} files there is not at least one of the categories '
                    f'{categories_to_check}'
                ),
                file=sys.stderr,
            )
        else:
            print(f'Everything is ok, have 0 files that do not have at least one of the categories {categories_to_check}')

    images_without_the_annotations: set[str] = set({})
    if check_if_images_have_annotations:
        # --no-check-images to skip this
        print('Checking if have any image without annotation...')
        all_images = {basename(k): v for k, v in find_files(output_images_dir, extensions, True).items()}
        images_with_annotation = set(ccagt_annotations.df['image_name'].unique())
        images_without_the_annotations = set(all_images.keys()).difference(images_with_annotation)

        if len(images_without_the_annotations) > 0:
            print(
                (f'A total of {len(images_without_the_annotations)} files there is not at least one annotation'),
                file=sys.stderr,
            )
        else:
            print(f'Everything is ok, have 0 files that do not have any annotation {categories_to_check}')

    if delete and (len(images_without_the_annotations) > 0 or len(images_without_the_categories) > 0):
        # --delete
        if len(images_without_the_categories) > 0:
            print(f'Will delete images that do not have at least one of the categories {categories_to_check}')
            ccagt_annotations.df = ccagt_annotations.df[
                ~ccagt_annotations.df['image_name'].isin(images_without_the_categories)
            ]
        if len(images_without_the_annotations) > 0:
            print('Will delete images that do not have at least one annotation')
            ccagt_annotations.df = ccagt_annotations.df[
                ~ccagt_annotations.df['image_name'].isin(images_without_the_annotations)
            ]
        basenames_to_delete = set(images_without_the_categories).union(set(images_without_the_annotations))

        print(f'Finding the images to delete at `{output_images_dir}`...')
        all_images = {basename(k): v for k, v in find_files(output_images_dir, extensions, True).items()}

        print(f'Deleting a total of {len(basenames_to_delete)} images...')
        for bn in basenames_to_delete:
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
    describe.dataset(output_annotations_path, categories_infos, output_dir, extensions)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog='CCAgT_converter')

    # https://stackoverflow.com/a/8521644/812183
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {VERSION}')

    subparsers = parser.add_subparsers(dest='command')

    create_subdataset_parser = subparsers.add_parser(
        'create-subdataset',
        help=(
            'Based on the original dataset create a personalized version of '
            'the dataset with the desired modifications. Examples: slice the '
            'images into smaller parts, select just images with some category,'
            ' images of a specific category.'
        ),
    )
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

        categories_to_keep = None
        if args.remove_images_without is not None:
            categories_to_keep = (0, set(args.remove_images_without))
        elif args.remove_annotations_different is not None:
            categories_to_keep = (1, set(args.remove_annotations_different))

        categories_to_check = None
        if args.check_if_all_have_at_least_one_of is not None:
            categories_to_check = set(args.check_if_all_have_at_least_one_of)

        slice_images = None if args.slice_images is None else tuple(args.slice_images)
        extract = None if args.extract is None else set(args.extract)

        return create_subdataset(
            name=str(args.name),
            original_dir=str(args.original),
            output_base=str(args.output),
            slice_images=slice_images,
            extract=extract,
            categories_to_keep=categories_to_keep,
            categories_to_check=categories_to_check,
            delete=args.delete,
            generate_masks=args.generate_masks,
            CCAgT_path=args.labels,
            paddings=args.paddings,
            check_if_images_have_annotations=args.no_check_images,
            extensions=tuple(args.extensions),
            aux_file_path=args.aux_file,
            overwrite=args.overwrite,
        )

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
