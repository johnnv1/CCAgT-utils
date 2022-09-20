from __future__ import annotations

import argparse
from typing import Sequence


def converter_command_parser(
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser] | None = None,
) -> argparse.ArgumentParser:

    if subparsers is not None:
        parser = subparsers.add_parser('convert')
    else:
        parser = argparse.ArgumentParser(
            prog='CCAgT-utils dataset converter command',
        )

    group_output = parser.add_argument_group(
        'Define the output format desired.',
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

    parser.add_argument(
        '--target',
        help='Define the target of the COCO format.',
        choices=[
            'object-detection', 'OD', 'panoptic-segmentation',
            'PS', 'instance-segmentation', 'IS',
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
        parser.error('--to-ccagt requires --target')


def converter_command(
    args: argparse.Namespace | None = None,
) -> int:
    # TODO:
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    parser = converter_command_parser()
    args = parser.parse_args(argv)
    return converter_command(args)


if __name__ == '__main__':
    raise SystemExit(main())
