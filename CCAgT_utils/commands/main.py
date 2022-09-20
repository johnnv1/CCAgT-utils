from __future__ import annotations

import argparse
import sys
from typing import Sequence

from CCAgT_utils.commands.converter import converter_command_parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='CCAgT utils CLI tools',
        usage='CCAgT-utils <command> [<args>]',
        description=(
            'This tool will help you to convert CCAgT between different '
            'datasets formats, to generate sub-datasets and in '
            'the visualization of the dataset.'
        ),
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='CCAgT-utils command helpers',
    )

    # Register commands
    converter_command_parser(subparsers)

    help = subparsers.add_parser(
        'help',
        help='Show help for a specific command.',
    )
    help.add_argument(
        'help_cmd',
        nargs='?',
        help='Command to show help for.',
    )

    if len(argv) == 0:
        argv = ['help']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    # Run
    if hasattr(args, 'checker'):
        args.checker(parser, args)

    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
