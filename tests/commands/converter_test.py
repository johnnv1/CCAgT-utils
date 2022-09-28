from __future__ import annotations

import pytest

from CCAgT_utils.base.errors import FileTypeError
from CCAgT_utils.commands.converter import check_arguments
from CCAgT_utils.commands.converter import converter_command
from CCAgT_utils.commands.converter import converter_command_parser
from CCAgT_utils.commands.converter import main
from CCAgT_utils.commands.converter import to_ccagt
from CCAgT_utils.commands.converter import to_coco
from CCAgT_utils.commands.converter import to_masks


def test_converter_command_parser(subparser, converter_required):
    parser_converter = converter_command_parser(subparser)
    args = parser_converter.parse_args(converter_required)

    assert args.to_coco is False
    assert args.to_ccagt
    assert len(args.in_path) > 0
    assert len(args.out_path) > 0
    assert hasattr(args, 'func')
    assert hasattr(args, 'checker')
    assert parser_converter.prog == 'CCAgT utils tests convert'

    parser_converter = converter_command_parser()
    assert parser_converter.prog == 'CCAgT-utils dataset converter command'


def test_check_arguments(subparser, converter_required):
    parser_converter = converter_command_parser(subparser)
    with pytest.raises(SystemExit):
        check_arguments(
            parser_converter,
            parser_converter.parse_args(converter_required),
        )

    out = check_arguments(
        parser_converter,
        parser_converter.parse_args(converter_required + ['-a', '']),
    )

    assert out is None

    converter_req = converter_required[1:] + ['--to-coco']
    args = parser_converter.parse_args(converter_req)
    with pytest.raises(SystemExit):
        check_arguments(parser_converter, args)


def test_to_ccagt(lbox_sample_complete, lbox_aux_path, tmpdir):
    filename = tmpdir.join('CCAgT_out.parquet.gzip')

    out = to_ccagt(
        lbox_sample_complete,
        str(filename),
        lbox_aux_path,
        True,
    )

    assert out == 0
    assert filename.isfile()

    filename = tmpdir.join('CCAgT_out2.parquet.gzip')

    out = to_ccagt(
        lbox_sample_complete,
        str(filename),
        lbox_aux_path,
        False,
    )

    assert out == 0
    assert filename.isfile()


def test_to_ccagt_wrong_outfilename():
    with pytest.raises(FileTypeError):
        to_ccagt('', '', '', False)


def test_to_coco():
    with pytest.raises(NotImplementedError):
        to_coco()


def test_to_masks(ccagt_df_single_nucleus_complete_path, tmpdir):

    outdir = tmpdir.mkdir('output_A/')
    to_masks(ccagt_df_single_nucleus_complete_path, str(outdir), False)

    assert len(outdir.listdir()) == 1

    outdir = tmpdir.mkdir('output_B/')
    to_masks(ccagt_df_single_nucleus_complete_path, str(outdir), True)

    assert outdir.listdir()[0].isdir()
    assert len(outdir.listdir()[0].listdir()) == 1


def test_converter_command(
    subparser,
    lbox_sample_complete,
    lbox_aux_path,
    ccagt_df_single_nucleus_complete_path,
    tmpdir,
):
    assert converter_command(None) == 1

    parser = converter_command_parser(subparser)
    filename = 'CCAgT_out.parquet.gzip'
    out_path = tmpdir.join(filename)
    args = parser.parse_args([
        '--to-ccagt', '-i', lbox_sample_complete, '-a',
        lbox_aux_path, '-o', str(tmpdir), '-f', filename,
    ])
    out = converter_command(args)
    assert out == 0
    assert out_path.isfile()

    with pytest.raises(NotImplementedError):
        converter_command(parser.parse_args(['--to-coco', '-i', '', '-o', '']))

    outdir = tmpdir.mkdir('output_A/')
    args = parser.parse_args([
        '--to-masks', '-i', ccagt_df_single_nucleus_complete_path, '-o',
        str(outdir),
    ])
    out = converter_command(args)
    assert out == 0
    assert len(outdir.listdir()) == 1


def test_main(
    lbox_sample_complete,
    lbox_aux_path,
    tmpdir,
):

    filename = 'CCAgT_out.parquet.gzip'
    out = main([
        '--to-ccagt', '-i', lbox_sample_complete, '-a', lbox_aux_path,
        '-o', str(tmpdir), '-f', filename,
    ])
    assert out == 0
