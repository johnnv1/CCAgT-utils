from __future__ import annotations

import pytest

from CCAgT_utils.commands.main import main


def test_main(
    lbox_sample_complete,
    aux_path,
    tmpdir,
):

    filename = 'CCAgT_out.parquet.gzip'
    out = main([
        'convert', '--to-ccagt', '-i', lbox_sample_complete, '-a',
        aux_path, '-o', str(tmpdir), '-f', filename,
    ])
    assert out == 0


def test_main_help():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main(['help'])
    assert pytest_wrapped_e1.value.code == 0

    with pytest.raises(SystemExit) as pytest_wrapped_e2:
        main([])
    assert pytest_wrapped_e2.value.code == 0


def test_main_help_other_command():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main(['help', 'convert'])

    assert pytest_wrapped_e1.value.code == 0


def test_main_wrong_command():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main(['wrong-command'])

    assert pytest_wrapped_e1.value.code == 2
