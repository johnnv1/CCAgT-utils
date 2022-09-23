from __future__ import annotations

import argparse

import pytest


@pytest.fixture
def parser():
    return argparse.ArgumentParser(prog='CCAgT utils tests')


@pytest.fixture
def subparser(parser):
    return parser.add_subparsers(dest='command')


@pytest.fixture
def converter_required():
    return ['--to-ccagt', '-i', 'in-path', '-o', 'out-path']
