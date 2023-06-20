from __future__ import annotations

import importlib.metadata as importlib_metadata


VERSION = importlib_metadata.version('CCAgT_utils')

FILENAME_SEP = '_'

STRUCTURE = {'i': 'images/', 'm': 'masks/', 'l': 'CCAgT.parquet.gzip'}
