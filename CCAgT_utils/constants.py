from __future__ import annotations

import sys

if sys.version_info >= (3, 8):  # pragma: >=3.8 cover
    import importlib.metadata as importlib_metadata
else:  # pragma: <3.8 cover
    import importlib_metadata


VERSION = importlib_metadata.version('CCAgT_utils')

FILENAME_SEP = '_'

STRUCTURE = {'i': 'images/', 'm': 'masks/', 'l': 'CCAgT.parquet.gzip'}
