from __future__ import annotations

import os


def basename(filename: str, with_extension: bool = False) -> str:
    """From a full filename get the basename with or not with the
    extension.

    Parameters
    ----------
    filename : str
        A full filename
    with_extension : bool, optional
        Flag to return the basename with extension, if True return
        the basename with the file extension, else will return just the
        basename, by default False

    Returns
    -------
    str
        The basename of the <filename> with or not the file extension
    """
    bn = os.path.basename(filename)
    if with_extension:
        return bn
    else:
        return os.path.splitext(bn)[0]
