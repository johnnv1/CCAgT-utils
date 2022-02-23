from __future__ import annotations


def is_2d(shape: tuple[int]) -> bool:
    """Verify if the shape is at a 2D shape expected (n X m)
    where n and m can be any integer that represents height (rows) and
    width (columns) size.

    Parameters
    ----------
    shape : tuple
        A tuple of the size for each axis. In general from
        `np.ndarray.shape`

    Returns
    -------
    bool
        True if the shape matches with (n X m) - just have 2 axis, any
        other else False
    """
    if len(shape) == 2:
        return True
    return False


def is_rgb_shape(shape: tuple[int]) -> bool:
    """Verify if the shape is at a RGB shape expected (n X m x 3)
    where n and m can be any integer that represents height (rows) and
    width (columns) size.

    Parameters
    ----------
    shape : tuple
        A tuple of the size for each axis. In general from
        `np.ndarray.shape`

    Returns
    -------
    bool
        True if the shape matches with (n X m x 3), any other else False
    """
    if len(shape) == 3:
        if shape[-1] == 3:
            return True
    return False
