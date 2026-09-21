from __future__ import annotations

import numpy as np


def project(
    points: np.ndarray,
    target_dimensions: int,
    axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Project N-dimensional points into M dimensions.

    Parameters
    ----------
    points:
        Array with shape (n_points, source_dimensions).
    target_dimensions:
        Number of dimensions in the projected result.
    axes:
        Optional source axes to retain. If omitted, the first
        target_dimensions axes are retained.

    Returns
    -------
    np.ndarray
        Projected points with shape (n_points, target_dimensions).
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array")

    source_dimensions = points.shape[1]

    if target_dimensions < 1:
        raise ValueError("target_dimensions must be >= 1")

    if target_dimensions > source_dimensions:
        raise ValueError(
            "target_dimensions cannot exceed source dimensions"
        )

    if axes is None:
        axes = tuple(range(target_dimensions))

    if len(axes) != target_dimensions:
        raise ValueError(
            "number of axes must equal target_dimensions"
        )

    if len(set(axes)) != len(axes):
        raise ValueError("axes must be unique")

    if any(axis < 0 or axis >= source_dimensions for axis in axes):
        raise ValueError("axes contain an invalid source dimension")

    return points[:, axes].copy()
