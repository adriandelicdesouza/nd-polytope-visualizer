from __future__ import annotations

import numpy as np


def rotate(
    points: np.ndarray,
    axis_a: int,
    axis_b: int,
    angle: float,
) -> np.ndarray:
    """Rotate points in the plane defined by two coordinate axes.

    Parameters
    ----------
    points:
        Array of shape (n_points, dimensions).
    axis_a:
        First coordinate axis.
    axis_b:
        Second coordinate axis.
    angle:
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotated points with the same shape as the input.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array")

    dimensions = points.shape[1]

    if not 0 <= axis_a < dimensions:
        raise ValueError(f"axis_a must be between 0 and {dimensions - 1}")

    if not 0 <= axis_b < dimensions:
        raise ValueError(f"axis_b must be between 0 and {dimensions - 1}")

    if axis_a == axis_b:
        raise ValueError("axis_a and axis_b must be different")

    cosine = np.cos(angle)
    sine = np.sin(angle)

    rotation = np.eye(dimensions)

    rotation[axis_a, axis_a] = cosine
    rotation[axis_a, axis_b] = -sine
    rotation[axis_b, axis_a] = sine
    rotation[axis_b, axis_b] = cosine

    return points @ rotation.T
