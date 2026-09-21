from __future__ import annotations

import numpy as np

from .slicing import Geometry


def project(
    geometry: Geometry,
    target_dimensions: int,
    axes: tuple[int, ...] | None = None,
) -> Geometry:
    """Project N-dimensional geometry into M dimensions.

    This initial implementation performs coordinate projection by
    retaining selected source axes.

    Topology is preserved because projection changes coordinates,
    not the relationships between the original vertices.
    """
    vertices = np.asarray(geometry.vertices, dtype=float)

    if vertices.ndim != 2:
        raise ValueError("geometry vertices must be a 2D array")

    source_dimensions = vertices.shape[1]

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

    projected_vertices = vertices[:, axes].copy()

    return Geometry(
        vertices=projected_vertices,
        edges=geometry.edges.copy(),
    )