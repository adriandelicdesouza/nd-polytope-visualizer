from __future__ import annotations

import numpy as np

from .sensitivity import SensitivityData
from .slicing import Geometry
from dataclasses import dataclass

@dataclass(frozen=True)
class ResponseGeometry:
    """Geometry representing an N-dimensional model response."""

    geometry: Geometry
    outputs: np.ndarray

def sensitivity_vertices(
    data: SensitivityData,
) -> Geometry:
    """Convert sensitivity parameter combinations into geometry vertices."""
    grids = np.meshgrid(
        *data.axes,
        indexing="ij",
    )

    vertices = np.stack(
        grids,
        axis=-1,
    ).reshape(-1, data.dimensions)

    return Geometry(
        vertices=vertices,
        edges=sensitivity_edges(data),
    )

def sensitivity_edges(
    data: SensitivityData,
) -> np.ndarray:
    """Return edges connecting neighboring parameter combinations."""
    shape = data.values.shape
    dimensions = data.dimensions

    edges: list[tuple[int, int]] = []

    for index in np.ndindex(shape):
        current = np.ravel_multi_index(index, shape)

        for axis in range(dimensions):
            if index[axis] + 1 >= shape[axis]:
                continue

            neighbor = list(index)
            neighbor[axis] += 1

            neighbor_flat = np.ravel_multi_index(
                tuple(neighbor),
                shape,
            )

            edges.append((current, neighbor_flat))

    return np.asarray(edges, dtype=int).reshape(-1, 2)

def sensitivity_outputs(
    data: SensitivityData,
) -> np.ndarray:
    """Return model outputs aligned with sensitivity geometry vertices."""
    return data.values.reshape(-1).copy()

def build_response_geometry(
    data: SensitivityData,
) -> ResponseGeometry:
    """Build geometry and aligned model outputs from sensitivity data."""
    geometry = sensitivity_vertices(data)
    outputs = sensitivity_outputs(data)

    return ResponseGeometry(
        geometry=geometry,
        outputs=outputs,
    )