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

def normalize_outputs(
    outputs: np.ndarray,
) -> np.ndarray:
    """Normalize output values to the range [0, 1]."""
    outputs = np.asarray(outputs, dtype=float)

    if outputs.ndim != 1:
        raise ValueError("outputs must be a 1D array")

    minimum = np.min(outputs)
    maximum = np.max(outputs)

    if np.isclose(minimum, maximum):
        return np.zeros_like(outputs)

    return (outputs - minimum) / (maximum - minimum)

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

def embed_outputs(
    response: ResponseGeometry,
    normalize: bool = False,
) -> Geometry:
    """Embed model outputs as an additional geometric coordinate."""
    outputs = (
        response.normalized_outputs
        if normalize
        else response.outputs
    )

    vertices = np.column_stack(
        (response.geometry.vertices, outputs)
    )

    return Geometry(
        vertices=vertices,
        edges=response.geometry.edges.copy(),
    )

@dataclass(frozen=True)
class ResponseGeometry:
    """Geometry representing an N-dimensional model response."""

    geometry: Geometry
    outputs: np.ndarray

    @property
    def normalized_outputs(self) -> np.ndarray:
        """Return outputs normalized to the range [0, 1]."""
        return normalize_outputs(self.outputs)

    @property
    def output_range(self) -> tuple[float, float]:
        """Return the minimum and maximum model outputs."""
        return (
            float(np.min(self.outputs)),
            float(np.max(self.outputs)),
        )

    def output_at(self, index: int) -> float:
        """Return the model output associated with a geometry vertex."""
        if not 0 <= index < len(self.outputs):
            raise IndexError("output index out of range")

        return float(self.outputs[index])   

def embed_outputs(
    response: ResponseGeometry,
    normalize: bool = False,
    scale: float = 1.0,
) -> Geometry:
    """Embed model outputs as an additional geometric coordinate."""
    if scale <= 0:
        raise ValueError("scale must be > 0")

    outputs = (
        response.normalized_outputs
        if normalize
        else response.outputs
    )

    vertices = np.column_stack(
        (
            response.geometry.vertices,
            outputs * scale,
        )
    )

    return Geometry(
        vertices=vertices,
        edges=response.geometry.edges.copy(),
    )