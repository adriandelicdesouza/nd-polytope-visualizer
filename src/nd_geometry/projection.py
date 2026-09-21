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

def deduplicate_geometry(
    geometry: Geometry,
    epsilon: float = 1e-9,
) -> Geometry:
    """Merge coincident vertices and remap their edges."""

    vertices = np.asarray(geometry.vertices, dtype=float)
    edges = np.asarray(geometry.edges, dtype=int)

    if vertices.ndim != 2:
        raise ValueError("geometry vertices must be a 2D array")

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("geometry edges must have shape (n_edges, 2)")

    unique_vertices: list[np.ndarray] = []
    vertex_map: dict[int, int] = {}

    for old_index, vertex in enumerate(vertices):
        new_index = None

        for candidate_index, candidate in enumerate(unique_vertices):
            if np.allclose(
                vertex,
                candidate,
                atol=epsilon,
                rtol=0,
            ):
                new_index = candidate_index
                break

        if new_index is None:
            new_index = len(unique_vertices)
            unique_vertices.append(vertex.copy())

        vertex_map[old_index] = new_index

    remapped_edges: set[tuple[int, int]] = set()

    for a, b in edges:
        new_a = vertex_map[a]
        new_b = vertex_map[b]

        if new_a == new_b:
            continue

        remapped_edges.add(
            (min(new_a, new_b), max(new_a, new_b))
        )

    return Geometry(
        vertices=np.asarray(unique_vertices, dtype=float),
        edges=np.asarray(
            sorted(remapped_edges),
            dtype=int,
        ).reshape(-1, 2),
    )