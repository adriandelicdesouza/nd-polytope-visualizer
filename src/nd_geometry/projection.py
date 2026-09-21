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

def linear_project(
    geometry: Geometry,
    matrix: np.ndarray,
) -> Geometry:
    """Project geometry using a linear N-dimensional → M-dimensional matrix.

    Parameters
    ----------
    geometry:
        Source geometry.
    matrix:
        Projection matrix with shape (target_dimensions, source_dimensions).
    """
    vertices = np.asarray(geometry.vertices, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    if vertices.ndim != 2:
        raise ValueError("geometry vertices must be a 2D array")

    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")

    source_dimensions = vertices.shape[1]

    if matrix.shape[1] != source_dimensions:
        raise ValueError(
            "matrix column count must equal source dimensions"
        )

    projected_vertices = vertices @ matrix.T

    return Geometry(
        vertices=projected_vertices,
        edges=geometry.edges.copy(),
    )

def orthogonal_projection_matrix(
    source_dimensions: int,
    target_dimensions: int,
    seed: int | None = None,
) -> np.ndarray:
    """Create an orthonormal projection matrix.

    Parameters
    ----------
    source_dimensions:
        Number of dimensions in the source space.
    target_dimensions:
        Number of dimensions in the projected space.
    seed:
        Optional random seed for reproducible orientations.
    """
    if source_dimensions < 1:
        raise ValueError("source_dimensions must be >= 1")

    if target_dimensions < 1:
        raise ValueError("target_dimensions must be >= 1")

    if target_dimensions > source_dimensions:
        raise ValueError(
            "target_dimensions cannot exceed source dimensions"
        )

    rng = np.random.default_rng(seed)

    basis = rng.normal(
        size=(source_dimensions, target_dimensions),
    )

    q, _ = np.linalg.qr(basis)

    return q[:, :target_dimensions].T

def rotate_projection_basis(
    matrix: np.ndarray,
    axis_a: int,
    axis_b: int,
    angle: float,
) -> np.ndarray:
    """Rotate a projection basis in a source-space coordinate plane."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")

    source_dimensions = matrix.shape[1]

    if not 0 <= axis_a < source_dimensions:
        raise ValueError(
            f"axis_a must be between 0 and {source_dimensions - 1}"
        )

    if not 0 <= axis_b < source_dimensions:
        raise ValueError(
            f"axis_b must be between 0 and {source_dimensions - 1}"
        )

    if axis_a == axis_b:
        raise ValueError("axis_a and axis_b must be different")

    cosine = np.cos(angle)
    sine = np.sin(angle)

    rotation = np.eye(source_dimensions)

    rotation[axis_a, axis_a] = cosine
    rotation[axis_a, axis_b] = -sine
    rotation[axis_b, axis_a] = sine
    rotation[axis_b, axis_b] = cosine

    return matrix @ rotation.T