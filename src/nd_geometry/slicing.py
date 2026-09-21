from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-9


@dataclass(frozen=True)
class Geometry:
    """Geometric points and their topology."""

    vertices: np.ndarray
    edges: np.ndarray


def slice_geometry(
    geometry: Geometry,
    axis: int,
    value: float,
    epsilon: float = EPSILON,
) -> Geometry:
    """Slice geometry with the hyperplane x_axis = value.

    The specified coordinate is removed from the resulting vertices,
    reducing dimensionality by one.
    """
    vertices = np.asarray(geometry.vertices, dtype=float)
    edges = np.asarray(geometry.edges, dtype=int)

    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D array")

    dimensions = vertices.shape[1]

    if dimensions < 2:
        raise ValueError("geometry must have at least 2 dimensions")

    if not 0 <= axis < dimensions:
        raise ValueError(
            f"axis must be between 0 and {dimensions - 1}"
        )

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")

    result_vertices: list[np.ndarray] = []
    result_edges: list[tuple[int, int]] = []

    # Map original vertex indices to indices in the sliced geometry.
    vertex_map: dict[int, int] = {}

    def add_vertex(point: np.ndarray, original_index: int | None = None) -> int:
        """Add a vertex unless an equivalent vertex already exists."""
        for index, existing in enumerate(result_vertices):
            if np.allclose(existing, point, atol=epsilon, rtol=0):
                if original_index is not None:
                    vertex_map[original_index] = index
                return index

        index = len(result_vertices)
        result_vertices.append(point)

        if original_index is not None:
            vertex_map[original_index] = index

        return index

    # Add vertices lying directly on the hyperplane.
    for index, point in enumerate(vertices):
        if abs(point[axis] - value) <= epsilon:
            sliced_point = np.delete(point, axis)
            add_vertex(sliced_point, index)

    # Find intersections between edges and the hyperplane.
    for a, b in edges:
        p1 = vertices[a]
        p2 = vertices[b]

        d1 = p1[axis] - value
        d2 = p2[axis] - value

        # Edge lies entirely on the slicing plane.
        if abs(d1) <= epsilon and abs(d2) <= epsilon:
            a_new = vertex_map[a]
            b_new = vertex_map[b]

            if a_new != b_new:
                edge = (min(a_new, b_new), max(a_new, b_new))
                if edge not in result_edges:
                    result_edges.append(edge)

            continue

        # Edge does not cross the plane.
        if d1 * d2 > 0:
            continue

        denominator = p2[axis] - p1[axis]

        if abs(denominator) <= epsilon:
            continue

        t = (value - p1[axis]) / denominator

        if t < -epsilon or t > 1 + epsilon:
            continue

        intersection = p1 + t * (p2 - p1)
        intersection[axis] = value

        sliced_point = np.delete(intersection, axis)

        intersection_index = add_vertex(sliced_point)

        # If the intersection is actually an existing endpoint,
        # remember that endpoint mapping.
        if abs(t) <= epsilon:
            vertex_map[a] = intersection_index
        elif abs(t - 1) <= epsilon:
            vertex_map[b] = intersection_index

    # Reconstruct edges for the resulting intersection geometry.
    #
    # For a coordinate slice of a polytope, every intersected
    # original edge contributes a sliced vertex. We need to connect
    # vertices belonging to the same sliced face.
    #
    # For the initial hypercube implementation, reconstructing
    # topology from the resulting points is more robust than assuming
    # a particular dimension.
    result_vertices_array = np.asarray(result_vertices, dtype=float)

    if len(result_vertices_array) > 1:
        result_edges = _infer_slice_edges(
            result_vertices_array,
            dimensions - 1,
            epsilon,
        )

    return Geometry(
        vertices=result_vertices_array,
        edges=np.asarray(result_edges, dtype=int).reshape(-1, 2),
    )


def _infer_slice_edges(
    vertices: np.ndarray,
    dimensions: int,
    epsilon: float,
) -> list[tuple[int, int]]:
    """Infer edges of a coordinate slice.

    For the initial hypercube implementation, vertices belonging to
    the slice retain the lower-dimensional hypercube structure.
    """
    edges: list[tuple[int, int]] = []

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            difference = np.abs(vertices[i] - vertices[j])

            nonzero = np.count_nonzero(difference > epsilon)

            if nonzero == 1:
                edges.append((i, j))

    return edges
