from __future__ import annotations

import numpy as np


def hypercube_edges(vertices: np.ndarray) -> np.ndarray:
    """Return hypercube edges as pairs of vertex indices."""
    vertices = np.asarray(vertices)

    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D array")

    edge_list: list[tuple[int, int]] = []

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            differences = np.count_nonzero(vertices[i] != vertices[j])

            if differences == 1:
                edge_list.append((i, j))

    return np.asarray(edge_list, dtype=int)


def vertex_adjacency(
    vertices: np.ndarray,
) -> list[list[int]]:
    """Return the adjacent vertex indices for every vertex."""
    edges = hypercube_edges(vertices)

    adjacency = [[] for _ in range(len(vertices))]

    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    return adjacency
