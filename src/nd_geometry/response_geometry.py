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
    parameter_names: tuple[str, ...]
    axes: tuple[np.ndarray, ...]


def response_grid_data(
    response: ResponseGeometry,
) -> SensitivityData:
    """Return sensitivity data reconstructed from response metadata."""
    shape = tuple(len(axis) for axis in response.axes)

    return SensitivityData(
        values=response.outputs.reshape(shape),
        parameter_names=response.parameter_names,
        axes=response.axes,
    )

@dataclass(frozen=True)
class LevelSetComponent:
    """A connected component of a response level set."""

    vertices: np.ndarray
    edges: np.ndarray

    @property
    def vertex_count(self) -> int:
        """Return the number of vertices in the component."""
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the component."""
        return len(self.edges)

@dataclass(frozen=True)
class LevelSetGeometry:
    """Geometry representing a continuous response level set."""

    geometry: Geometry
    target: float
    components: tuple[LevelSetComponent, ...]

    @property
    def component_count(self) -> int:
        """Return the number of connected level-set components."""
        return len(self.components)

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

def sensitivity_cells(
    data: SensitivityData | ResponseGeometry,
) -> np.ndarray:
    """Return the vertex indices belonging to every grid cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    shape = data.values.shape
    dimensions = data.dimensions

    cell_shape = tuple(size - 1 for size in shape)

    cells: list[list[int]] = []

    for index in np.ndindex(cell_shape):
        vertices: list[int] = []

        for corner in np.ndindex(*(2,) * dimensions):
            vertex_index = tuple(
                index[axis] + corner[axis]
                for axis in range(dimensions)
            )

            vertices.append(
                int(np.ravel_multi_index(vertex_index, shape))
            )

        cells.append(vertices)

    return np.asarray(cells, dtype=int)

def sensitivity_cell_simplices(
    data: SensitivityData | ResponseGeometry,
) -> np.ndarray:
    """Return an N-dimensional simplex decomposition for every grid cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    cells = sensitivity_cells(data)
    dimensions = data.dimensions

    simplices: list[list[int]] = []

    for cell in cells:
        for permutation in __import__("itertools").permutations(
            range(dimensions)
        ):
            simplex = [int(cell[0])]

            corner = [0] * dimensions

            for axis in permutation:
                corner[axis] = 1

                corner_index = 0
                for dimension in range(dimensions):
                    corner_index *= 2
                    corner_index += corner[dimension]

                simplex.append(int(cell[corner_index]))

            simplices.append(simplex)

    return np.asarray(
        simplices,
        dtype=int,
    ).reshape(-1, dimensions + 1)

def sensitivity_simplex_level_set_points(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return target-response intersection points for every simplex."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_vertices(data).vertices
    outputs = data.values.reshape(-1)
    simplices = sensitivity_cell_simplices(data)

    intersections: list[np.ndarray] = []

    for simplex in simplices:
        points: list[np.ndarray] = []

        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a = int(simplex[i])
                b = int(simplex[j])

                output_a = outputs[a]
                output_b = outputs[b]

                if output_a == output_b:
                    continue

                if not (
                    min(output_a, output_b)
                    <= target
                    <= max(output_a, output_b)
                ):
                    continue

                point = interpolate_response_crossing(
                    vertices[a],
                    output_a,
                    vertices[b],
                    output_b,
                    target,
                )

                if not any(
                    np.allclose(
                        point,
                        existing,
                        atol=1e-9,
                        rtol=0,
                    )
                    for existing in points
                ):
                    points.append(point)

        if len(points) >= 2:
            intersections.append(
                np.asarray(points, dtype=float)
            )

    return intersections

def sensitivity_simplex_level_set_facets(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return the local level-set facets generated by each simplex."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    intersections = sensitivity_simplex_level_set_points(
        data,
        target,
    )

    dimensions = data.dimensions
    facets: list[np.ndarray] = []

    for points in intersections:
        if len(points) == dimensions:
            facets.append(
                np.asarray(
                    points,
                    dtype=float,
                )
            )
        elif dimensions == 3 and len(points) == 4:
            facets.append(
                np.asarray(
                    points[[0, 1, 2]],
                    dtype=float,
                )
            )
            facets.append(
                np.asarray(
                    points[[0, 2, 3]],
                    dtype=float,
                )
            )

    return facets

def unique_sensitivity_simplex_level_set_facets(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return unique level-set facets across all simplices."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_simplex_level_set_facets(
        data,
        target,
    )

    unique_facets: list[np.ndarray] = []

    for facet in facets:
        if any(
            len(facet) == len(existing)
            and all(
                any(
                    np.allclose(
                        point,
                        existing_point,
                        atol=1e-9,
                        rtol=0,
                    )
                    for existing_point in existing
                )
                for point in facet
            )
            for existing in unique_facets
        ):
            continue

        unique_facets.append(facet.copy())

    return unique_facets

def sensitivity_simplex_level_set_vertices(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> np.ndarray:
    """Return unique vertices generated by simplex level-set intersections."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_simplex_level_set_facets(
        data,
        target,
    )

    unique_vertices: list[np.ndarray] = []

    for facet in facets:
        for point in facet:
            if not any(
                np.allclose(
                    point,
                    existing,
                    atol=1e-9,
                    rtol=0,
                )
                for existing in unique_vertices
            ):
                unique_vertices.append(point.copy())

    return np.asarray(
        sorted(
            unique_vertices,
            key=lambda point: tuple(
                -coordinate
                for coordinate in point
            ),
        ),
        dtype=float,
    ).reshape(-1, data.dimensions)

def sensitivity_simplex_level_set_facet_indices(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return global level-set vertex indices for every simplex facet."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_simplex_level_set_vertices(
        data,
        target,
    )

    facets = unique_sensitivity_simplex_level_set_facets(
        data,
        target,
    )

    indexed_facets: list[np.ndarray] = []

    for facet in facets:
        indices: list[int] = []

        for point in facet:
            index = next(
                (
                    vertex_index
                    for vertex_index, vertex in enumerate(vertices)
                    if np.allclose(
                        point,
                        vertex,
                        atol=1e-9,
                        rtol=0,
                    )
                ),
                None,
            )

            if index is not None and index not in indices:
                indices.append(index)

        if len(indices) >= 2:
            indexed_facets.append(
                np.asarray(
                    sorted(indices),
                    dtype=int,
                )
            )

    return indexed_facets


def sensitivity_simplex_level_set_facet_adjacency(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[list[int]]:
    """Return adjacency between level-set simplex facets."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_simplex_level_set_facet_indices(
        data,
        target,
    )

    adjacency: list[list[int]] = [
        [] for _ in range(len(facets))
    ]

    for i in range(len(facets)):
        vertices_a = set(
            int(vertex)
            for vertex in facets[i]
        )

        for j in range(i + 1, len(facets)):
            if len(facets[i]) != len(facets[j]):
                continue

            vertices_b = set(
                int(vertex)
                for vertex in facets[j]
            )

            shared = vertices_a & vertices_b

            if len(shared) >= 1:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency

def sensitivity_simplex_level_set_components(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return connected components of simplex level-set facets."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_simplex_level_set_facet_indices(
        data,
        target,
    )

    adjacency = sensitivity_simplex_level_set_facet_adjacency(
        data,
        target,
    )

    visited: set[int] = set()
    components: list[np.ndarray] = []

    for start in range(len(facets)):
        if start in visited:
            continue

        stack = [start]
        component: list[int] = []
        visited.add(start)

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(
            np.asarray(
                sorted(component),
                dtype=int,
            )
        )

    return components

def build_simplex_level_set_geometry(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> Geometry:
    """Build level-set geometry from simplicial intersections."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_level_set_vertices(
        data,
        target,
    )

    edges = sensitivity_level_set_edges(
        data,
        target,
    )

    return Geometry(
        vertices=vertices,
        edges=np.asarray(
            edges,
            dtype=int,
        ).reshape(-1, 2),
    )

def continuous_response_simplex_level_set(
    response: ResponseGeometry,
    target: float,
) -> LevelSetGeometry:
    """Return a simplicial parameter-space representation of a response level set."""
    geometry = build_simplex_level_set_geometry(
        response,
        target,
    )

    facet_indices = sensitivity_simplex_level_set_facet_indices(
        response,
        target,
    )

    vertices = sensitivity_simplex_level_set_vertices(
        response,
        target,
    )

    components = sensitivity_simplex_level_set_components(
        response,
        target,
    )

    level_set_components: list[LevelSetComponent] = []

    for component in components:
        global_indices = sorted(
            {
                int(vertex)
                for facet_index in component
                for vertex in facet_indices[int(facet_index)]
            }
        )

        component_vertices = vertices[global_indices].copy()

        index_map = {
            global_index: local_index
            for local_index, global_index in enumerate(global_indices)
        }

        component_edges: set[tuple[int, int]] = set()

        for facet_index in component:
            facet = facet_indices[int(facet_index)]

            for i in range(len(facet)):
                for j in range(i + 1, len(facet)):
                    a = index_map[int(facet[i])]
                    b = index_map[int(facet[j])]

                    if a != b:
                        component_edges.add(
                            tuple(sorted((a, b)))
                        )

        level_set_components.append(
            LevelSetComponent(
                vertices=component_vertices,
                edges=np.asarray(
                    sorted(component_edges),
                    dtype=int,
                ).reshape(-1, 2),
            )
        )

    return LevelSetGeometry(
        geometry=geometry,
        target=target,
        components=tuple(level_set_components),
    )

def sensitivity_cell_indices(
    data: SensitivityData,
) -> np.ndarray:
    """Return the grid index of every sensitivity cell."""
    cell_shape = tuple(
        size - 1
        for size in data.values.shape
    )

    if any(size <= 0 for size in cell_shape):
        return np.empty(
            (0, data.dimensions),
            dtype=int,
        )

    return np.asarray(
        list(np.ndindex(cell_shape)),
        dtype=int,
    ).reshape(-1, data.dimensions)

def sensitivity_cell_neighbors(
    data: SensitivityData | ResponseGeometry,
) -> list[list[int]]:
    """Return neighboring cell indices for every sensitivity cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    cell_indices = sensitivity_cell_indices(data)

    if len(cell_indices) == 0:
        return []

    index_map = {
        tuple(index): cell_number
        for cell_number, index in enumerate(cell_indices)
    }

    neighbors: list[list[int]] = [
        []
        for _ in range(len(cell_indices))
    ]

    for cell_number, index in enumerate(cell_indices):
        for axis in range(data.dimensions):
            for direction in (-1, 1):
                neighbor_index = index.copy()
                neighbor_index[axis] += direction

                neighbor_number = index_map.get(
                    tuple(neighbor_index)
                )

                if neighbor_number is not None:
                    neighbors[cell_number].append(
                        neighbor_number
                    )

    return neighbors

def sensitivity_cell_facets(
    data: SensitivityData | ResponseGeometry,
) -> np.ndarray:
    """Return the boundary edges/facets of every grid cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    shape = data.values.shape
    dimensions = data.dimensions

    cells = sensitivity_cells(data)
    facets: list[list[int]] = []

    for cell in cells:
        for axis in range(dimensions):
            for side in range(2):
                facet: list[int] = []

                for corner in np.ndindex(*(2,) * dimensions):
                    if corner[axis] != side:
                        continue

                    corner_index = 0

                    for dimension in range(dimensions):
                        corner_index *= 2
                        corner_index += corner[dimension]

                    facet.append(int(cell[corner_index]))

                facets.append(facet)

    return np.asarray(facets, dtype=int)

def unique_sensitivity_facets(
    data: SensitivityData,
) -> np.ndarray:
    """Return unique facets across all sensitivity grid cells."""
    facets = sensitivity_cell_facets(data)

    unique_facets = {
        tuple(sorted(facet))
        for facet in facets
    }

    return np.asarray(
        sorted(unique_facets),
        dtype=int,
    )

def sensitivity_cell_facet_adjacency(
    data: SensitivityData | ResponseGeometry,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return unique facets and the facet indices belonging to each cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)
        
    cells = sensitivity_cells(data)
    cell_facets = sensitivity_cell_facets(data)

    facet_map: dict[tuple[int, ...], int] = {}
    unique_facets: list[tuple[int, ...]] = []
    adjacency: list[list[int]] = [
        []
        for _ in range(len(cells))
    ]

    facets_per_cell = 2 ** data.dimensions

    for cell_index in range(len(cells)):
        start = cell_index * facets_per_cell
        end = start + facets_per_cell

        for facet in cell_facets[start:end]:
            key = tuple(sorted(int(vertex) for vertex in facet))

            if key not in facet_map:
                facet_map[key] = len(unique_facets)
                unique_facets.append(key)

            adjacency[cell_index].append(facet_map[key])

    return (
        np.asarray(unique_facets, dtype=int),
        adjacency,
    )

def sensitivity_boundary_facets(
    data: SensitivityData | ResponseGeometry,
) -> np.ndarray:
    """Return facets belonging to only one sensitivity grid cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)
    facets, adjacency = sensitivity_cell_facet_adjacency(data)

    facet_usage = np.zeros(
        len(facets),
        dtype=int,
    )

    for cell_facets in adjacency:
        for facet_index in cell_facets:
            facet_usage[facet_index] += 1

    return facets[facet_usage == 1].copy()

def sensitivity_facet_output_ranges(
    data: SensitivityData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return facets and their minimum and maximum response values."""
    facets, _ = sensitivity_cell_facet_adjacency(data)

    outputs = data.values.reshape(-1)

    minimums = np.min(
        outputs[facets],
        axis=1,
    )

    maximums = np.max(
        outputs[facets],
        axis=1,
    )

    return (
        facets,
        minimums,
        maximums,
    )

def sensitivity_intersected_facets(
    data: SensitivityData,
    target: float,
) -> np.ndarray:
    """Return facets whose response range contains the target."""
    facets, minimums, maximums = sensitivity_facet_output_ranges(data)

    mask = (
        (minimums <= target)
        & (target <= maximums)
    )

    return facets[mask].copy()

def sensitivity_facet_crossings(
    data: SensitivityData,
    target: float,
) -> list[np.ndarray]:
    """Return interpolated level-set crossings on grid facets."""
    facets = sensitivity_intersected_facets(data, target)
    outputs = data.values.reshape(-1)
    vertices = sensitivity_vertices(data).vertices

    crossings: list[np.ndarray] = []

    for facet in facets:
        facet_crossings: list[np.ndarray] = []

        for i in range(len(facet)):
            for j in range(i + 1, len(facet)):
                a = int(facet[i])
                b = int(facet[j])

                output_a = outputs[a]
                output_b = outputs[b]

                if output_a == output_b:
                    continue

                if not (
                    min(output_a, output_b)
                    <= target
                    <= max(output_a, output_b)
                ):
                    continue

                crossing = interpolate_response_crossing(
                    vertices[a],
                    output_a,
                    vertices[b],
                    output_b,
                    target,
                )

                if not any(
                    np.allclose(
                        crossing,
                        existing,
                        atol=1e-9,
                        rtol=0,
                    )
                    for existing in facet_crossings
                ):
                    facet_crossings.append(crossing)

        crossings.extend(facet_crossings)

    return crossings

def sensitivity_cell_crossings(
    data: SensitivityData,
    target: float,
) -> list[np.ndarray]:
    """Return unique level-set crossing points for every grid cell."""
    facets, adjacency = sensitivity_cell_facet_adjacency(data)
    outputs = data.values.reshape(-1)
    vertices = sensitivity_vertices(data).vertices

    crossings_by_cell: list[list[np.ndarray]] = []

    for cell_facets in adjacency:
        cell_crossings: list[np.ndarray] = []

        for facet_index in cell_facets:
            facet = facets[facet_index]

            for i in range(len(facet)):
                for j in range(i + 1, len(facet)):
                    a = int(facet[i])
                    b = int(facet[j])

                    output_a = outputs[a]
                    output_b = outputs[b]

                    if output_a == output_b:
                        continue

                    if not (
                        min(output_a, output_b)
                        <= target
                        <= max(output_a, output_b)
                    ):
                        continue

                    crossing = interpolate_response_crossing(
                        vertices[a],
                        output_a,
                        vertices[b],
                        output_b,
                        target,
                    )

                    if not any(
                        np.allclose(
                            crossing,
                            existing,
                            atol=1e-9,
                            rtol=0,
                        )
                        for existing in cell_crossings
                    ):
                        cell_crossings.append(crossing)

        crossings_by_cell.append(cell_crossings)

    return crossings_by_cell

def sensitivity_cell_level_set_boundary(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return ordered level-set boundary points for each intersected cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    dimensions = data.dimensions
    crossings_by_cell = sensitivity_cell_crossings(
        data,
        target,
    )

    boundaries: list[np.ndarray] = []

    for crossings in crossings_by_cell:
        points = np.asarray(
            crossings,
            dtype=float,
        )

        if len(points) < 2:
            continue

        boundaries.append(points)

    return boundaries

def sensitivity_cell_level_set_edges(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return level-set edges for each intersected sensitivity cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    boundaries = sensitivity_cell_level_set_boundary(
        data,
        target,
    )

    edges: list[np.ndarray] = []

    for boundary in boundaries:
        if len(boundary) < 2:
            continue

        if len(boundary) == 2:
            edges.append(boundary.copy())
            continue

        if data.dimensions > 3:
            continue

        for index in range(len(boundary)):
            next_index = (index + 1) % len(boundary)

            edges.append(
                np.asarray(
                    [
                        boundary[index],
                        boundary[next_index],
                    ],
                    dtype=float,
                )
            )

    return edges


def sensitivity_cell_level_set_points(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return level-set crossing points for every intersected cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)
    crossings_by_cell = sensitivity_cell_crossings(
        data,
        target,
    )

    return [
        np.asarray(crossings, dtype=float).reshape(
            -1,
            data.dimensions,
        )
        for crossings in crossings_by_cell
        if crossings
    ]

def sensitivity_level_set_vertices(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> np.ndarray:
    """Return unique vertices of the continuous response level set."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)
    cell_points = sensitivity_cell_level_set_points(
        data,
        target,
    )

    unique_vertices: list[np.ndarray] = []

    for points in cell_points:
        for point in points:
            if any(
                np.allclose(
                    point,
                    existing,
                    atol=1e-9,
                    rtol=0,
                )
                for existing in unique_vertices
            ):
                continue

            unique_vertices.append(point.copy())

    return np.asarray(
        sorted(
            unique_vertices,
            key=lambda point: tuple(
                -coordinate
                for coordinate in point
            ),
        ),
        dtype=float,
    ).reshape(
        -1,
        data.dimensions,
    )

def sensitivity_level_set_edges(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> np.ndarray:
    """Return unique edges of the continuous response level set."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_level_set_vertices(
        data,
        target,
    )

    cell_edges = sensitivity_cell_level_set_edges(
        data,
        target,
    )

    edges: set[tuple[int, int]] = set()

    for cell_edge in cell_edges:
        indices: list[int] = []

        for point in cell_edge:
            index = next(
                (
                    vertex_index
                    for vertex_index, vertex in enumerate(vertices)
                    if np.allclose(
                        point,
                        vertex,
                        atol=1e-9,
                        rtol=0,
                    )
                ),
                None,
            )

            if index is not None:
                indices.append(index)

        if len(indices) != 2:
            continue

        a, b = sorted(indices)

        if a != b:
            edges.add((a, b))

    return np.asarray(
        sorted(edges),
        dtype=int,
    ).reshape(-1, 2)

def sensitivity_level_set_facets(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return level-set facets generated within each sensitivity cell."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    cell_points = sensitivity_cell_level_set_points(
        data,
        target,
    )

    facets: list[np.ndarray] = []

    for points in cell_points:
        if len(points) < 2:
            continue

        facets.append(
            np.asarray(
                points,
                dtype=float,
            )
        )

    return facets

def unique_sensitivity_level_set_facets(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return unique level-set facets across sensitivity cells."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_level_set_facets(
        data,
        target,
    )

    unique_facets: list[np.ndarray] = []

    for facet in facets:
        if any(
            len(facet) == len(existing)
            and all(
                any(
                    np.allclose(
                        point,
                        existing_point,
                        atol=1e-9,
                        rtol=0,
                    )
                    for existing_point in existing
                )
                for point in facet
            )
            for existing in unique_facets
        ):
            continue

        unique_facets.append(facet.copy())

    return unique_facets

def sensitivity_level_set_facet_indices(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return global vertex indices for every unique level-set facet."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_level_set_vertices(
        data,
        target,
    )

    facets = unique_sensitivity_level_set_facets(
        data,
        target,
    )

    indexed_facets: list[np.ndarray] = []

    for facet in facets:
        indices: list[int] = []

        for point in facet:
            index = next(
                (
                    vertex_index
                    for vertex_index, vertex in enumerate(vertices)
                    if np.allclose(
                        point,
                        vertex,
                        atol=1e-9,
                        rtol=0,
                    )
                ),
                None,
            )

            if index is not None:
                indices.append(index)

        if len(indices) >= 2:
            indexed_facets.append(
                np.asarray(
                    sorted(set(indices)),
                    dtype=int,
                )
            )

    return indexed_facets

def sensitivity_level_set_facet_adjacency(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[list[int]]:
    """Return neighboring level-set facets that share vertices."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_level_set_facet_indices(
        data,
        target,
    )

    adjacency: list[list[int]] = [
        [] for _ in range(len(facets))
    ]

    for i in range(len(facets)):
        vertices_a = set(int(vertex) for vertex in facets[i])

        for j in range(i + 1, len(facets)):
            vertices_b = set(int(vertex) for vertex in facets[j])

            if vertices_a & vertices_b:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency

def sensitivity_level_set_components(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return connected components of the response level-set facets."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    facets = sensitivity_level_set_facet_indices(
        data,
        target,
    )

    adjacency = sensitivity_level_set_facet_adjacency(
        data,
        target,
    )

    visited: set[int] = set()
    components: list[np.ndarray] = []

    for start in range(len(facets)):
        if start in visited:
            continue

        stack = [start]
        component: list[int] = []
        visited.add(start)

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(
            np.asarray(
                sorted(component),
                dtype=int,
            )
        )

    return components

def sensitivity_level_set_component_vertices(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return parameter-space vertices for each level-set component."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_level_set_vertices(
        data,
        target,
    )

    facets = sensitivity_level_set_facet_indices(
        data,
        target,
    )

    components = sensitivity_level_set_components(
        data,
        target,
    )

    component_vertices: list[np.ndarray] = []

    for component in components:
        indices = sorted(
            {
                int(vertex)
                for facet_index in component
                for vertex in facets[int(facet_index)]
            }
        )

        component_vertices.append(
            vertices[indices].copy()
        )

    return component_vertices

def sensitivity_level_set_component_edges(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[np.ndarray]:
    """Return global level-set edges for each connected component."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_level_set_vertices(
        data,
        target,
    )

    edges = sensitivity_level_set_edges(
        data,
        target,
    )

    components = sensitivity_level_set_components(
        data,
        target,
    )

    component_edges: list[np.ndarray] = []

    for component in components:
        component_facets = {
            int(facet_index)
            for facet_index in component
        }

        facets = sensitivity_level_set_facet_indices(
            data,
            target,
        )

        component_vertices = {
            int(vertex)
            for facet_index in component_facets
            for vertex in facets[facet_index]
        }

        selected_edges: list[np.ndarray] = []

        for edge in edges:
            if all(
                int(vertex) in component_vertices
                for vertex in edge
            ):
                selected_edges.append(edge)

        component_edges.append(
            np.asarray(
                selected_edges,
                dtype=int,
            ).reshape(-1, 2)
        )

    return component_edges

def build_level_set_components(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> list[LevelSetComponent]:
    """Build connected level-set components."""
    if isinstance(data, ResponseGeometry):
        data = response_grid_data(data)

    vertices = sensitivity_level_set_vertices(
        data,
        target,
    )

    component_vertices = sensitivity_level_set_component_vertices(
        data,
        target,
    )

    component_edges = sensitivity_level_set_component_edges(
        data,
        target,
    )

    components: list[LevelSetComponent] = []

    for points, edges in zip(
        component_vertices,
        component_edges,
    ):
        local_indices: dict[int, int] = {}

        for local_index, point in enumerate(points):
            global_index = next(
                vertex_index
                for vertex_index, vertex in enumerate(vertices)
                if np.allclose(
                    point,
                    vertex,
                    atol=1e-9,
                    rtol=0,
                )
            )

            local_indices[global_index] = local_index

        local_edges = np.asarray(
            [
                [
                    local_indices[int(edge[0])],
                    local_indices[int(edge[1])],
                ]
                for edge in edges
            ],
            dtype=int,
        ).reshape(-1, 2)

        components.append(
            LevelSetComponent(
                vertices=points.copy(),
                edges=local_edges,
            )
        )

    return components

def build_continuous_level_set_geometry(
    data: SensitivityData | ResponseGeometry,
    target: float,
) -> Geometry:
    """Build continuous level-set geometry from sensitivity data."""
    return Geometry(
        vertices=sensitivity_level_set_vertices(data, target),
        edges=sensitivity_level_set_edges(data, target),
    )

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
        parameter_names=data.parameter_names,
        axes=data.axes,
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
    parameter_names: tuple[str, ...]
    axes: tuple[np.ndarray, ...]

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

def response_level_set(
    response: ResponseGeometry,
    target: float,
    tolerance: float = 1e-9,
) -> Geometry:
    """Return geometry for parameter combinations near a target output."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    mask = np.isclose(
        response.outputs,
        target,
        atol=tolerance,
        rtol=0,
    )

    selected_indices = np.flatnonzero(mask)

    vertices = response.geometry.vertices[mask].copy()

    if len(selected_indices) < 2:
        edges = np.empty((0, 2), dtype=int)
    else:
        index_map = {
            original: new
            for new, original in enumerate(selected_indices)
        }

        selected_set = set(selected_indices)

        edges_list: list[tuple[int, int]] = []

        for a, b in response.geometry.edges:
            if a in selected_set and b in selected_set:
                edges_list.append(
                    (index_map[a], index_map[b])
                )

        edges = np.asarray(
            edges_list,
            dtype=int,
        ).reshape(-1, 2)

    return Geometry(
        vertices=vertices,
        edges=edges,
    )

def interpolate_response_crossing(
    point_a: np.ndarray,
    output_a: float,
    point_b: np.ndarray,
    output_b: float,
    target: float,
) -> np.ndarray:
    """Interpolate a parameter-space point where a response reaches target."""
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)

    if point_a.ndim != 1 or point_b.ndim != 1:
        raise ValueError("points must be 1D arrays")

    if point_a.shape != point_b.shape:
        raise ValueError("points must have matching dimensions")

    if np.isclose(output_a, output_b):
        raise ValueError(
            "cannot interpolate between equal output values"
        )

    fraction = (target - output_a) / (output_b - output_a)

    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(
            "target must lie between the endpoint outputs"
        )

    return point_a + fraction * (point_b - point_a)

def continuous_response_level_set(
    response: ResponseGeometry,
    target: float,
) -> LevelSetGeometry:
    """Return parameter-space geometry where response equals target."""
    geometry = build_continuous_level_set_geometry(
        response,
        target,
    )

    components = tuple(
        build_level_set_components(
            response,
            target,
        )
    )

    return LevelSetGeometry(
        geometry=geometry,
        target=target,
        components=components,
    )