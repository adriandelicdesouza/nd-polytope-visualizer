from __future__ import annotations

import numpy as np
import pytest
from nd_geometry.response_geometry import (
    ResponseGeometry,
    build_response_geometry,
    sensitivity_outputs,
    sensitivity_vertices,
    normalize_outputs,
    embed_outputs,
    response_level_set,
    interpolate_response_crossing,
    continuous_response_level_set,
    sensitivity_cells,
    sensitivity_cell_simplices,
    sensitivity_simplex_level_set_points,
    sensitivity_simplex_level_set_facets,
    unique_sensitivity_simplex_level_set_facets,
    sensitivity_simplex_level_set_facet_indices,
    sensitivity_simplex_level_set_facet_adjacency,
    sensitivity_simplex_level_set_components,
    build_simplex_level_set_geometry,
    continuous_response_simplex_level_set,
    sensitivity_simplex_level_set_vertices,
    sensitivity_cell_facets,
    unique_sensitivity_facets,
    sensitivity_cell_facet_adjacency,
    sensitivity_cell_indices,
    sensitivity_cell_neighbors,
    sensitivity_boundary_facets,
    sensitivity_facet_output_ranges,
    sensitivity_intersected_facets,
    sensitivity_facet_crossings,
    sensitivity_cell_crossings,
    sensitivity_cell_level_set_edges,
    sensitivity_cell_level_set_points,
    sensitivity_cell_level_set_boundary,
    sensitivity_level_set_vertices,
    sensitivity_level_set_edges,
    sensitivity_level_set_facets,
    sensitivity_level_set_facet_indices,
    sensitivity_level_set_facet_adjacency,
    sensitivity_level_set_components,
    sensitivity_level_set_component_vertices,
    sensitivity_level_set_component_edges,
    build_continuous_level_set_geometry,
    build_level_set_components,
    response_grid_data,
    unique_sensitivity_level_set_facets,
    LevelSetComponent,
    LevelSetGeometry,
)
from nd_geometry.slicing import Geometry
from nd_geometry.sensitivity import SensitivityData

def test_sensitivity_vertices_returns_parameter_grid():
    data = SensitivityData(
        values=np.zeros((2, 3)),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.175, 0.25]),
        ),
    )

    geometry = sensitivity_vertices(data)

    assert geometry.vertices.shape == (6, 2)
    assert geometry.edges.shape == (7, 2)


def test_sensitivity_vertices_contains_all_parameter_combinations():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    geometry = sensitivity_vertices(data)

    expected = np.array([
        [0.02, 0.10],
        [0.02, 0.20],
        [0.10, 0.10],
        [0.10, 0.20],
    ])

    np.testing.assert_allclose(
        geometry.vertices,
        expected,
    )


def test_sensitivity_edges_connect_grid_neighbors():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    geometry = sensitivity_vertices(data)

    expected_edges = {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    }

    actual_edges = {
        tuple(edge)
        for edge in geometry.edges
    }

    assert actual_edges == expected_edges

def test_sensitivity_vertices_builds_3d_grid_topology():
    data = SensitivityData(
        values=np.zeros((2, 2, 2)),
        parameter_names=("a", "b", "c"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    geometry = sensitivity_vertices(data)

    assert geometry.vertices.shape == (8, 3)
    assert geometry.edges.shape == (12, 2)

def test_sensitivity_outputs_align_with_vertices():
    data = SensitivityData(
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    geometry = sensitivity_vertices(data)
    outputs = sensitivity_outputs(data)

    assert len(outputs) == len(geometry.vertices)

    np.testing.assert_allclose(
        outputs,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

def test_build_response_geometry():
    data = SensitivityData(
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    assert isinstance(response, ResponseGeometry)
    assert response.geometry.vertices.shape == (4, 2)
    assert response.geometry.edges.shape == (4, 2)

    np.testing.assert_allclose(
        response.outputs,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

def test_normalize_outputs():
    outputs = np.array([10.0, 20.0, 30.0, 40.0])

    normalized = normalize_outputs(outputs)

    np.testing.assert_allclose(
        normalized,
        np.array([0.0, 1 / 3, 2 / 3, 1.0]),
    )


def test_normalize_outputs_constant_values():
    outputs = np.array([5.0, 5.0, 5.0])

    normalized = normalize_outputs(outputs)

    np.testing.assert_allclose(
        normalized,
        np.zeros(3),
    )


def test_normalize_outputs_rejects_non_1d():
    outputs = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError):
        normalize_outputs(outputs)

def test_response_geometry_normalized_outputs():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    np.testing.assert_allclose(
        response.normalized_outputs,
        np.array([0.0, 1 / 3, 2 / 3, 1.0]),
    )

def test_response_geometry_output_range():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    assert response.output_range == (10.0, 40.0)

def test_response_geometry_output_at():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    assert response.output_at(0) == 10.0
    assert response.output_at(3) == 40.0


def test_response_geometry_output_at_rejects_invalid_index():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    with pytest.raises(IndexError):
        response.output_at(4)

def test_embed_outputs():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)
    geometry = embed_outputs(response)

    assert geometry.vertices.shape == (4, 3)
    assert geometry.edges.shape == (4, 2)

    np.testing.assert_allclose(
        geometry.vertices,
        np.array([
            [0.02, 0.10, 10.0],
            [0.02, 0.20, 20.0],
            [0.10, 0.10, 30.0],
            [0.10, 0.20, 40.0],
        ]),
    )

def test_embed_outputs_normalized():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)
    geometry = embed_outputs(response, normalize=True)

    np.testing.assert_allclose(
        geometry.vertices[:, -1],
        np.array([0.0, 1 / 3, 2 / 3, 1.0]),
    )

def test_embed_outputs_preserves_edges():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)
    geometry = embed_outputs(response)

    np.testing.assert_array_equal(
        geometry.edges,
        response.geometry.edges,
    )

def test_embed_outputs_does_not_modify_response():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    original_vertices = response.geometry.vertices.copy()

    embed_outputs(response)

    np.testing.assert_array_equal(
        response.geometry.vertices,
        original_vertices,
    )

def test_embed_outputs_scales_response_axis():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)
    geometry = embed_outputs(response, scale=0.1)

    np.testing.assert_allclose(
        geometry.vertices[:, -1],
        np.array([1.0, 2.0, 3.0, 4.0]),
    )


def test_embed_outputs_rejects_non_positive_scale():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    with pytest.raises(ValueError):
        embed_outputs(response, scale=0.0)

    with pytest.raises(ValueError):
        embed_outputs(response, scale=-1.0)

def test_response_level_set():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    level_set = response_level_set(
        response,
        target=20.0,
    )

    np.testing.assert_allclose(
        level_set.vertices,
        np.array([
            [0.02, 0.20],
        ]),
    )

    assert level_set.edges.shape == (0, 2)


def test_response_level_set_tolerance():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.001],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    level_set = response_level_set(
        response,
        target=20.0,
        tolerance=0.01,
    )

    assert level_set.vertices.shape == (1, 2)
    assert level_set.edges.shape == (0, 2)

def test_response_level_set_rejects_negative_tolerance():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    with pytest.raises(ValueError):
        response_level_set(
            response,
            target=0.0,
            tolerance=-0.01,
        )

def test_response_level_set_can_be_empty():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    level_set = response_level_set(
        response,
        target=100.0,
    )

    assert level_set.vertices.shape == (0, 2)
    assert level_set.edges.shape == (0, 2)

def test_response_level_set_preserves_edges():
    data = SensitivityData(
        values=np.array([
            [10.0, 10.0],
            [20.0, 30.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    response = build_response_geometry(data)

    level_set = response_level_set(
        response,
        target=10.0,
    )

    np.testing.assert_allclose(
        level_set.vertices,
        np.array([
            [0.02, 0.10],
            [0.02, 0.20],
        ]),
    )

    np.testing.assert_array_equal(
        level_set.edges,
        np.array([
            [0, 1],
        ]),
    )

def test_interpolate_response_crossing():
    point_a = np.array([0.02, 0.10])
    point_b = np.array([0.10, 0.20])

    crossing = interpolate_response_crossing(
        point_a,
        10.0,
        point_b,
        30.0,
        target=20.0,
    )

    np.testing.assert_allclose(
        crossing,
        np.array([0.06, 0.15]),
    )


def test_interpolate_response_crossing_endpoint():
    point_a = np.array([0.02, 0.10])
    point_b = np.array([0.10, 0.20])

    crossing = interpolate_response_crossing(
        point_a,
        10.0,
        point_b,
        30.0,
        target=10.0,
    )

    np.testing.assert_allclose(
        crossing,
        point_a,
    )


def test_interpolate_response_crossing_rejects_equal_outputs():
    with pytest.raises(ValueError):
        interpolate_response_crossing(
            np.array([0.0]),
            10.0,
            np.array([1.0]),
            10.0,
            target=10.0,
        )

def test_interpolate_response_crossing_rejects_non_1d_points():
    with pytest.raises(ValueError):
        interpolate_response_crossing(
            np.array([[0.0]]),
            10.0,
            np.array([1.0]),
            20.0,
            target=15.0,
        )


def test_interpolate_response_crossing_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        interpolate_response_crossing(
            np.array([0.0, 1.0]),
            10.0,
            np.array([2.0]),
            20.0,
            target=15.0,
        )

def test_interpolate_response_crossing_rejects_target_outside_range():
    with pytest.raises(ValueError):
        interpolate_response_crossing(
            np.array([0.0]),
            10.0,
            np.array([1.0]),
            20.0,
            target=25.0,
        )

def test_interpolate_response_crossing_decreasing_output():
    point_a = np.array([0.02, 0.10])
    point_b = np.array([0.10, 0.20])

    crossing = interpolate_response_crossing(
        point_a,
        30.0,
        point_b,
        10.0,
        target=20.0,
    )

    np.testing.assert_allclose(
        crossing,
        np.array([0.06, 0.15]),
    )

def test_interpolate_response_crossing_upper_endpoint():
    point_a = np.array([0.02, 0.10])
    point_b = np.array([0.10, 0.20])

    crossing = interpolate_response_crossing(
        point_a,
        10.0,
        point_b,
        30.0,
        target=30.0,
    )

    np.testing.assert_allclose(
        crossing,
        point_b,
    )

def test_continuous_response_level_set():
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    crossings = continuous_response_level_set(
        response,
        target=20.0,
    )

    assert crossings.geometry.vertices.shape == (2, 2)
    assert crossings.target == 20.0

    np.testing.assert_allclose(
        crossings.geometry.vertices,
        np.array([
            [0.5, 0.0],
            [0.0, 1.0],
        ]),
    )

def test_continuous_response_level_set_builds_topology() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    level_set = continuous_response_level_set(
        response,
        target=20.0,
    )

    assert level_set.geometry.vertices.shape == (2, 2)
    assert level_set.geometry.edges.shape == (1, 2)

    np.testing.assert_array_equal(
        level_set.geometry.edges,
        np.array([[0, 1]]),
    )

def test_continuous_response_level_set_builds_3d_topology() -> None:
    data = SensitivityData(
        values=np.array([
            [[0.0, 1.0],
             [1.0, 2.0]],

            [[1.0, 2.0],
             [2.0, 3.0]],
        ]),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    level_set = continuous_response_level_set(
        response,
        target=1.0,
    )

    assert level_set.geometry.vertices.shape[1] == 3
    assert len(level_set.geometry.vertices) > 0
    assert len(level_set.geometry.edges) > 0

def test_sensitivity_cells_returns_2d_grid_cells() -> None:
    data = SensitivityData(
        values=np.zeros((3, 3)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    cells = sensitivity_cells(data)

    assert cells.shape == (4, 4)

    np.testing.assert_array_equal(
        cells[0],
        np.array([0, 1, 3, 4]),
    )

def test_sensitivity_cells_returns_3d_grid_cells() -> None:
    data = SensitivityData(
        values=np.zeros((2, 2, 2)),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    cells = sensitivity_cells(data)

    assert cells.shape == (1, 8)

    np.testing.assert_array_equal(
        cells[0],
        np.arange(8),
    )

def test_sensitivity_cells_returns_4d_grid_cells() -> None:
    data = SensitivityData(
        values=np.zeros((2, 2, 2, 2)),
        parameter_names=("a", "b", "c", "d"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    cells = sensitivity_cells(data)

    assert cells.shape == (1, 16)

    np.testing.assert_array_equal(
        cells[0],
        np.arange(16),
    )

def test_sensitivity_cell_facets_returns_2d_cell_edges() -> None:
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_cell_facets(data)

    assert facets.shape == (4, 2)

    expected = {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    }

    actual = {
        tuple(sorted(facet))
        for facet in facets
    }

    assert actual == expected

def test_sensitivity_cell_facets_returns_3d_cell_faces() -> None:
    data = SensitivityData(
        values=np.zeros((2, 2, 2)),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_cell_facets(data)

    assert facets.shape == (6, 4)

    assert {
        tuple(sorted(facet))
        for facet in facets
    } == {
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (2, 3, 6, 7),
        (4, 5, 6, 7),
    }

def test_sensitivity_cell_facets_returns_4d_cell_facets() -> None:
    data = SensitivityData(
        values=np.zeros((2, 2, 2, 2)),
        parameter_names=("a", "b", "c", "d"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_cell_facets(data)

    assert facets.shape == (8, 8)

    assert {
        tuple(sorted(facet))
        for facet in facets
    } == {
        (0, 1, 2, 3, 4, 5, 6, 7),
        (8, 9, 10, 11, 12, 13, 14, 15),
        (0, 1, 2, 3, 8, 9, 10, 11),
        (4, 5, 6, 7, 12, 13, 14, 15),
        (0, 1, 4, 5, 8, 9, 12, 13),
        (2, 3, 6, 7, 10, 11, 14, 15),
        (0, 2, 4, 6, 8, 10, 12, 14),
        (1, 3, 5, 7, 9, 11, 13, 15),
    }

def test_sensitivity_cell_facets_returns_facets_for_multiple_cells() -> None:
    data = SensitivityData(
        values=np.zeros((3, 2)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_cell_facets(data)

    assert facets.shape == (8, 2)

    actual = {
        tuple(sorted(facet))
        for facet in facets
    }

    expected = {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
    }

    assert actual == expected

def test_unique_sensitivity_facets_removes_shared_facets() -> None:
    data = SensitivityData(
        values=np.zeros((3, 2)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = unique_sensitivity_facets(data)

    assert facets.shape == (7, 2)

    np.testing.assert_array_equal(
        facets,
        np.array([
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [3, 5],
            [4, 5],
        ]),
    )

def test_sensitivity_cell_facet_adjacency_identifies_shared_facets() -> None:
    data = SensitivityData(
        values=np.zeros((3, 2)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets, adjacency = sensitivity_cell_facet_adjacency(data)

    assert facets.shape == (7, 2)
    assert len(adjacency) == 2
    assert all(len(cell) == 4 for cell in adjacency)

    shared = set(adjacency[0]) & set(adjacency[1])

    assert len(shared) == 1

    shared_facet = facets[next(iter(shared))]

    np.testing.assert_array_equal(
        shared_facet,
        np.array([2, 3]),
    )

def test_sensitivity_cell_indices_returns_grid_indices() -> None:
    data = SensitivityData(
        values=np.zeros((3, 2)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
        ),
    )

    indices = sensitivity_cell_indices(data)

    assert indices.shape == (2, 2)

    np.testing.assert_array_equal(
        indices,
        np.array([
            [0, 0],
            [1, 0],
        ]),
    )

def test_sensitivity_cell_neighbors_returns_2d_neighbors() -> None:
    data = SensitivityData(
        values=np.zeros((3, 3)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    neighbors = sensitivity_cell_neighbors(data)

    assert len(neighbors) == 4

    assert set(neighbors[0]) == {1, 2}
    assert set(neighbors[1]) == {0, 3}
    assert set(neighbors[2]) == {0, 3}
    assert set(neighbors[3]) == {1, 2}

def test_sensitivity_boundary_facets_excludes_internal_facets() -> None:
    data = SensitivityData(
        values=np.zeros((3, 2)),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_boundary_facets(data)

    assert facets.shape == (6, 2)

    assert {
        tuple(sorted(facet))
        for facet in facets
    } == {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 5),
    }

def test_sensitivity_facet_output_ranges_returns_minimums_and_maximums() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets, minimums, maximums = sensitivity_facet_output_ranges(data)

    assert facets.shape == (4, 2)

    np.testing.assert_array_equal(
        minimums,
        np.array([10.0, 30.0, 10.0, 20.0]),
    )

    np.testing.assert_array_equal(
        maximums,
        np.array([20.0, 40.0, 30.0, 40.0]),
    )

def test_sensitivity_intersected_facets_returns_target_crossings() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_intersected_facets(
        data,
        target=20.0,
    )

    assert facets.shape == (3, 2)

    assert {
        tuple(sorted(facet))
        for facet in facets
    } == {
        (0, 1),
        (0, 2),
        (1, 3),
    }

def test_sensitivity_facet_crossings_returns_interpolated_points() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    crossings = sensitivity_facet_crossings(
        data,
        target=20.0,
    )

    assert len(crossings) == 3

    actual = {
        tuple(np.round(crossing, 9))
        for crossing in crossings
    }

    assert actual == {
        (0.0, 1.0),
        (0.5, 0.0),
        (0.0, 1.0),
    }

def test_sensitivity_cell_crossings_deduplicates_shared_crossings() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    crossings = sensitivity_cell_crossings(
        data,
        target=20.0,
    )

    assert len(crossings) == 1
    assert len(crossings[0]) == 2

    np.testing.assert_allclose(
        sorted(crossings[0], key=lambda point: tuple(point)),
        np.array([
            [0.0, 1.0],
            [0.5, 0.0],
        ]),
    )

def test_sensitivity_cell_level_set_edges_connects_2d_crossings() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    edges = sensitivity_cell_level_set_edges(
        data,
        target=20.0,
    )

    assert len(edges) == 1
    assert edges[0].shape == (2, 2)

    np.testing.assert_allclose(
        sorted(edges[0], key=lambda point: tuple(point)),
        np.array([
            [0.0, 1.0],
            [0.5, 0.0],
        ]),
    )

def test_sensitivity_cell_level_set_points_supports_3d_cells() -> None:
    data = SensitivityData(
        values=np.array([
            [[0.0, 1.0],
             [1.0, 2.0]],

            [[1.0, 2.0],
             [2.0, 3.0]],
        ]),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    points = sensitivity_cell_level_set_points(
        data,
        target=1.0,
    )

    assert len(points) == 1
    assert points[0].shape[1] == 3
    assert len(points[0]) > 2

def test_sensitivity_level_set_vertices_deduplicates_cell_points() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    vertices = sensitivity_level_set_vertices(
        data,
        target=20.0,
    )

    assert vertices.shape == (2, 2)

    np.testing.assert_allclose(
        sorted(vertices, key=lambda point: tuple(point)),
        np.array([
            [0.0, 1.0],
            [0.5, 0.0],
        ]),
    )

def test_sensitivity_level_set_edges_builds_global_2d_topology() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    edges = sensitivity_level_set_edges(
        data,
        target=20.0,
    )

    assert edges.shape == (1, 2)

    np.testing.assert_array_equal(
        edges,
        np.array([[0, 1]]),
    )

def test_build_continuous_level_set_geometry_returns_geometry() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    geometry = build_continuous_level_set_geometry(
        data,
        target=20.0,
    )

    assert isinstance(geometry, Geometry)
    assert geometry.vertices.shape == (2, 2)
    assert geometry.edges.shape == (1, 2)

    np.testing.assert_array_equal(
        geometry.edges,
        np.array([[0, 1]]),
    )

def test_response_grid_data_reconstructs_sensitivity_data() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)
    reconstructed = response_grid_data(response)

    np.testing.assert_array_equal(
        reconstructed.values,
        data.values,
    )
    assert reconstructed.parameter_names == data.parameter_names

    for actual, expected in zip(
        reconstructed.axes,
        data.axes,
    ):
        np.testing.assert_array_equal(actual, expected)

def test_sensitivity_cells_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.arange(9.0).reshape(3, 3),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    response = build_response_geometry(data)

    np.testing.assert_array_equal(
        sensitivity_cells(response),
        sensitivity_cells(data),
    )

def test_sensitivity_cell_facets_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.arange(9.0).reshape(3, 3),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    response = build_response_geometry(data)

    np.testing.assert_array_equal(
        sensitivity_cell_facets(response),
        sensitivity_cell_facets(data),
    )

def test_sensitivity_cell_facet_adjacency_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.arange(9.0).reshape(3, 3),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    response = build_response_geometry(data)

    facets_data, adjacency_data = sensitivity_cell_facet_adjacency(data)
    facets_response, adjacency_response = (
        sensitivity_cell_facet_adjacency(response)
    )

    np.testing.assert_array_equal(
        facets_response,
        facets_data,
    )
    assert adjacency_response == adjacency_data

def test_sensitivity_cell_neighbors_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.arange(9.0).reshape(3, 3),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    response = build_response_geometry(data)

    assert (
        sensitivity_cell_neighbors(response)
        == sensitivity_cell_neighbors(data)
    )

def test_sensitivity_boundary_facets_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.arange(6.0).reshape(3, 2),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    np.testing.assert_array_equal(
        sensitivity_boundary_facets(response),
        sensitivity_boundary_facets(data),
    )

def test_sensitivity_cell_level_set_points_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    response_points = sensitivity_cell_level_set_points(
        response,
        target=20.0,
    )
    data_points = sensitivity_cell_level_set_points(
        data,
        target=20.0,
    )

    assert len(response_points) == len(data_points)

    for actual, expected in zip(response_points, data_points):
        np.testing.assert_allclose(actual, expected)

def test_sensitivity_level_set_vertices_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    np.testing.assert_allclose(
        sensitivity_level_set_vertices(
            response,
            target=20.0,
        ),
        sensitivity_level_set_vertices(
            data,
            target=20.0,
        ),
    )

def test_sensitivity_level_set_edges_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    np.testing.assert_array_equal(
        sensitivity_level_set_edges(
            response,
            target=20.0,
        ),
        sensitivity_level_set_edges(
            data,
            target=20.0,
        ),
    )

def test_build_continuous_level_set_geometry_accepts_response_geometry() -> None:
    data = SensitivityData(
        values=np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    geometry_from_response = build_continuous_level_set_geometry(
        response,
        target=20.0,
    )
    geometry_from_data = build_continuous_level_set_geometry(
        data,
        target=20.0,
    )

    np.testing.assert_allclose(
        geometry_from_response.vertices,
        geometry_from_data.vertices,
    )
    np.testing.assert_array_equal(
        geometry_from_response.edges,
        geometry_from_data.edges,
    )

def test_sensitivity_level_set_edges_connect_3d_level_set_vertices() -> None:
    data = SensitivityData(
        values=np.array([
            [
                [0.0, 1.0],
                [1.0, 2.0],
            ],
            [
                [1.0, 2.0],
                [2.0, 3.0],
            ],
        ]),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    level_set = continuous_response_level_set(
        response,
        target=1.0,
    )

    assert len(level_set.geometry.vertices) == 6
    assert len(level_set.geometry.edges) > 0

    assert np.all(
        level_set.geometry.edges[:, 0]
        < len(level_set.geometry.vertices)
    )
    assert np.all(
        level_set.geometry.edges[:, 1]
        < len(level_set.geometry.vertices)
    )

def test_sensitivity_level_set_facets_returns_3d_level_set_points() -> None:
    data = SensitivityData(
        values=np.array([
            [
                [0.0, 1.0],
                [1.0, 2.0],
            ],
            [
                [1.0, 2.0],
                [2.0, 3.0],
            ],
        ]),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    response = build_response_geometry(data)

    facets = sensitivity_level_set_facets(
        response,
        target=1.0,
    )

    assert len(facets) == 1
    assert facets[0].shape[1] == 3
    assert len(facets[0]) == 6

def test_unique_sensitivity_level_set_facets_deduplicates_facets() -> None:
    data = SensitivityData(
        values=np.array([
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ]),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    response = build_response_geometry(data)

    facets = unique_sensitivity_level_set_facets(
        response,
        target=1.0,
    )

    assert len(facets) == 1
    assert np.allclose(
        facets[0],
        np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ]),
    )

    for facet in facets:
        assert facet.shape[1] == 2

def test_sensitivity_cell_level_set_boundary_orders_3d_points():
    data = SensitivityData(
        values=np.array(
            [
                [[0.0, 1.0], [1.0, 2.0]],
                [[1.0, 2.0], [2.0, 3.0]],
            ]
        ),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    boundaries = sensitivity_cell_level_set_boundary(
        data,
        target=1.0,
    )

    assert len(boundaries) == 1
    assert boundaries[0].shape[1] == 3
    assert len(boundaries[0]) >= 3

def test_sensitivity_cell_level_set_boundary_2d_returns_segment():
    data = SensitivityData(
        values=np.array(
            [
                [0.0, 2.0],
                [2.0, 0.0],
            ]
        ),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    boundaries = sensitivity_cell_level_set_boundary(
        data,
        target=1.0,
    )

    assert len(boundaries) == 1
    assert boundaries[0].shape == (4, 2)

    expected = np.array([
        [0.0, 0.5],
        [0.5, 0.0],
        [0.5, 1.0],
        [1.0, 0.5],
    ])

    for point in expected:
        assert any(
            np.allclose(
                point,
                existing,
                atol=1e-9,
                rtol=0,
            )
            for existing in boundaries[0]
        )

def test_sensitivity_cell_level_set_boundary_4d_returns_all_crossings():
    data = SensitivityData(
        values=np.array(
            [
                [
                    [[0.0, 1.0], [1.0, 2.0]],
                    [[1.0, 2.0], [2.0, 3.0]],
                ],
                [
                    [[1.0, 2.0], [2.0, 3.0]],
                    [[2.0, 3.0], [3.0, 4.0]],
                ],
            ]
        ),
        parameter_names=("x", "y", "z", "w"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    boundaries = sensitivity_cell_level_set_boundary(
        data,
        target=2.0,
    )

    assert len(boundaries) == 1
    assert boundaries[0].shape[1] == 4
    assert len(boundaries[0]) >= 8
    assert np.allclose(
        np.sum(boundaries[0], axis=1),
        2.0,
    )
    assert len(np.unique(boundaries[0], axis=0)) == len(boundaries[0])

def test_sensitivity_cell_level_set_edges_4d_does_not_create_cyclic_edges():
    data = SensitivityData(
        values=np.array(
            [
                [
                    [[0.0, 1.0], [1.0, 2.0]],
                    [[1.0, 2.0], [2.0, 3.0]],
                ],
                [
                    [[1.0, 2.0], [2.0, 3.0]],
                    [[2.0, 3.0], [3.0, 4.0]],
                ],
            ]
        ),
        parameter_names=("x", "y", "z", "w"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    edges = sensitivity_cell_level_set_edges(
        data,
        target=2.0,
    )

    assert len(edges) == 0

def test_sensitivity_cell_level_set_edges_3d_forms_closed_boundary():
    data = SensitivityData(
        values=np.array(
            [
                [[0.0, 1.0], [1.0, 2.0]],
                [[1.0, 2.0], [2.0, 3.0]],
            ]
        ),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    edges = sensitivity_cell_level_set_edges(
        data,
        target=1.0,
    )

    assert len(edges) == 6

    assert all(
        edge.shape == (2, 3)
        for edge in edges
    )

def test_sensitivity_level_set_facet_indices_match_vertices():
    data = SensitivityData(
        values=np.array(
            [
                [[0.0, 1.0], [1.0, 2.0]],
                [[1.0, 2.0], [2.0, 3.0]],
            ]
        ),
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_level_set_facet_indices(
        data,
        target=1.0,
    )

    assert len(facets) == 1
    assert facets[0].shape == (6,)
    assert np.array_equal(
        facets[0],
        np.arange(6),
    )

def test_sensitivity_level_set_facet_adjacency_detects_shared_vertices():
    data = SensitivityData(
        values=np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    facets = sensitivity_level_set_facet_indices(
        data,
        target=1.0,
    )

    adjacency = sensitivity_level_set_facet_adjacency(
        data,
        target=1.0,
    )

    assert len(facets) == 2
    assert adjacency == [[1], [0]]

def test_sensitivity_level_set_components_groups_connected_facets():
    data = SensitivityData(
        values=np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    components = sensitivity_level_set_components(
        data,
        target=1.0,
    )

    assert len(components) == 1
    assert np.array_equal(
        components[0],
        np.array([0, 1]),
    )

def test_sensitivity_level_set_component_vertices_returns_component_geometry():
    data = SensitivityData(
        values=np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    components = sensitivity_level_set_component_vertices(
        data,
        target=1.0,
    )

    assert len(components) == 1
    assert components[0].shape == (7, 2)

def test_sensitivity_level_set_component_edges_returns_component_edges():
    data = SensitivityData(
        values=np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    components = sensitivity_level_set_component_edges(
        data,
        target=1.0,
    )

    assert len(components) == 1
    assert components[0].shape[1] == 2
    assert len(components[0]) > 0

def test_build_level_set_components_returns_local_geometry():
    data = SensitivityData(
        values=np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        ),
        parameter_names=("x", "y"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )

    components = build_level_set_components(
        data,
        target=1.0,
    )

    assert len(components) == 1
    assert components[0].vertices.shape == (7, 2)
    assert components[0].edges.shape[1] == 2
    assert len(components[0].edges) > 0

def test_level_set_geometry_reports_component_count():
    response = ResponseGeometry(
        geometry=Geometry(
            vertices=np.array([
                [0.0, 0.0],
                [1.0, 0.0],
            ]),
            edges=np.array([
                [0, 1],
            ]),
        ),
        outputs=np.array([
            0.0,
            1.0,
        ]),
        parameter_names=("x",),
        axes=(
            np.array([0.0, 1.0]),
        ),
    )

    level_set = continuous_response_level_set(
        response,
        target=0.5,
    )

    assert level_set.component_count == 0

def test_level_set_component_reports_vertex_and_edge_counts():
    component = LevelSetComponent(
        vertices=np.array([
            [0.0, 0.5],
            [0.5, 0.0],
            [0.5, 1.0],
            [1.0, 0.5],
        ]),
        edges=np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ]),
    )

    assert component.vertex_count == 4
    assert component.edge_count == 4

def test_sensitivity_cell_simplices() -> None:
    values = np.zeros((2, 2, 2))

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    simplices = sensitivity_cell_simplices(data)

    assert simplices.shape == (6, 4)
    assert len({tuple(simplex) for simplex in simplices}) == 6

def test_sensitivity_cell_simplices_scales_with_dimension() -> None:
    for dimensions in range(2, 6):
        shape = (2,) * dimensions

        data = SensitivityData(
            values=np.zeros(shape),
            parameter_names=tuple(
                f"x{axis}"
                for axis in range(dimensions)
            ),
            axes=tuple(
                np.array([0.0, 1.0])
                for _ in range(dimensions)
            ),
        )

        simplices = sensitivity_cell_simplices(data)

        assert simplices.shape == (
            __import__("math").factorial(dimensions),
            dimensions + 1,
        )

def test_sensitivity_simplex_level_set_points() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    intersections = sensitivity_simplex_level_set_points(
        data,
        1.0,
    )

    assert len(intersections) == 6
    assert all(points.shape[1] == 3 for points in intersections)
    assert all(len(points) >= 2 for points in intersections)

def test_sensitivity_simplex_level_set_facets() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_simplex_level_set_facets(
        data,
        1.0,
    )

    assert len(facets) == 6
    assert all(facet.shape == (3, 3) for facet in facets)

def test_unique_sensitivity_simplex_level_set_facets() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = unique_sensitivity_simplex_level_set_facets(
        data,
        1.0,
    )

    assert len(facets) == 6
    assert all(facet.shape == (3, 3) for facet in facets)

def test_sensitivity_simplex_level_set_facets_triangulates_quadrilateral() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_simplex_level_set_facets(
        data,
        1.5,
    )

    assert len(facets) == 12
    assert all(facet.shape == (3, 3) for facet in facets)

def test_sensitivity_simplex_level_set_facet_indices() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    facets = sensitivity_simplex_level_set_facet_indices(
        data,
        1.0,
    )

    assert all(
        facet.shape == (3,)
        for facet in facets
    )
    assert all(
        len(set(int(vertex) for vertex in facet)) == 3
        for facet in facets
    )
    
def test_sensitivity_simplex_level_set_facet_adjacency() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    adjacency = sensitivity_simplex_level_set_facet_adjacency(
        data,
        1.0,
    )

    assert len(adjacency) == 6
    assert all(
        all(
            neighbor != facet
            for neighbor in neighbors
        )
        for facet, neighbors in enumerate(adjacency)
    )

def test_sensitivity_simplex_level_set_components() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    components = sensitivity_simplex_level_set_components(
        data,
        1.0,
    )

    assert len(components) == 1
    assert components[0].shape == (6,)
    assert np.array_equal(
        components[0],
        np.arange(6),
    )

def test_build_simplex_level_set_geometry() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    data = SensitivityData(
        values=values,
        parameter_names=("x", "y", "z"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    geometry = build_simplex_level_set_geometry(
        data,
        1.0,
    )

    assert geometry.vertices.shape == (6, 3)
    assert geometry.edges.shape == (6, 2)

def test_continuous_response_simplex_level_set() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    response = build_response_geometry(
        SensitivityData(
            values=values,
            parameter_names=("x", "y", "z"),
            axes=(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
            ),
        )
    )

    level_set = continuous_response_simplex_level_set(
        response,
        1.0,
    )

    assert isinstance(level_set, LevelSetGeometry)
    assert level_set.target == 1.0
    assert level_set.geometry.vertices.shape == (6, 3)
    assert level_set.geometry.edges.shape == (6, 2)
    assert level_set.component_count == 1

def test_continuous_response_simplex_level_set_non_degenerate() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 2.0], [2.0, 3.0]],
        ]
    )

    response = build_response_geometry(
        SensitivityData(
            values=values,
            parameter_names=("x", "y", "z"),
            axes=(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
            ),
        )
    )

    level_set = continuous_response_simplex_level_set(
        response,
        1.5,
    )

    assert level_set.component_count == 1
    assert level_set.geometry.vertices.shape[1] == 3
    assert level_set.geometry.edges.shape[1] == 2

    facets = sensitivity_simplex_level_set_facet_indices(
        response,
        1.5,
    )

    assert len(facets) > 0
    assert all(
        len(facet) == 3
        for facet in facets
    )

