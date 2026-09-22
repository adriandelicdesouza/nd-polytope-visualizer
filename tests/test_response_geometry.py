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
)

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