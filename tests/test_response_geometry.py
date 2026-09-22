from __future__ import annotations

import numpy as np

from nd_geometry.response_geometry import (
    ResponseGeometry,
    build_response_geometry,
    sensitivity_outputs,
    sensitivity_vertices,
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