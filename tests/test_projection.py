import numpy as np
import pytest

from nd_geometry.projection import (
    linear_project,
    project,
)
from nd_geometry.slicing import Geometry



def make_geometry(dimensions: int) -> Geometry:
    vertices = np.arange(
        dimensions * 3,
        dtype=float,
    ).reshape(-1, dimensions)

    edges = np.array([
        [0, 1],
        [1, 2],
    ])

    return Geometry(
        vertices=vertices,
        edges=edges,
    )

    
def test_4d_to_3d():
    geometry = Geometry(
        vertices=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]),
        edges=np.array([
            [0, 1],
        ]),
    )

    projected = project(geometry, 3)

    expected = np.array([
        [1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0],
    ])

    np.testing.assert_allclose(
        projected.vertices,
        expected,
    )

    np.testing.assert_array_equal(
        projected.edges,
        geometry.edges,
    )


def test_custom_axes():
    geometry = Geometry(
        vertices=np.array([
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ]),
        edges=np.empty((0, 2), dtype=int),
    )

    projected = project(
        geometry,
        target_dimensions=3,
        axes=(0, 3, 5),
    )

    np.testing.assert_allclose(
        projected.vertices,
        [[10.0, 40.0, 60.0]],
    )


def test_topology_is_preserved():
    geometry = Geometry(
        vertices=np.zeros((4, 6)),
        edges=np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ]),
    )

    projected = project(
        geometry,
        target_dimensions=3,
    )

    np.testing.assert_array_equal(
        projected.edges,
        geometry.edges,
    )


@pytest.mark.parametrize(
    "source_dimensions,target_dimensions",
    [
        (3, 2),
        (4, 3),
        (5, 3),
        (6, 3),
        (8, 4),
        (10, 5),
        (20, 3),
    ],
)
def test_arbitrary_dimension_reduction(
    source_dimensions,
    target_dimensions,
):
    geometry = Geometry(
        vertices=np.zeros((10, source_dimensions)),
        edges=np.empty((0, 2), dtype=int),
    )

    projected = project(
        geometry,
        target_dimensions,
    )

    assert projected.vertices.shape == (
        10,
        target_dimensions,
    )

    assert projected.edges.shape == (
        0,
        2,
    )


def test_invalid_target_dimension():
    geometry = make_geometry(4)

    with pytest.raises(ValueError):
        project(geometry, 5)


def test_invalid_target_dimension_zero():
    geometry = make_geometry(4)

    with pytest.raises(ValueError):
        project(geometry, 0)


def test_invalid_axis():
    geometry = make_geometry(4)

    with pytest.raises(ValueError):
        project(
            geometry,
            target_dimensions=3,
            axes=(0, 1, 4),
        )


def test_duplicate_axes():
    geometry = make_geometry(4)

    with pytest.raises(ValueError):
        project(
            geometry,
            target_dimensions=3,
            axes=(0, 0, 1),
        )


def test_wrong_number_of_axes():
    geometry = make_geometry(4)

    with pytest.raises(ValueError):
        project(
            geometry,
            target_dimensions=3,
            axes=(0, 1),
        )

def test_deduplicate_projected_geometry():
    geometry = Geometry(
        vertices=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        edges=np.array([
            [0, 2],
            [1, 3],
        ]),
    )

    from nd_geometry.projection import deduplicate_geometry

    deduplicated = deduplicate_geometry(geometry)

    assert len(deduplicated.vertices) == 2

    np.testing.assert_array_equal(
        deduplicated.edges,
        np.array([[0, 1]]),
    )

def test_linear_projection():
    geometry = Geometry(
        vertices=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ]),
        edges=np.array([
            [0, 1],
        ]),
    )

    matrix = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])

    projected = linear_project(
        geometry,
        matrix,
    )

    expected = np.array([
        [5.0, 6.0, 7.0],
        [10.0, 12.0, 14.0],
    ])

    np.testing.assert_allclose(
        projected.vertices,
        expected,
    )

    np.testing.assert_array_equal(
        projected.edges,
        geometry.edges,
    )

def test_orthogonal_projection_matrix():
    from nd_geometry.projection import orthogonal_projection_matrix

    matrix = orthogonal_projection_matrix(
        5,
        3,
        seed=42,
    )

    assert matrix.shape == (3, 5)

    np.testing.assert_allclose(
        matrix @ matrix.T,
        np.eye(3),
        atol=1e-12,
    )

    matrix_again = orthogonal_projection_matrix(
        5,
        3,
        seed=42,
    )

    np.testing.assert_allclose(
        matrix,
        matrix_again,
    )

def test_orthogonal_matrix_works_with_linear_projection():
    from nd_geometry.projection import (
        linear_project,
        orthogonal_projection_matrix,
    )

    geometry = Geometry(
        vertices=np.eye(5),
        edges=np.empty((0, 2), dtype=int),
    )

    matrix = orthogonal_projection_matrix(
        source_dimensions=5,
        target_dimensions=3,
        seed=42,
    )

    projected = linear_project(
        geometry,
        matrix,
    )

    assert projected.vertices.shape == (5, 3)

    np.testing.assert_allclose(
        projected.vertices,
        matrix.T,
        atol=1e-12,
    )

def test_rotate_projection_basis_preserves_orthogonality():
    from nd_geometry.projection import (
        orthogonal_projection_matrix,
        rotate_projection_basis,
    )

    matrix = orthogonal_projection_matrix(
        source_dimensions=5,
        target_dimensions=3,
        seed=42,
    )

    rotated = rotate_projection_basis(
        matrix,
        axis_a=0,
        axis_b=4,
        angle=np.pi / 4,
    )

    assert rotated.shape == (3, 5)

    np.testing.assert_allclose(
        rotated @ rotated.T,
        np.eye(3),
        atol=1e-12,
    )

def test_rotate_projection_basis_sequence():
    from nd_geometry.projection import (
        orthogonal_projection_matrix,
        rotate_projection_basis_sequence,
    )

    matrix = orthogonal_projection_matrix(
        source_dimensions=5,
        target_dimensions=3,
        seed=42,
    )

    rotated = rotate_projection_basis_sequence(
        matrix,
        [
            (0, 4, np.pi / 6),
            (1, 3, np.pi / 8),
        ],
    )

    np.testing.assert_allclose(
        rotated @ rotated.T,
        np.eye(3),
        atol=1e-12,
    )