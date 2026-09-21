import numpy as np
import pytest

from nd_geometry.projection import project
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