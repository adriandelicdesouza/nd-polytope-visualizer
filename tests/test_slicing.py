import numpy as np
import pytest

from nd_geometry import Hypercube
from nd_geometry.slicing import Geometry, slice_geometry
from nd_geometry.topology import hypercube_edges


def geometry_from_hypercube(dimensions: int) -> Geometry:
    cube = Hypercube(dimensions)

    return Geometry(
        vertices=cube.vertices,
        edges=hypercube_edges(cube.vertices),
    )


def test_3d_cube_slice():
    geometry = geometry_from_hypercube(3)

    sliced = slice_geometry(
        geometry,
        axis=2,
        value=0.0,
    )

    assert sliced.vertices.shape == (4, 2)
    assert sliced.edges.shape == (4, 2)


def test_4d_tesseract_slice():
    geometry = geometry_from_hypercube(4)

    sliced = slice_geometry(
        geometry,
        axis=3,
        value=0.0,
    )

    assert sliced.vertices.shape == (8, 3)
    assert sliced.edges.shape == (12, 2)


def test_6d_hypercube_slice():
    geometry = geometry_from_hypercube(6)

    sliced = slice_geometry(
        geometry,
        axis=5,
        value=0.0,
    )

    assert sliced.vertices.shape == (32, 5)
    assert sliced.edges.shape == (80, 2)


def test_slice_at_existing_boundary():
    geometry = geometry_from_hypercube(3)

    sliced = slice_geometry(
        geometry,
        axis=2,
        value=1.0,
    )

    assert sliced.vertices.shape == (4, 2)
    assert sliced.edges.shape == (4, 2)


def test_slice_reduces_dimension():
    for dimensions in range(2, 8):
        geometry = geometry_from_hypercube(dimensions)

        sliced = slice_geometry(
            geometry,
            axis=dimensions - 1,
            value=0.0,
        )

        assert sliced.vertices.shape[1] == dimensions - 1


def test_invalid_axis():
    geometry = geometry_from_hypercube(4)

    with pytest.raises(ValueError):
        slice_geometry(geometry, axis=4, value=0)


def test_cannot_slice_1d():
    geometry = geometry_from_hypercube(1)

    with pytest.raises(ValueError):
        slice_geometry(geometry, axis=0, value=0)
