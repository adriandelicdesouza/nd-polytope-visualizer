import numpy as np
import pytest

from nd_geometry import Hypercube


@pytest.mark.parametrize("dimensions", range(1, 8))
def test_hypercube_vertex_count(dimensions):
    cube = Hypercube(dimensions)
    assert cube.vertices.shape == (2**dimensions, dimensions)


def test_hypercube_vertices_are_valid():
    cube = Hypercube(4)

    assert np.all(np.isin(cube.vertices, [-1.0, 1.0]))


def test_hypercube_size():
    cube = Hypercube(3, size=2.5)

    assert np.all(np.isin(cube.vertices, [-2.5, 2.5]))


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        Hypercube(0)


def test_invalid_size():
    with pytest.raises(ValueError):
        Hypercube(3, size=0)
