from nd_geometry import Hypercube
from nd_geometry.topology import hypercube_edges, vertex_adjacency


def test_3d_cube_edge_count():
    cube = Hypercube(3)

    edges = hypercube_edges(cube.vertices)

    assert len(edges) == 12


def test_4d_hypercube_edge_count():
    cube = Hypercube(4)

    edges = hypercube_edges(cube.vertices)

    assert len(edges) == 32


def test_6d_hypercube_edge_count():
    cube = Hypercube(6)

    edges = hypercube_edges(cube.vertices)

    assert len(edges) == 192


def test_vertex_degree():
    cube = Hypercube(6)

    adjacency = vertex_adjacency(cube.vertices)

    assert all(len(neighbors) == 6 for neighbors in adjacency)
