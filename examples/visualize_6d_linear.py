import numpy as np

from nd_geometry import (
    Geometry,
    Hypercube,
    PlotlyRenderer,
    linear_project,
    rotate,
    slice_geometry,
)
from nd_geometry.topology import hypercube_edges


cube = Hypercube(6)

geometry = Geometry(
    vertices=cube.vertices,
    edges=hypercube_edges(cube.vertices),
)

# Rotate in the 0-5 plane.
rotated_vertices = rotate(
    geometry.vertices,
    axis_a=0,
    axis_b=5,
    angle=np.pi / 6,
)

geometry = Geometry(
    vertices=rotated_vertices,
    edges=geometry.edges,
)

# Slice x5 = 0.
geometry = slice_geometry(
    geometry,
    axis=5,
    value=0.0,
)

# Project all 5 dimensions into 3D.
matrix = np.array([
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 1.0],
])

geometry = linear_project(
    geometry,
    matrix,
)

print(f"Vertices: {len(geometry.vertices)}")
print(f"Edges:    {len(geometry.edges)}")

PlotlyRenderer().render(geometry)