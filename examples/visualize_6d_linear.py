import numpy as np

from nd_geometry import (
    Geometry,
    Hypercube,
    PlotlyRenderer,
    linear_project,
    orthogonal_projection_matrix,
    rotate,
    slice_geometry,
    rotate_projection_basis,
    rotate_projection_basis_sequence,
    Projection,
)

from nd_geometry.topology import hypercube_edges

PROJECTION_SEED = 7

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
projection = Projection(
    target_dimensions=3,
    seed=PROJECTION_SEED,
    rotations=(
        (0, 4, np.pi / 6),
        (1, 3, np.pi / 8),
    ),
)

matrix = projection.matrix(source_dimensions=5)

geometry = linear_project(
    geometry,
    matrix,
)

print(f"Vertices: {len(geometry.vertices)}")
print(f"Edges:    {len(geometry.edges)}")

PlotlyRenderer().render(geometry)