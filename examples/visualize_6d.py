import numpy as np

from nd_geometry import (
    Geometry,
    Hypercube,
    PlotlyRenderer,
    deduplicate_geometry,
    project,
    rotate,
    slice_geometry,
)



from nd_geometry.topology import hypercube_edges


cube = Hypercube(6)

geometry = Geometry(
    vertices=cube.vertices,
    edges=hypercube_edges(cube.vertices),
)

# Rotate in the 0-5 coordinate plane.
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

# Reduce 5D → 3D.
geometry = project(
    geometry,
    target_dimensions=3,
)

geometry = deduplicate_geometry(geometry)

print(f"Vertices: {len(geometry.vertices)}")
print(f"Edges:    {len(geometry.edges)}")

PlotlyRenderer().render(geometry)