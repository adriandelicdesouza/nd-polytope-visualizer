from .polytopes import Hypercube
from .projection import (
    deduplicate_geometry,
    linear_project,
    orthogonal_projection_matrix,
    project,
)
from .rendering import PlotlyRenderer, Renderer
from .rotations import rotate
from .slicing import Geometry, slice_geometry

__all__ = [
    "Geometry",
    "Hypercube",
    "PlotlyRenderer",
    "Renderer",
    "deduplicate_geometry",
    "linear_project",
    "orthogonal_projection_matrix",
    "project",
    "rotate",
    "slice_geometry",
]