from .polytopes import Hypercube
from .projection import (
    deduplicate_geometry,
    linear_project,
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
    "project",
    "rotate",
    "slice_geometry",
]