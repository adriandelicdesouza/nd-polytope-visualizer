import numpy as np
import pytest

from nd_geometry.rendering import Renderer
from nd_geometry.slicing import Geometry


def test_renderer_interface_exists():
    geometry = Geometry(
        vertices=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        edges=np.array([
            [0, 1],
        ]),
    )

    renderer = Renderer()

    with pytest.raises(NotImplementedError):
        renderer.render(geometry)

def test_plotly_renderer_requires_3d():
    from nd_geometry.rendering import PlotlyRenderer

    geometry = Geometry(
        vertices=np.zeros((4, 4)),
        edges=np.empty((0, 2), dtype=int),
    )

    renderer = PlotlyRenderer()

    with pytest.raises(ValueError):
        renderer.render(geometry)