from __future__ import annotations

import plotly.graph_objects as go

from .slicing import Geometry


class Renderer:
    """Base interface for rendering geometric objects."""

    def render(self, geometry: Geometry) -> None:
        """Render the supplied geometry."""
        raise NotImplementedError


class PlotlyRenderer(Renderer):
    """Interactive 3D renderer using Plotly."""

    def render(self, geometry: Geometry) -> None:
        if geometry.vertices.shape[1] != 3:
            raise ValueError("PlotlyRenderer requires 3D geometry")

        vertices = geometry.vertices
        edges = geometry.edges

        x = []
        y = []
        z = []

        for a, b in edges:
            x.extend([vertices[a, 0], vertices[b, 0], None])
            y.extend([vertices[a, 1], vertices[b, 1], None])
            z.extend([vertices[a, 2], vertices[b, 2], None])

        figure = go.Figure()

        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name="Edges",
            )
        )

        figure.add_trace(
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                name="Vertices",
            )
        )

        figure.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            showlegend=True,
        )

        figure.show()