from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

from .analysis import AnalysisResult


def input_surface(
    result: AnalysisResult,
    x_input: str,
    y_input: str,
    output: str,
    fixed: dict[str, float] | None = None,
    points: int = 25,
) -> dict:
    """Evaluate one output over a 2D grid of two inputs, holding the others fixed.

    `fixed` overrides base values for the remaining inputs. Returns the axis values
    and a z array shaped (len(y), len(x)).
    """
    model = result.model
    names = model.input_names
    if x_input not in names or y_input not in names or x_input == y_input:
        raise ValueError("x_input and y_input must be two different model inputs")
    if output not in model.output_names:
        raise ValueError(f"unknown output {output!r}")

    values = model.base_case()
    for name, value in (fixed or {}).items():
        if name not in names:
            raise ValueError(f"unknown input {name!r} in fixed values")
        values[name] = value

    spec = {item.name: item for item in model.inputs}
    x_values = np.linspace(spec[x_input].low, spec[x_input].high, points)
    y_values = np.linspace(spec[y_input].low, spec[y_input].high, points)
    z = np.empty((points, points))

    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            trial = {**values, x_input: float(x_value), y_input: float(y_value)}
            z[row, col] = model.evaluate(trial)[output]

    return {"x": x_values, "y": y_values, "z": z, "x_input": x_input, "y_input": y_input, "output": output}


def surface_figure(surface: dict, title: str | None = None) -> go.Figure:
    """Build an interactive 3D surface figure from `input_surface` output."""
    figure = go.Figure(go.Surface(x=surface["x"], y=surface["y"], z=surface["z"], colorscale="Viridis"))
    figure.update_layout(
        title=title or f"{surface['output']} over {surface['x_input']} and {surface['y_input']}",
        scene={
            "xaxis_title": surface["x_input"],
            "yaxis_title": surface["y_input"],
            "zaxis_title": surface["output"],
        },
    )
    return figure


def envelope_projection(result: AnalysisResult, x_output: str, y_output: str) -> go.Figure:
    """Plot the envelope's 2D projection onto two outputs, with the hull outline."""
    names = result.model.output_names
    if x_output not in names or y_output not in names or x_output == y_output:
        raise ValueError("x_output and y_output must be two different model outputs")

    columns = [names.index(x_output), names.index(y_output)]
    points = result.outputs[:, columns]
    hull = ConvexHull(points)
    outline = np.append(hull.vertices, hull.vertices[0])

    base = result.base_outputs
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers",
                                marker={"size": 4, "opacity": 0.4}, name="samples"))
    figure.add_trace(go.Scatter(x=points[outline, 0], y=points[outline, 1], mode="lines",
                                line={"width": 2}, name="envelope"))
    figure.add_trace(go.Scatter(x=[base[x_output]], y=[base[y_output]], mode="markers",
                                marker={"size": 12, "symbol": "star"}, name="base case"))
    figure.update_layout(title="Output envelope projection", xaxis_title=x_output, yaxis_title=y_output)
    return figure


def write_html(figure: go.Figure, path: str | Path) -> None:
    """Write a figure to a standalone HTML file that works offline."""
    figure.write_html(str(path), include_plotlyjs=True)
