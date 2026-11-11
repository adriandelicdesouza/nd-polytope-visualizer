from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._dashboard_template import TEMPLATE
from .probability import MonteCarloResult


def dashboard_payload(result: MonteCarloResult, title: str | None = None) -> dict:
    """Return the JSON data the dashboard embeds: draws, columns and metadata."""
    model = result.model
    columns = (
        list(model.input_names)
        + list(model.output_names)
        + [f"occurred:{event.input_name}" for event in result.events]
    )
    matrix = np.column_stack([result.inputs, result.outputs, result.occurred.astype(float)])
    return {
        "title": title or f"{model.name} | probability surface",
        "model": model.name,
        "seed": result.seed,
        "draws": int(len(matrix)),
        "columns": columns,
        "inputs": [{"name": i.name, "unit": i.unit, "low": i.low, "high": i.high} for i in model.inputs],
        "outputs": [{"name": o.name, "unit": o.unit} for o in model.outputs],
        "events": [
            {"input": e.input_name, "probability": e.probability, "low": e.low, "high": e.high}
            for e in result.events
        ],
        "data": [[float(f"{value:.6g}") for value in row] for row in matrix],
    }


def write_dashboard(result: MonteCarloResult, path: str | Path, title: str | None = None) -> None:
    """Write a self-contained interactive HTML dashboard for a Monte Carlo result."""
    payload = dashboard_payload(result, title)
    html = TEMPLATE.replace("__TITLE__", payload["title"]).replace("__DATA__", json.dumps(payload))
    Path(path).write_text(html)
