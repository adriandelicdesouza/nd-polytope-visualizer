from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import Model
from .output_polytope import OutputPolytope, output_polytope
from .parameter_space import ParameterSpace


@dataclass(frozen=True)
class AnalysisResult:
    """Results of running a model template over its input ranges."""

    model: Model
    points_per_dimension: int
    outputs: np.ndarray
    polytope: OutputPolytope
    base_outputs: dict[str, float]
    base_inside: bool

    @property
    def output_bounds(self) -> dict[str, tuple[float, float]]:
        """Return (low, high) of each output across the envelope."""
        bounds = self.polytope.bounds()
        return {
            name: (float(bounds[index, 0]), float(bounds[index, 1]))
            for index, name in enumerate(self.model.output_names)
        }

    def base_position(self) -> dict[str, float]:
        """Return where the base case sits in each output's range (0 = low, 1 = high)."""
        return {
            name: (self.base_outputs[name] - low) / (high - low)
            for name, (low, high) in self.output_bounds.items()
        }

    def to_dict(self) -> dict:
        """Return a JSON-ready summary of the analysis."""
        bounds = self.output_bounds
        positions = self.base_position()
        return {
            "model": self.model.name,
            "points_per_dimension": self.points_per_dimension,
            "grid_size": int(len(self.outputs)),
            "inputs": [
                {"name": i.name, "unit": i.unit, "low": i.low, "high": i.high, "base": i.base}
                for i in self.model.inputs
            ],
            "outputs": [
                {
                    "name": o.name,
                    "unit": o.unit,
                    "low": bounds[o.name][0],
                    "high": bounds[o.name][1],
                    "base": self.base_outputs[o.name],
                    "base_position": positions[o.name],
                }
                for o in self.model.outputs
            ],
            "base_case_inside_envelope": self.base_inside,
            "polytope_vertices": int(len(self.polytope.vertices)),
            "polytope_volume": self.polytope.volume,
        }

    def save(self, path: str | Path) -> None:
        """Write the summary to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def analyse(model: Model, points_per_dimension: int = 5) -> AnalysisResult:
    """Sample the model over its input ranges and build its output envelope."""
    space = ParameterSpace(
        names=model.input_names,
        lower_bounds=tuple(item.low for item in model.inputs),
        upper_bounds=tuple(item.high for item in model.inputs),
    )
    grid = space.sample_grid(points_per_dimension=points_per_dimension)

    rows = [
        [model.evaluate(dict(zip(model.input_names, sample)))[name] for name in model.output_names]
        for sample in grid
    ]
    outputs = np.asarray(rows, dtype=float)
    polytope = output_polytope(outputs, output_names=model.output_names)

    base_outputs = model.evaluate(model.base_case())
    base_point = np.array([base_outputs[name] for name in model.output_names])

    return AnalysisResult(
        model=model,
        points_per_dimension=points_per_dimension,
        outputs=outputs,
        polytope=polytope,
        base_outputs=base_outputs,
        base_inside=polytope.contains(base_point),
    )
