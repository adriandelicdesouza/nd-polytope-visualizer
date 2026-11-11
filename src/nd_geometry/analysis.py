from __future__ import annotations

import itertools
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
    method: str
    inputs: np.ndarray
    outputs: np.ndarray
    polytope: OutputPolytope
    base_outputs: dict[str, float]
    base_inside: bool
    points_per_dimension: int | None = None

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
            "method": self.method,
            "points_per_dimension": self.points_per_dimension,
            "sample_count": int(len(self.outputs)),
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


def evaluate_rows(model: Model, inputs: np.ndarray) -> np.ndarray:
    """Evaluate the model at each row of input values; return an (n, M) output array."""
    return np.asarray(
        [
            [model.evaluate(dict(zip(model.input_names, row)))[name] for name in model.output_names]
            for row in inputs
        ],
        dtype=float,
    )


def analyse(
    model: Model,
    points_per_dimension: int = 5,
    method: str = "grid",
    samples: int = 2000,
    seed: int = 0,
    include_corners: bool = True,
) -> AnalysisResult:
    """Sample the model over its input ranges and build its output envelope.

    method="grid" uses a Cartesian grid (cost grows as points**inputs).
    method="random" draws uniform samples with a fixed seed, plus the 2**inputs box
    corners when include_corners is set (extremes of monotone models sit there).
    """
    lows = np.array([item.low for item in model.inputs])
    highs = np.array([item.high for item in model.inputs])

    if method == "grid":
        space = ParameterSpace(
            names=model.input_names,
            lower_bounds=tuple(lows),
            upper_bounds=tuple(highs),
        )
        inputs = space.sample_grid(points_per_dimension=points_per_dimension)
        grid_points: int | None = points_per_dimension
    elif method == "random":
        rng = np.random.default_rng(seed)
        inputs = lows + rng.uniform(size=(samples, len(lows))) * (highs - lows)
        if include_corners:
            if len(lows) > 16:
                raise ValueError("include_corners is limited to 16 inputs (2**n corners)")
            corners = np.array(list(itertools.product(*zip(lows, highs))))
            inputs = np.vstack([corners, inputs])
        grid_points = None
    else:
        raise ValueError("method must be 'grid' or 'random'")

    outputs = evaluate_rows(model, inputs)
    polytope = output_polytope(outputs, output_names=model.output_names)

    base_outputs = model.evaluate(model.base_case())
    base_point = np.array([base_outputs[name] for name in model.output_names])

    return AnalysisResult(
        model=model,
        method=method,
        inputs=inputs,
        outputs=outputs,
        polytope=polytope,
        base_outputs=base_outputs,
        base_inside=polytope.contains(base_point),
        points_per_dimension=grid_points,
    )
