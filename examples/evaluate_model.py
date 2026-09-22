from __future__ import annotations

import numpy as np

from nd_geometry.evaluation import evaluate_model
from nd_geometry.parameter_space import ParameterSpace


space = ParameterSpace(
    names=("growth", "margin", "wacc"),
    lower_bounds=(0.02, 0.10, 0.06),
    upper_bounds=(0.10, 0.25, 0.12),
)

samples = space.sample_grid(points_per_dimension=3)


def model(parameters: np.ndarray) -> float:
    growth, margin, wacc = parameters
    return (1 + growth) * margin / wacc


result = evaluate_model(samples, model)

for parameters, output in zip(
    result.samples,
    result.outputs,
):
    print(parameters, "->", output)