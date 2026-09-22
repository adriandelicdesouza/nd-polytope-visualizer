from __future__ import annotations

import numpy as np

from nd_geometry.evaluation import evaluate_model
from nd_geometry.parameter_space import ParameterSpace
from nd_geometry.sensitivity import SensitivityData


space = ParameterSpace(
    names=("growth", "margin", "wacc"),
    lower_bounds=(0.02, 0.10, 0.06),
    upper_bounds=(0.10, 0.25, 0.12),
)

points_per_dimension = 5

samples = space.sample_grid(
    points_per_dimension=points_per_dimension,
)


def model(parameters: np.ndarray) -> float:
    growth, margin, wacc = parameters
    return (1 + growth) * margin / wacc


result = evaluate_model(samples, model)

axes = tuple(
    np.linspace(
        lower,
        upper,
        points_per_dimension,
    )
    for lower, upper in zip(
        space.lower_bounds,
        space.upper_bounds,
    )
)

sensitivity = SensitivityData.from_evaluation(
    result=result,
    parameter_names=space.names,
    axes=axes,
)

print("Dimensions:", sensitivity.dimensions)
print("Shape:", sensitivity.values.shape)
print("Minimum:", sensitivity.values.min())
print("Maximum:", sensitivity.values.max())