from __future__ import annotations

import numpy as np

from nd_geometry.evaluation import evaluate_vector_model
from nd_geometry.output_polytope import output_polytope
from nd_geometry.parameter_space import ParameterSpace


BASE_REVENUE = 100.0
TAX_RATE = 0.25
YEARS = 5

space = ParameterSpace(
    names=("revenue_growth", "ebit_margin", "wacc", "terminal_growth"),
    lower_bounds=(0.02, 0.10, 0.08, 0.01),
    upper_bounds=(0.10, 0.25, 0.12, 0.03),
)


def dcf(parameters: np.ndarray) -> np.ndarray:
    """Return [enterprise value, terminal value share of enterprise value]."""
    growth, margin, wacc, terminal_growth = parameters
    years = np.arange(1, YEARS + 1)

    revenue = BASE_REVENUE * (1 + growth) ** years
    free_cash_flow = revenue * margin * (1 - TAX_RATE)
    discount = (1 + wacc) ** years

    pv_cash_flows = np.sum(free_cash_flow / discount)
    terminal_value = free_cash_flow[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / discount[-1]

    enterprise_value = pv_cash_flows + pv_terminal
    return np.array([enterprise_value, pv_terminal / enterprise_value])


samples = space.sample_grid(points_per_dimension=5)
result = evaluate_vector_model(samples, dcf)
polytope = output_polytope(
    result.outputs,
    output_names=("enterprise_value", "terminal_share"),
)

print("Parameters:", space.names)
print("Grid samples:", len(samples))
print("Output bounds (min, max):")
for name, (low, high) in zip(polytope.output_names, polytope.bounds()):
    print(f"  {name}: {low:.4f} to {high:.4f}")
print("Polytope vertices:", len(polytope.vertices))
print("Polytope area:", round(polytope.volume, 4))
print("Base case inside:", polytope.contains(dcf(np.array([0.06, 0.175, 0.10, 0.02]))))
