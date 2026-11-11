from __future__ import annotations

from nd_geometry.analysis import analyse
from nd_geometry.model import Input, Model, Output

BASE_REVENUE = 100.0
TAX_RATE = 0.25
YEARS = 5


def dcf(values: dict[str, float]) -> dict[str, float]:
    years = list(range(1, YEARS + 1))
    wacc = values["wacc"]
    terminal_growth = values["terminal_growth"]

    revenue = [BASE_REVENUE * (1 + values["revenue_growth"]) ** t for t in years]
    free_cash_flow = [r * values["ebit_margin"] * (1 - TAX_RATE) for r in revenue]
    discount = [(1 + wacc) ** t for t in years]

    pv_cash_flows = sum(f / d for f, d in zip(free_cash_flow, discount))
    terminal_value = free_cash_flow[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / discount[-1]

    enterprise_value = pv_cash_flows + pv_terminal
    return {"enterprise_value": enterprise_value, "terminal_share": pv_terminal / enterprise_value}


MODEL = Model(
    name="DCF (simplified)",
    inputs=(
        Input("revenue_growth", "%", 0.02, 0.10, 0.06),
        Input("ebit_margin", "%", 0.10, 0.25, 0.175),
        Input("wacc", "%", 0.08, 0.12, 0.10),
        Input("terminal_growth", "%", 0.01, 0.03, 0.02),
    ),
    outputs=(Output("enterprise_value", "$m"), Output("terminal_share", "%")),
    function=dcf,
)

if __name__ == "__main__":
    result = analyse(MODEL)
    for name, (low, high) in result.output_bounds.items():
        print(f"{name}: {low:.2f} to {high:.2f}")
    print("Base case inside envelope:", result.base_inside)
