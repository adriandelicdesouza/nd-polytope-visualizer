from __future__ import annotations

from nd_geometry.model import Input, Model, Output

YEARS = 5
BASE_REVENUE = 100.0
NET_DEBT = 200.0  # used to read an implied share price off enterprise value
SHARES = 50.0


def dcf_full(values: dict[str, float]) -> dict[str, float]:
    """Simplified multi-input DCF: revenue, margins, SG&A drift, capex, NWC, tax, WACC."""
    wacc = values["wacc"]
    terminal_growth = values["terminal_growth"]
    tax = values["tax_rate"]

    previous_revenue = BASE_REVENUE
    fcfs: list[float] = []
    for year in range(1, YEARS + 1):
        revenue = BASE_REVENUE * (1 + values["revenue_growth"]) ** year
        sga_share = values["sga_pct_base"] * (1 + values["sga_growth"]) ** year
        ebit = revenue * (values["gross_margin"] - sga_share - values["da_pct"])
        nopat = ebit * (1 - tax)
        depreciation = revenue * values["da_pct"]
        capex = revenue * values["capex_pct"]
        delta_nwc = values["nwc_pct"] * (revenue - previous_revenue)
        fcfs.append(nopat + depreciation - capex - delta_nwc)
        previous_revenue = revenue

    discount = [(1 + wacc) ** year for year in range(1, YEARS + 1)]
    pv_cash_flows = sum(f / d for f, d in zip(fcfs, discount))
    terminal_value = fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / discount[-1]

    enterprise_value = pv_cash_flows + pv_terminal
    return {
        "enterprise_value": enterprise_value,
        "pv_terminal_value": pv_terminal,
        "year5_free_cash_flow": fcfs[-1],
    }


MODEL = Model(
    name="DCF (ten inputs)",
    inputs=(
        Input("revenue_growth", "%", 0.02, 0.10, 0.06),
        Input("gross_margin", "%", 0.40, 0.60, 0.50),
        Input("sga_pct_base", "%", 0.15, 0.25, 0.20),
        Input("sga_growth", "%", -0.05, 0.05, 0.0),
        Input("da_pct", "%", 0.03, 0.06, 0.045),
        Input("capex_pct", "%", 0.03, 0.07, 0.05),
        Input("nwc_pct", "%", 0.05, 0.15, 0.10),
        Input("tax_rate", "%", 0.20, 0.30, 0.25),
        Input("wacc", "%", 0.08, 0.12, 0.10),
        Input("terminal_growth", "%", 0.01, 0.03, 0.02),
    ),
    outputs=(
        Output("enterprise_value", "$m"),
        Output("pv_terminal_value", "$m"),
        Output("year5_free_cash_flow", "$m"),
    ),
    function=dcf_full,
)
