from __future__ import annotations

from nd_geometry.model import Input, Model, Output

# Single-company DCF with explicit conventions, borrowed from equity-research-platform
# (cumulative discounting, mid-period timing, tax only on positive EBIT) and dcf-core/terminal
# (ΔNWC on a revenue basis). Money in $m. Company scale is fixed here so only drivers vary.
BASE_REVENUE = 1000.0
SHARES = 100.0
NET_DEBT = 0.0
YEARS = 5
MID_YEAR = True  # cash flows arrive mid-year; set False for end-of-year discounting


def dcf_v2(v: dict[str, float]) -> dict[str, float]:
    g, margin, tax = v["revenue_growth"], v["ebit_margin"], v["tax_rate"]
    capex, da, nwc = v["capex_pct"], v["da_pct"], v["nwc_pct"]
    wacc, tg = v["wacc"], v["terminal_growth"]
    if wacc <= tg:
        raise ValueError("WACC must be greater than terminal growth.")

    pv_fcf, rev_prev, fcf = 0.0, BASE_REVENUE, 0.0
    for t in range(1, YEARS + 1):
        rev = rev_prev * (1 + g)
        ebit = rev * margin
        nopat = ebit - max(ebit, 0.0) * tax          # no tax credit on losses
        fcf = nopat + rev * da - rev * capex - nwc * (rev - rev_prev)  # ΔNWC on the revenue increment
        timing = t - 0.5 if MID_YEAR else t
        pv_fcf += fcf / (1 + wacc) ** timing
        rev_prev = rev

    tv = fcf * (1 + tg) / (wacc - tg)
    pv_tv = tv / (1 + wacc) ** (YEARS - 0.5 if MID_YEAR else YEARS)
    ev = pv_fcf + pv_tv
    return {
        "implied_price": (ev - NET_DEBT) / SHARES,
        "enterprise_value": ev,
        "tv_share": pv_tv / ev if ev > 0 else float("nan"),
    }


MODEL = Model(
    name="Single-company DCF (v2 conventions)",
    inputs=(
        Input("revenue_growth", "%", 0.02, 0.10, 0.06),
        Input("ebit_margin", "%", 0.10, 0.30, 0.20),
        Input("tax_rate", "%", 0.20, 0.30, 0.25),
        Input("capex_pct", "%", 0.03, 0.07, 0.05),
        Input("da_pct", "%", 0.03, 0.06, 0.045),
        Input("nwc_pct", "%", 0.05, 0.15, 0.10),
        Input("wacc", "%", 0.08, 0.12, 0.10),
        Input("terminal_growth", "%", 0.01, 0.03, 0.02),
    ),
    outputs=(Output("implied_price", "$/share"), Output("enterprise_value", "$m"), Output("tv_share", "%")),
    function=dcf_v2,
)

if __name__ == "__main__":
    base = {"revenue_growth": 0.06, "ebit_margin": 0.20, "tax_rate": 0.25, "capex_pct": 0.05,
            "da_pct": 0.045, "nwc_pct": 0.10, "wacc": 0.10, "terminal_growth": 0.02}
    print(dcf_v2(base))
