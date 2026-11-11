"""Company DCF: the cash-flow arithmetic and the Model that wraps it.

Conventions follow examples/dcf_v2_model.py: mid-year discounting, tax only on positive
EBIT, change in working capital on the revenue increment, terminal value discounted with the
final-year timing. `other_reinvest_pct` is the cash-conversion gap measured from filings
(reinvestment that capex and working capital do not capture, such as content spend).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .filings import CompanyInputs
from .model import Input, Model, Output
from .revenue_build import RevenueBuild, TopDownBuild

FINANCIAL_DRIVERS = ("ebit_margin", "tax_rate", "capex_pct", "da_pct", "nwc_pct", "other_reinvest_pct",
                     "wacc", "terminal_growth")
OUTPUTS = (Output("implied_price", "$/share"), Output("tv_share", "%"), Output("exit_multiple", "x EBITDA"))


def project(v: Mapping[str, float], revenues: Sequence[float]) -> dict[str, list[float]]:
    """Yearly revenue, EBIT, EBITDA and unlevered free cash flow (index 0 of revenue is the base year)."""
    margin, tax = v["ebit_margin"], v["tax_rate"]
    capex, da, nwc = v["capex_pct"], v["da_pct"], v["nwc_pct"]
    other = v.get("other_reinvest_pct", 0.0)
    out: dict[str, list[float]] = {"revenue": list(revenues), "ebit": [], "ebitda": [], "fcf": []}
    for prev, rev in zip(revenues[:-1], revenues[1:]):
        ebit = rev * margin
        nopat = ebit - max(ebit, 0.0) * tax
        out["ebit"].append(ebit)
        out["ebitda"].append(ebit + rev * da)
        out["fcf"].append(nopat + rev * da - rev * capex - nwc * (rev - prev) - rev * other)
    return out


def dcf_value(v: Mapping[str, float], revenues: Sequence[float], *, shares: float, net_debt: float,
              mid_year: bool = True) -> dict[str, float]:
    wacc, tg = v["wacc"], v["terminal_growth"]
    if wacc <= tg:
        raise ValueError("WACC must be greater than terminal growth.")
    path = project(v, revenues)
    n = len(path["fcf"])
    shift = 0.5 if mid_year else 0.0
    pv_fcf = sum(f / (1 + wacc) ** (t - shift) for t, f in enumerate(path["fcf"], 1))
    tv = path["fcf"][-1] * (1 + tg) / (wacc - tg)
    pv_tv = tv / (1 + wacc) ** (n - shift)
    ev = pv_fcf + pv_tv
    ebitda_n = path["ebitda"][-1]
    return {
        "implied_price": (ev - net_debt) / shares,
        "enterprise_value": ev,
        "tv_share": pv_tv / ev if ev > 0 else float("nan"),
        "exit_multiple": tv / ebitda_n if ebitda_n > 0 else float("nan"),
    }


def company_model(company: CompanyInputs, build: RevenueBuild | None = None, years: int = 5,
                  mid_year: bool = True) -> tuple[Model, RevenueBuild]:
    """A Model for one company: the revenue build's drivers first, then the financial drivers."""
    if build is None:
        g, f = company.drivers["revenue_growth"], company.drivers["growth_fade_to"]
        build = TopDownBuild(company.base_revenue, years, (g.low, g.base, g.high), (f.low, f.base, f.high))
    inputs: tuple[Input, ...] = build.inputs() + company.inputs(FINANCIAL_DRIVERS)
    shares, net_debt = company.shares, company.net_debt

    def function(v: dict[str, float]) -> dict[str, float]:
        out = dcf_value(v, build.revenues(v), shares=shares, net_debt=net_debt, mid_year=mid_year)
        return {o.name: out[o.name] for o in OUTPUTS}

    return Model(name=f"{company.ticker or company.name} DCF ({build.name})", inputs=inputs, outputs=OUTPUTS,
                 function=function), build