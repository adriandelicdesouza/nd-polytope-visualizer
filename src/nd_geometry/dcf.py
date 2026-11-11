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


PER_YEAR = ("ebit_margin", "tax_rate", "capex_pct", "da_pct", "nwc_pct", "other_reinvest_pct")
WACC_PARTS = (Input("rf", "frac", 0.03, 0.055, 0.04), Input("equity_beta", "x", 0.8, 1.3, 1.0),
              Input("erp", "frac", 0.04, 0.07, 0.05), Input("cost_of_debt", "frac", 0.04, 0.07, 0.05),
              Input("debt_weight", "frac", 0.05, 0.30, 0.15))


def _at(v: Mapping[str, float], name: str, t: int, default: float | None = None) -> float:
    """Driver `name` in forecast year t: its year-specific input `<name>_y<t>` if present, else the single input."""
    key = f"{name}_y{t}"
    if key in v:
        return v[key]
    return v[name] if default is None else v.get(name, default)


def wacc_of(v: Mapping[str, float]) -> float:
    """WACC: given directly, or built from rf, beta, equity risk premium, cost of debt and debt weight."""
    if "wacc" in v:
        return v["wacc"]
    ke = v["rf"] + v["equity_beta"] * v["erp"]
    return (1 - v["debt_weight"]) * ke + v["debt_weight"] * v["cost_of_debt"] * (1 - _at(v, "tax_rate", 1))


def project(v: Mapping[str, float], revenues: Sequence[float]) -> dict[str, list[float]]:
    """Yearly revenue, EBIT, EBITDA and unlevered free cash flow (index 0 of revenue is the base year)."""
    out: dict[str, list[float]] = {"revenue": list(revenues), "ebit": [], "ebitda": [], "fcf": []}
    for t, (prev, rev) in enumerate(zip(revenues[:-1], revenues[1:]), 1):
        margin, tax = _at(v, "ebit_margin", t), _at(v, "tax_rate", t)
        capex, da, nwc = _at(v, "capex_pct", t), _at(v, "da_pct", t), _at(v, "nwc_pct", t)
        other = _at(v, "other_reinvest_pct", t, 0.0)
        ebit = rev * margin
        nopat = ebit - max(ebit, 0.0) * tax
        out["ebit"].append(ebit)
        out["ebitda"].append(ebit + rev * da)
        out["fcf"].append(nopat + rev * da - rev * capex - nwc * (rev - prev) - rev * other)
    return out


def dcf_value(v: Mapping[str, float], revenues: Sequence[float], *, shares: float, net_debt: float,
              mid_year: bool = True) -> dict[str, float]:
    wacc, tg = wacc_of(v), v["terminal_growth"]
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
        # At extreme corners of a wide box the enterprise value can be zero or negative; the output polytope
        # needs finite numbers, so the terminal-value share is 1 (all of the value is terminal) and the exit multiple 0.
        "tv_share": pv_tv / ev if ev > 0 else 1.0,
        "exit_multiple": tv / ebitda_n if ebitda_n > 0 else 0.0,
    }


def company_model(company: CompanyInputs, build: RevenueBuild | None = None, years: int = 5,
                  mid_year: bool = True, wacc_build: bool = False,
                  per_year: bool = False) -> tuple[Model, RevenueBuild]:
    """A Model for one company: the revenue build's drivers first, then the financial drivers.

    wacc_build: replace the single WACC input with rf, equity beta, equity risk premium, cost of debt and debt weight.
    per_year: give margin, tax, capex, D&A, working capital and reinvestment one input per forecast year.
    """
    if build is None:
        g, f = company.drivers["revenue_growth"], company.drivers["growth_fade_to"]
        build = TopDownBuild(company.base_revenue, years, (g.low, g.base, g.high), (f.low, f.base, f.high))
    if not company.shares == company.shares:  # NaN: the filings gave no share count
        raise ValueError(f"{company.ticker}: no share count in the filings, so a per-share DCF cannot be built")
    names = tuple(n for n in FINANCIAL_DRIVERS if not (wacc_build and n == "wacc"))
    if per_year:
        horizon = len(build.revenues({i.name: i.base for i in build.inputs()})) - 1
        financial: tuple[Input, ...] = tuple(
            Input(f"{i.name}_y{t}", i.unit, i.low, i.high, i.base) if i.name in PER_YEAR else i
            for i in company.inputs(names) for t in (range(1, horizon + 1) if i.name in PER_YEAR else (0,)))
    else:
        financial = company.inputs(names)
    inputs: tuple[Input, ...] = build.inputs() + financial + (WACC_PARTS if wacc_build else ())
    shares, net_debt = company.shares, company.net_debt

    def function(v: dict[str, float]) -> dict[str, float]:
        out = dcf_value(v, build.revenues(v), shares=shares, net_debt=net_debt, mid_year=mid_year)
        return {o.name: out[o.name] for o in OUTPUTS}

    return Model(name=f"{company.ticker or company.name} DCF ({build.name})", inputs=inputs, outputs=OUTPUTS,
                 function=function), build