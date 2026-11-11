"""One call from ticker to answer: inputs from filings, beliefs, reverse DCF, backtest, and a report.

    python -m nd_geometry.company NFLX --build examples/specs/nflx_revenue_build.json \
        --set wacc=0.095,0.115,0.135 --out outputs/nflx.html

Needs ND_SEC_USER_AGENT (the SEC requires a contact) unless --facts-file is given.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backtest import BacktestResult, backtest_company
from .correlation import Correlation, default_dcf_correlation, historical_correlation
from .dcf import company_model
from .density import DensityResult, belief_density
from .filings import (SEC_FACTS_URL, CompanyInputs, Fetch, cached_get, company_from_facts, http_get,
                      resolve_company)
from .model import Model
from .prices import PriceHistory, fetch_prices
from .reverse import ReverseDCF, reverse_dcf
from .revenue_build import RevenueBuild, SegmentBuild


@dataclass(frozen=True)
class CompanyAnalysis:
    company: CompanyInputs
    model: Model
    build: RevenueBuild
    market_price: float
    price_source: str
    result: DensityResult  # beliefs as chosen (correlated when a correlation is used)
    independent: DensityResult | None  # the same beliefs with independent drivers, for comparison
    correlation: Correlation | None
    reverse: ReverseDCF
    backtest: BacktestResult | None

    def lines(self) -> list[str]:
        q = self.result.quantiles("implied_price")
        out = [self.company.table(), "",
               f"Model: {self.model.name}, {len(self.model.inputs)} drivers; market price ${self.market_price:,.2f} ({self.price_source})",
               "Belief price p5/p25/p50/p75/p95: " + "  ".join(f"${v:,.2f}" for v in q.values()),
               "Correlation: " + ("independent drivers" if self.correlation is None else "Gaussian copula over the drivers")]
        if self.independent is not None:
            out.append("Price sd, independent vs correlated: "
                       f"${self.independent.column('implied_price').std():,.2f} vs ${self.result.column('implied_price').std():,.2f}")
        out += ["", "Reverse DCF", *self.reverse.lines()]
        if self.backtest is not None:
            out += ["", "Backtest of past belief distributions against later filings", *self.backtest.lines()]
        return out


def analyse_company(identifier: str, *, build: RevenueBuild | None = None, overrides=None, years: int = 10,
                    draws: int = 20_000, seed: int = 0, correlation: str = "prior", backtest: bool = True,
                    fetch: Fetch = http_get, cache_dir=".cache", market_price: float | None = None,
                    raw: dict | None = None, prices: PriceHistory | None = None) -> CompanyAnalysis:
    """Full workflow for one company. `raw` (XBRL company facts) and `prices` can be supplied to run offline."""
    if raw is None:
        cik, ticker, _ = resolve_company(identifier, fetch, cache_dir)
        raw = json.loads(cached_get(SEC_FACTS_URL.format(cik=cik), fetch, cache_dir))
    else:
        cik, ticker = 0, str(identifier).upper()
    company = company_from_facts(raw, ticker=ticker, cik=cik, overrides=overrides)
    if market_price is None:
        prices = prices or fetch_prices(ticker, cache_dir=cache_dir)
        market_price, source = prices.last_price, f"Yahoo Finance close {prices.last_date}"
    else:
        source = "supplied"
    model, build = company_model(company, build, years)
    names = model.input_names
    if correlation == "none":
        corr = None
    elif correlation == "prior":
        corr = default_dcf_correlation(names)
    elif correlation == "historical":
        corr = historical_correlation(company.history, names, default_dcf_correlation(names))
    else:
        raise ValueError("correlation must be none, prior or historical")
    result = belief_density(model, draws=draws, seed=seed, correlation=corr)
    independent = belief_density(model, draws=draws, seed=seed) if corr is not None else None
    bt = backtest_company(raw, ticker, draws=2000, seed=seed) if backtest else None
    return CompanyAnalysis(company, model, build, float(market_price), source, result, independent, corr,
                           reverse_dcf(result, float(market_price)), bt)

def build_report(a: CompanyAnalysis, path) -> Path:
    """Write one HTML page: price density against the market price, what the price implies, backtest PITs."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {"type": "table"}]], vertical_spacing=0.14, subplot_titles=(
        f"{a.company.ticker}: where the DCF price could be vs the market price",
        "What the market price implies: shift in each driver, in belief standard deviations",
        "Backtest: where later outcomes landed in past forecasts (flat = calibrated)",
        "Market-implied drivers"))
    for res, name, colour in ((a.independent, "independent drivers", "#5c6370"),
                              (a.result, "correlated drivers" if a.correlation is not None else "independent drivers", "#f0a500")):
        if res is not None:
            grid, density = res.density("implied_price")
            fig.add_trace(go.Scatter(x=grid, y=density, name=name, line=dict(color=colour, width=2)), row=1, col=1)
    fig.add_vline(x=a.market_price, line=dict(color="#e74c3c", width=2), annotation_text=f"market ${a.market_price:,.2f}",
                  row=1, col=1)
    base = float(a.model.evaluate(a.model.base_case())["implied_price"])
    fig.add_vline(x=base, line=dict(color="#2ecc71", dash="dot"), annotation_text=f"base ${base:,.2f}",
                  annotation_position="bottom right", row=1, col=1)
    implied = sorted((d for d in a.reverse.implied.values() if np.isfinite(d.implied_mean)), key=lambda d: d.shift_in_sd)
    fig.add_trace(go.Bar(x=[d.shift_in_sd for d in implied], y=[d.name for d in implied], orientation="h",
                         marker_color=["#3498db" if d.shift_in_sd < 0 else "#f0a500" for d in implied], showlegend=False),
                  row=1, col=2)
    if a.backtest is not None and a.backtest.rows:
        fig.add_trace(go.Histogram(x=[r.pit for r in a.backtest.rows], xbins=dict(start=0, end=1, size=0.1),
                                   histnorm="probability", marker_color="#f0a500", showlegend=False), row=2, col=1)
        fig.add_hline(y=0.1, line=dict(color="#2ecc71", dash="dot"), row=2, col=1)
    rows = sorted(a.reverse.implied.values(), key=lambda d: -abs(d.shift_in_sd))
    fig.add_trace(go.Table(
        header=dict(values=["driver", "belief mean", "implied mean", "p10 to p90", "shift (sd)"], fill_color="#1b1d22"),
        cells=dict(values=[[d.name for d in rows], [f"{d.prior_mean:.4f}" for d in rows], [f"{d.implied_mean:.4f}" for d in rows],
                           [f"{d.implied_p10:.4f} to {d.implied_p90:.4f}" for d in rows],
                           [f"{d.shift_in_sd:+.2f}" for d in rows]], fill_color="#0f1115")), row=2, col=2)
    r = a.reverse
    fig.update_xaxes(title_text="implied price $/share", row=1, col=1)
    fig.update_xaxes(title_text="PIT (share of forecast draws below the outcome)", row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0b0d", plot_bgcolor="#0f1115", height=1000,
        font=dict(family="IBM Plex Mono, monospace", color="#c8cdd6", size=11),
        title=dict(text=f"{a.company.name or a.company.ticker}: belief mass at or above the market price {r.prob_at_or_above:.1%}"
                        f" (market at the {r.belief_percentile:.0%} percentile of beliefs)", font=dict(color="#f0a500")))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m nd_geometry.company", description=__doc__.split("\n\n")[0])
    p.add_argument("ticker")
    p.add_argument("--build", help="segment revenue build JSON (default: top-down growth with fade)")
    p.add_argument("--set", action="append", default=[], metavar="DRIVER=LOW,BASE,HIGH", help="override a driver's range")
    p.add_argument("--years", type=int, default=10, help="forecast years for the top-down build")
    p.add_argument("--draws", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--correlation", choices=("none", "prior", "historical"), default="prior")
    p.add_argument("--no-backtest", action="store_true")
    p.add_argument("--market-price", type=float, help="use this price instead of fetching one")
    p.add_argument("--facts-file", help="XBRL company facts JSON to use instead of downloading")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--out", help="write an HTML report here")
    args = p.parse_args(argv)
    overrides = {}
    for item in args.set:
        name, sep, values = item.partition("=")
        try:
            low, base, high = (float(x) for x in values.split(","))
        except ValueError:
            raise SystemExit(f"bad --set {item!r}; use DRIVER=LOW,BASE,HIGH") from None
        if not sep:
            raise SystemExit(f"bad --set {item!r}; use DRIVER=LOW,BASE,HIGH")
        overrides[name] = (low, base, high)
    analysis = analyse_company(
        args.ticker, build=SegmentBuild.from_json(args.build) if args.build else None, overrides=overrides,
        years=args.years, draws=args.draws, seed=args.seed, correlation=args.correlation,
        backtest=not args.no_backtest, market_price=args.market_price, cache_dir=args.cache_dir,
        raw=json.loads(Path(args.facts_file).read_text()) if args.facts_file else None)
    print("\n".join(analysis.lines()))
    if args.out:
        print(f"\nReport written to {build_report(analysis, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())