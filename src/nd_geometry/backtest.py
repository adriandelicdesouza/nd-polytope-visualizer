"""Backtest: were past belief distributions calibrated against what the company then reported?

At each past fiscal year-end (the forecast origin) the company's inputs are rebuilt from only the
filings available on that 10-K's filing date. The belief distribution over the drivers is pushed
through the model to give a forecast distribution for revenue, EBIT and a pre-reinvestment cash
flow proxy (NOPAT + D&A - capex) one to three years ahead. The realised figure from later filings
is located in that distribution (its PIT, the share of draws at or below it). If beliefs are
calibrated the PITs are uniform: 80% of outcomes land in the central 80% band, and so on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
from scipy.stats import kstest, spearmanr

from .correlation import default_dcf_correlation
from .dcf import company_model, project
from .density import sample_beliefs
from .filings import (ANNUAL_FORMS, CAPEX, DA, EBIT, PRETAX, REVENUE, SEC_FACTS_URL, TAX, Facts, Fetch, _day,
                      cached_get, company_from_facts, http_get, resolve_company)
from .prices import PriceHistory

METRICS = ("revenue", "ebit", "cash_flow_proxy")
MIN_REVENUE_OBSERVATIONS = 6


@dataclass(frozen=True)
class BacktestRow:
    ticker: str
    origin: date  # fiscal year-end the forecast starts from
    as_of: date  # 10-K filing date: nothing filed later is used
    horizon: int  # fiscal years ahead
    metric: str
    realized: float  # $m
    p10: float
    p50: float
    p90: float
    pit: float


def pit(draws: np.ndarray, realized: float) -> float:
    """Share of draws at or below the realised value (ties count half)."""
    return float((np.sum(draws < realized) + 0.5 * np.sum(draws == realized)) / len(draws))


@dataclass(frozen=True)
class BacktestResult:
    rows: tuple[BacktestRow, ...]
    skipped: dict[str, str]

    def select(self, metric: str | None = None, horizon: int | None = None) -> list[BacktestRow]:
        return [r for r in self.rows if (metric is None or r.metric == metric) and (horizon is None or r.horizon == horizon)]

    def coverage(self, level: float = 0.8, metric: str | None = None, horizon: int | None = None) -> float:
        """Share of realised values inside the central `level` band of their forecast distribution."""
        rows = self.select(metric, horizon)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return float(np.mean([lo <= r.pit <= hi for r in rows])) if rows else float("nan")

    def mean_pit(self, metric: str | None = None, horizon: int | None = None) -> float:
        rows = self.select(metric, horizon)
        return float(np.mean([r.pit for r in rows])) if rows else float("nan")

    def ks_pvalue(self, metric: str | None = None, horizon: int | None = None) -> float:
        """KS test of the PITs against uniform. Rows overlap across horizons, so treat it as a rough guide."""
        rows = self.select(metric, horizon)
        return float(kstest([r.pit for r in rows], "uniform").pvalue) if len(rows) >= 5 else float("nan")

    def lines(self) -> list[str]:
        out = [f"{len(self.rows)} forecasts; central 50% band holds {self.coverage(0.5):.0%} (ideal 50%), "
               f"central 80% band holds {self.coverage(0.8):.0%} (ideal 80%), mean PIT {self.mean_pit():.2f} (ideal 0.50; "
               "below 0.5 means outcomes came in under the forecast)"]
        for metric in METRICS:
            for h in sorted({r.horizon for r in self.rows}):
                rows = self.select(metric, h)
                if rows:
                    out.append(f"  {metric:<16} +{h}y  n={len(rows):3d}  80% band {self.coverage(0.8, metric, h):4.0%}  "
                               f"50% band {self.coverage(0.5, metric, h):4.0%}  mean PIT {self.mean_pit(metric, h):.2f}  "
                               f"KS p {self.ks_pvalue(metric, h):.2f}  median width/p50 "
                               f"{np.median([(r.p90 - r.p10) / abs(r.p50) for r in rows]):.2f}")
        out += [f"  skipped {t}: {why}" for t, why in self.skipped.items()]
        return out


def _origin_filing_date(facts: Facts, end: date) -> date | None:
    days = [_day(r["filed"]) for tag in REVENUE for r in facts._rows(tag, "USD", "us-gaap")
            if r.get("form") in ANNUAL_FORMS and "start" in r and r["end"] == end.isoformat()]
    return min(days) if days else None


def _near(series: dict[date, object], target: date, days: int = 20):
    for end, fact in series.items():
        if abs((end - target).days) <= days:
            return fact
    return None


def _shift_years(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year + years)
    except ValueError:  # 29 February
        return end.replace(year=end.year + years, day=28)

def _forecast_draws(company, horizon: int, draws: int, seed: int, correlated: bool) -> dict[str, np.ndarray]:
    model, build = company_model(company, years=horizon)
    corr = default_dcf_correlation(model.input_names) if correlated else None
    x = sample_beliefs(model, {}, draws, seed, corr)
    out = {m: np.empty((draws, horizon)) for m in METRICS}
    for i, row in enumerate(x):
        v = dict(zip(model.input_names, row))
        revs = build.revenues(v)
        path = project(v, revs)
        r = np.asarray(revs[1:])
        nopat = r * v["ebit_margin"]
        nopat = nopat - np.maximum(nopat, 0.0) * v["tax_rate"]
        out["revenue"][i], out["ebit"][i] = r, path["ebit"]
        out["cash_flow_proxy"][i] = nopat + r * (v["da_pct"] - v["capex_pct"])
    return out


def _realized(series: dict[str, dict[date, object]], end: date) -> dict[str, float] | None:
    get = lambda name: _near(series[name], end)
    revenue, ebit = get("revenue"), get("ebit")
    if revenue is None or ebit is None:
        return None
    out = {"revenue": revenue.value / 1e6, "ebit": ebit.value / 1e6}
    tax, pretax, da, capex = get("tax"), get("pretax"), get("da"), get("capex")
    if all(f is not None for f in (tax, pretax, da, capex)) and pretax.value > 0 and 0 <= tax.value / pretax.value <= 0.5:
        e = ebit.value / 1e6
        out["cash_flow_proxy"] = e - max(e, 0.0) * tax.value / pretax.value + (da.value - capex.value) / 1e6
    return out


def backtest_company(raw: dict, ticker: str = "", horizons: Sequence[int] = (1, 2, 3), draws: int = 3000, seed: int = 0,
                     range_width: float = 1.0, correlated: bool = False, years_history: int = 8,
                     base_years: int = 3) -> BacktestResult:
    """Backtest one company's facts. Every past fiscal year-end with enough history and a realised outcome is an origin."""
    facts = Facts(raw)
    series = {"revenue": facts.annual(REVENUE), "ebit": facts.annual(EBIT), "tax": facts.annual(TAX),
              "pretax": facts.annual(PRETAX), "da": facts.annual(DA), "capex": facts.annual(CAPEX)}
    rows: list[BacktestRow] = []
    skipped: dict[str, str] = {}
    for origin in sorted(series["revenue"]):
        as_of = _origin_filing_date(facts, origin)
        known = [e for e in Facts(raw, as_of).annual(REVENUE) if e <= origin] if as_of else []
        if as_of is None or len(known) < MIN_REVENUE_OBSERVATIONS:
            continue
        realized = {h: _realized(series, _shift_years(origin, h)) for h in horizons}
        realized = {h: r for h, r in realized.items() if r is not None}
        if not realized:
            continue
        try:
            company = company_from_facts(raw, ticker=ticker, as_of=as_of, range_width=range_width,
                                         years_history=years_history, base_years=base_years)
        except (ValueError, KeyError) as error:
            skipped[f"{ticker} {origin}"] = str(error)
            continue
        if company.fiscal_year_end != origin:
            continue
        simulated = _forecast_draws(company, max(realized), draws, seed + origin.year, correlated)
        for h, values in realized.items():
            for metric, value in values.items():
                d = simulated[metric][:, h - 1]
                p10, p50, p90 = (float(q) for q in np.quantile(d, [0.1, 0.5, 0.9]))
                rows.append(BacktestRow(ticker, origin, as_of, h, metric, value, p10, p50, p90, pit(d, value)))
    return BacktestResult(tuple(rows), skipped)


@dataclass(frozen=True)
class WidthPoint:
    width: float
    coverage_80: float
    coverage_50: float
    median_relative_width: float


def calibrate_width(raw: dict, widths: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0), **kwargs) -> list[WidthPoint]:
    """Backtest at several belief widths (a multiplier on each driver's half-range) to see what width is calibrated."""
    points = []
    for w in widths:
        result = backtest_company(raw, range_width=w, **kwargs)
        rel = [(r.p90 - r.p10) / abs(r.p50) for r in result.rows if r.p50]
        points.append(WidthPoint(w, result.coverage(0.8), result.coverage(0.5), float(np.median(rel)) if rel else float("nan")))
    return points


def backtest_basket(tickers: Sequence[str], *, fetch: Fetch = http_get, cache_dir=None, **kwargs) -> BacktestResult:
    """Backtest several companies and pool the forecasts. A company the filings cannot support is reported, not hidden."""
    rows: list[BacktestRow] = []
    skipped: dict[str, str] = {}
    for ticker in tickers:
        try:
            cik, symbol, _ = resolve_company(ticker, fetch, cache_dir)
            raw = json.loads(cached_get(SEC_FACTS_URL.format(cik=cik), fetch, cache_dir))
            result = backtest_company(raw, symbol, **kwargs)
        except Exception as error:  # a network or filing-format failure for one name must not sink the basket
            skipped[ticker] = f"{type(error).__name__}: {error}"
            continue
        rows += result.rows
        skipped.update(result.skipped)
        if not result.rows:
            skipped[ticker] = "no forecast origin had enough history and a realised outcome"
    return BacktestResult(tuple(rows), skipped)


@dataclass(frozen=True)
class PriceRow:
    ticker: str
    origin: date
    as_of: date
    market_price: float
    p10: float
    p50: float
    p90: float
    pit: float  # where the market price sat in the DCF belief distribution
    forward_return: float | None  # price change over the following year (no dividends)


def price_check(raw: dict, prices: PriceHistory, ticker: str = "", draws: int = 3000, seed: int = 0,
                range_width: float = 1.0, years: int = 10, overrides=None) -> list[PriceRow]:
    """Where the market price stood in the as-of DCF belief distribution, and what the price did next.

    This is not a calibration of the DCF as a price forecaster (a DCF is a value, not a forecast); it shows whether
    the market sat inside the belief range and whether sitting low in it preceded higher returns. Share counts
    are restated for later splits so they match the split-adjusted price series.
    """
    facts = Facts(raw)
    rows = []
    for origin in sorted(facts.annual(REVENUE)):
        as_of = _origin_filing_date(facts, origin)
        if as_of is None or prices.price_on(as_of) is None:
            continue
        try:
            company = company_from_facts(raw, ticker=ticker, as_of=as_of, range_width=range_width, overrides=overrides)
        except (ValueError, KeyError):
            continue
        if company.fiscal_year_end != origin:
            continue
        company = replace(company, shares=company.shares * prices.split_factor_after(as_of))
        model, _ = company_model(company, years=years)
        x = sample_beliefs(model, {}, draws, seed + origin.year)
        d = np.array([model.function(dict(zip(model.input_names, row)))["implied_price"] for row in x])
        market = prices.price_on(as_of)
        later = prices.price_on(as_of + timedelta(days=365)) if as_of + timedelta(days=365) <= prices.last_date else None
        rows.append(PriceRow(ticker, origin, as_of, market, *(float(q) for q in np.quantile(d, [0.1, 0.5, 0.9])),
                             pit(d, market), later / market - 1 if later else None))
    return rows


def price_rank_correlation(rows: Sequence[PriceRow]) -> tuple[float, int]:
    """Spearman correlation between where the market sat in the belief distribution and the next year's return."""
    pairs = [(r.pit, r.forward_return) for r in rows if r.forward_return is not None]
    if len(pairs) < 5:
        return float("nan"), len(pairs)
    return float(spearmanr(*zip(*pairs))[0]), len(pairs)