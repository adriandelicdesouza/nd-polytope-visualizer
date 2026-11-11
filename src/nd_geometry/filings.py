"""Company inputs from SEC filings (XBRL company facts).

`load_company` turns a ticker into the base figures a DCF needs (revenue, share count,
net debt) and a low / base / high range for every driver, taken from the company's own
reported history. Every number carries a source label ("filings", "default", "override").
`as_of` drops facts filed after that date, which the backtest needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .model import Input

SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = os.environ.get("ND_SEC_USER_AGENT", "")  # the SEC requires a contact, e.g. "Your Name you@example.com"
ANNUAL_FORMS = {"10-K", "10-K/A", "10-K405", "10-KT", "20-F", "40-F"}

REVENUE = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueServicesNet")
EBIT = ("OperatingIncomeLoss",)
PRETAX = ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments")
TAX = ("IncomeTaxExpenseBenefit",)
CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
DA = ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
      "DepreciationAmortizationAndAccretionNet", "DepreciationAmortizationAndOther")
CFO = ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
DILUTED_SHARES = ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic")
CASH = ("CashAndCashEquivalentsAtCarryingValue",)
SHORT_INVESTMENTS = ("ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent")
DEBT_TOTAL = ("LongTermDebt",)
DEBT_PARTS = ("LongTermDebtNoncurrent", "LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent")

RANGE_YEARS = 5  # ranges span the latest five fiscal years: older growth and margins are poor bounds for the future
NWC_BASE = 0.10  # working-capital drag already in the model; the cash-conversion gap is measured net of it

Fetch = Callable[[str], bytes]


def http_get(url: str, timeout: float = 60.0, user_agent: str | None = None) -> bytes:
    agent = user_agent or os.environ.get("ND_SEC_USER_AGENT", USER_AGENT)
    if ".sec.gov" in url or "//sec.gov" in url:
        if "@" not in agent:
            raise RuntimeError("The SEC needs a contact in the User-Agent. Set ND_SEC_USER_AGENT to 'Your Name you@example.com'.")
    request = urllib.request.Request(url, headers={"User-Agent": agent or "Mozilla/5.0", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def cached_get(url: str, fetch: Fetch = http_get, cache_dir: str | Path | None = None, max_age_hours: float = 24.0) -> bytes:
    """Fetch a URL, keeping the body in `cache_dir` for `max_age_hours` when a directory is given."""
    if cache_dir is None:
        return fetch(url)
    path = Path(cache_dir) / (hashlib.sha1(url.encode()).hexdigest() + ".bin")
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return path.read_bytes()
    data = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def resolve_company(identifier: str | int, fetch: Fetch = http_get, cache_dir=None) -> tuple[int, str, str]:
    """Return (cik, ticker, name) for a ticker symbol or a numeric CIK."""
    text = str(identifier).strip()
    if text.isdigit():
        return int(text), text, ""
    table = json.loads(cached_get(SEC_TICKERS_URL, fetch, cache_dir, 24 * 7))
    wanted = text.upper().replace(".", "-")
    for row in table.values():
        if row["ticker"].upper() == wanted:
            return int(row["cik_str"]), row["ticker"], row.get("title", "")
    raise KeyError(f"ticker {identifier!r} not found in the SEC ticker table")


def _day(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


@dataclass(frozen=True)
class Fact:
    end: date
    value: float
    filed: date
    form: str
    start: date | None = None


class Facts:
    """Read access to one company's XBRL facts, optionally frozen at an `as_of` filing date."""

    def __init__(self, raw: Mapping, as_of: date | None = None):
        self.raw = raw
        self.as_of = as_of
        self.name = raw.get("entityName", "")

    def _rows(self, tag: str, unit: str, taxonomy: str) -> list[dict]:
        return self.raw.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, [])

    def _visible(self, row: dict) -> bool:
        return self.as_of is None or _day(row["filed"]) <= self.as_of

    def annual(self, tags, unit: str = "USD", taxonomy: str = "us-gaap") -> dict[date, Fact]:
        """Full-year flow values by period end. The latest filing wins per period; earlier tags win per period."""
        merged: dict[date, Fact] = {}
        for tag in tags:
            best: dict[date, Fact] = {}
            for row in self._rows(tag, unit, taxonomy):
                if row.get("form") not in ANNUAL_FORMS or "start" not in row or not self._visible(row):
                    continue
                start, end = _day(row["start"]), _day(row["end"])
                if not 350 <= (end - start).days <= 380:
                    continue
                fact = Fact(end, float(row["val"]), _day(row["filed"]), row["form"], start)
                if end not in best or fact.filed > best[end].filed:
                    best[end] = fact
            for end, fact in best.items():
                merged.setdefault(end, fact)
        return dict(sorted(merged.items()))

    def instants(self, tags, unit: str = "USD", taxonomy: str = "us-gaap") -> dict[date, Fact]:
        """Balance-sheet values by date, from any form; the latest filing wins per date."""
        merged: dict[date, Fact] = {}
        for tag in tags:
            for row in self._rows(tag, unit, taxonomy):
                if "start" in row or not self._visible(row):
                    continue
                fact = Fact(_day(row["end"]), float(row["val"]), _day(row["filed"]), row["form"])
                if fact.end not in merged or fact.filed > merged[fact.end].filed:
                    merged[fact.end] = fact
        return dict(sorted(merged.items()))

    def cover_shares(self) -> Fact | None:
        rows = [r for r in self._rows("EntityCommonStockSharesOutstanding", "shares", "dei") if self._visible(r)]
        if not rows:
            return None
        row = max(rows, key=lambda r: (r["end"], r["filed"]))
        return Fact(_day(row["end"]), float(row["val"]), _day(row["filed"]), row.get("form", ""))

@dataclass(frozen=True)
class DriverSpec:
    """One driver's range, with where it came from ("filings", "default" or "override")."""

    name: str
    low: float
    base: float
    high: float
    source: str
    note: str = ""


@dataclass(frozen=True)
class CompanyInputs:
    ticker: str
    name: str
    cik: int
    as_of: date | None
    fiscal_year_end: date
    base_revenue: float  # $m, latest full fiscal year
    shares: float  # millions
    net_debt: float  # $m
    drivers: dict[str, DriverSpec]
    history: dict[str, tuple[tuple[date, float], ...]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def inputs(self, names=None) -> tuple[Input, ...]:
        chosen = names if names is not None else list(self.drivers)
        return tuple(Input(n, "%", self.drivers[n].low, self.drivers[n].high, self.drivers[n].base) for n in chosen)

    def table(self) -> str:
        lines = [f"{self.ticker} {self.name}  fiscal year end {self.fiscal_year_end}"
                 + (f"  as of {self.as_of}" if self.as_of else ""),
                 f"base revenue ${self.base_revenue:,.0f}m  shares {self.shares:,.1f}m  net debt ${self.net_debt:,.0f}m"]
        for s in self.drivers.values():
            lines.append(f"  {s.name:<20} {s.low:8.4f} {s.base:8.4f} {s.high:8.4f}  [{s.source}] {s.note}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def make_range(values, *, widen: float = 1.0, min_width: float = 0.02, clip=(None, None), recent: int = 3):
    """Low / base / high from a history: base is the recent mean, the range spans the history.

    `widen` scales both half-widths about the base; `min_width` is the narrowest full width allowed.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    base = float(np.mean(v[-recent:]))
    lo_c, hi_c = clip
    if lo_c is not None:
        base = max(base, lo_c)
    if hi_c is not None:
        base = min(base, hi_c)
    below = max(base - float(v.min()), min_width / 2) * widen
    above = max(float(v.max()) - base, min_width / 2) * widen
    low, high = base - below, base + above
    if lo_c is not None:
        low = max(low, lo_c)
    if hi_c is not None:
        high = min(high, hi_c)
    if not low < high:
        high = low + 1e-6
    return low, min(max(base, low), high), high


def _per_year(numer: dict[date, Fact], denom: dict[date, Fact]):
    return [(e, numer[e].value / denom[e].value) for e in denom if e in numer and denom[e].value]


def _net_debt(facts: Facts) -> tuple[float, str]:
    cash = facts.instants(CASH)
    if not cash:
        return 0.0, "no cash balance in filings; net debt set to zero"
    when = max(cash)
    short = facts.instants(SHORT_INVESTMENTS)
    cash_total = cash[when].value + (short[when].value if when in short else 0.0)
    total = facts.instants(DEBT_TOTAL)
    if when in total:
        debt, label = total[when].value, "LongTermDebt"
    else:
        found = {}
        for tag in DEBT_PARTS:
            series = facts.instants([tag])
            if when in series:
                found[tag] = series[when].value
        if "DebtCurrent" in found:
            found.pop("ShortTermBorrowings", None)  # DebtCurrent already includes it
        debt, label = sum(found.values()), (" + ".join(found) or "none reported at that date")
    return (debt - cash_total) / 1e6, f"net debt at {when} (debt: {label}; operating leases excluded)"


def _shares(facts: Facts, diluted: dict[date, Fact]) -> tuple[float, str]:
    cover = facts.cover_shares()
    latest = diluted[max(diluted)].value if diluted else None
    if cover and (latest is None or cover.value >= 0.6 * latest):
        return cover.value / 1e6, f"shares from the cover page ({cover.end})"
    if latest:
        return latest / 1e6, "shares from diluted weighted average (cover page missing or partial)"
    return float("nan"), "NO SHARE COUNT in filings: per-share values are not available (enough for revenue backtests)"

def company_from_facts(raw: Mapping, *, ticker: str = "", cik: int = 0, as_of: date | None = None,
                       overrides: Mapping[str, tuple[float, float, float]] | None = None,
                       range_width: float = 1.0, years_history: int = 8, base_years: int = 3) -> CompanyInputs:
    facts = Facts(raw, as_of)
    revenue = facts.annual(REVENUE)
    if len(revenue) < 3:
        raise ValueError("fewer than three annual revenue observations in the filings")
    ends = sorted(revenue)[-(years_history + 1):]
    revenue = {e: revenue[e] for e in ends}
    fy_end = ends[-1]
    ebit, tax, pretax = facts.annual(EBIT), facts.annual(TAX), facts.annual(PRETAX)
    capex, da, cfo = facts.annual(CAPEX), facts.annual(DA), facts.annual(CFO)
    diluted = facts.annual(DILUTED_SHARES, unit="shares")
    notes: list[str] = []

    growth = [(b, revenue[b].value / revenue[a].value - 1) for a, b in zip(ends[:-1], ends[1:])
              if 350 <= (b - a).days <= 380]
    margin = _per_year(ebit, revenue)
    tax_rate = [(e, v) for e, v in _per_year(tax, {e: f for e, f in pretax.items() if e in revenue})
                if 0 <= v <= 0.5 and pretax[e].value > 0]
    capex_pct, da_pct = _per_year(capex, revenue), _per_year(da, revenue)
    g_by_end, t_by_end = dict(growth), dict(tax_rate)
    gap = []
    for e in ends:
        if e in ebit and e in da and e in cfo and e in t_by_end and e in g_by_end:
            nopat = ebit[e].value * (1 - t_by_end[e])
            gap.append((e, (nopat + da[e].value - cfo[e].value) / revenue[e].value - NWC_BASE * g_by_end[e]))

    drivers: dict[str, DriverSpec] = {}

    def add(name, series, default, **range_args):
        recent = series[-RANGE_YEARS:]
        if len(recent) >= 2:
            low, base, high = make_range([v for _, v in recent], widen=range_width, recent=base_years, **range_args)
            drivers[name] = DriverSpec(name, low, base, high, "filings", f"{len(recent)} fiscal years to {recent[-1][0].year}")
        else:
            drivers[name] = DriverSpec(name, *default, "default", "not enough filed history")
            notes.append(f"{name}: default range, filings gave {len(series)} usable year(s)")

    add("revenue_growth", growth, (0.02, 0.06, 0.10), min_width=0.04, clip=(-0.05, 0.5))
    g1 = drivers["revenue_growth"]
    drivers["growth_fade_to"] = DriverSpec("growth_fade_to", min(0.0, g1.base), min(0.03, g1.base), max(0.06, g1.base), "default",
                                           "assumption: growth fades linearly to this by the last forecast year")
    add("ebit_margin", margin, (0.10, 0.20, 0.30), min_width=0.06, clip=(-0.2, 0.6))
    add("tax_rate", tax_rate, (0.20, 0.25, 0.30), min_width=0.04, clip=(0.0, 0.40))
    add("capex_pct", capex_pct, (0.03, 0.05, 0.07), min_width=0.02, clip=(0.0, 0.30))
    add("da_pct", da_pct, (0.03, 0.045, 0.06), min_width=0.02, clip=(0.0, 0.30))
    drivers["nwc_pct"] = DriverSpec("nwc_pct", 0.05, NWC_BASE, 0.15, "default", "assumption: drag on the revenue increment")
    add("other_reinvest_pct", gap, (0.0, 0.0, 0.03), min_width=0.02, clip=(0.0, 0.15))
    drivers["wacc"] = DriverSpec("wacc", 0.08, 0.10, 0.12, "default", "assumption: replace with a CAPM estimate via overrides")
    drivers["terminal_growth"] = DriverSpec("terminal_growth", 0.01, 0.02, 0.03, "default", "assumption")
    for name, (low, base, high) in (overrides or {}).items():
        if name not in drivers:
            raise KeyError(f"unknown driver {name!r}; drivers are {list(drivers)}")
        drivers[name] = DriverSpec(name, low, base, high, "override", "user supplied")
    for s in drivers.values():
        if not s.low <= s.base <= s.high:
            raise ValueError(f"driver {s.name}: need low <= base <= high, got {s.low}, {s.base}, {s.high}")
    if drivers["wacc"].low <= drivers["terminal_growth"].high:
        raise ValueError("the WACC range must sit above the terminal growth range")

    net_debt, debt_note = _net_debt(facts)
    shares, share_note = _shares(facts, diluted)
    history = {"revenue_m": tuple((e, f.value / 1e6) for e, f in revenue.items()), "revenue_growth": tuple(growth),
               "ebit_margin": tuple(margin), "tax_rate": tuple(tax_rate), "capex_pct": tuple(capex_pct),
               "da_pct": tuple(da_pct), "other_reinvest_pct": tuple(gap)}
    return CompanyInputs(ticker=ticker, name=facts.name, cik=cik, as_of=as_of, fiscal_year_end=fy_end,
                         base_revenue=revenue[fy_end].value / 1e6, shares=shares, net_debt=net_debt,
                         drivers=drivers, history=history, notes=tuple(notes + [debt_note, share_note]))


def load_company(identifier, *, as_of=None, overrides=None, range_width: float = 1.0, years_history: int = 8,
                 fetch: Fetch = http_get, cache_dir=None) -> CompanyInputs:
    """Download a company's XBRL facts from the SEC and build its DCF inputs."""
    cik, ticker, _ = resolve_company(identifier, fetch, cache_dir)
    raw = json.loads(cached_get(SEC_FACTS_URL.format(cik=cik), fetch, cache_dir))
    return company_from_facts(raw, ticker=ticker, cik=cik, as_of=as_of, overrides=overrides,
                              range_width=range_width, years_history=years_history)