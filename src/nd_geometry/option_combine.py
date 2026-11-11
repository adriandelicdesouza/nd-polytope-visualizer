"""Option-implied price density, combined with the DCF belief density.

Two densities over the same stock:
  - the option market's risk-neutral density of the price at expiry, recovered from a single-expiry chain by
    Breeden-Litzenberger (e^{rT} d2C/dK2) on a smooth implied-vol fit across strikes;
  - the DCF belief density of today's fair value per share, from the belief draws.

To put both on one scale, the option density is brought back to today by the risk-neutral forward factor
e^{(r-q)T}: x_today = S_T * e^{-(r-q)T}. That mapping is an assumption (it treats the option market's price at
expiry as a spot-equivalent value today); the report says so.

The combined density is the normalised product p_dcf(x) * p_opt(x): mass is high only where both agree. A
mixture (0.5 p_dcf + 0.5 p_opt) is reported alongside, since a product becomes empty when the two do not overlap.
"""

from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass


import numpy as np
from scipy.stats import gaussian_kde, norm


CALL, PUT = "call", "put"


def _bsm_call(spot, strike, t, rate, vol, q):
    sq = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * vol * vol) * t) / sq
    d2 = d1 - sq
    return spot * math.exp(-q * t) * norm.cdf(d1) - strike * math.exp(-rate * t) * norm.cdf(d2)


def implied_vol_call(price: float, spot: float, strike: float, t: float, rate: float, q: float = 0.0):
    """Black-Scholes implied vol of a call by bisection; None if the price is outside the no-arbitrage bounds."""
    lower = max(spot * math.exp(-q * t) - strike * math.exp(-rate * t), 0.0)
    if not (lower + 1e-10 < price < spot * math.exp(-q * t)):
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _bsm_call(spot, strike, t, rate, mid, q) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def otm_call_quotes(quotes, spot: float, rate: float, t: float, q: float = 0.0):
    """Out-of-the-money quotes as call prices: calls at or above spot, puts below spot converted by put-call parity.

    quotes: iterable of (strike, option_type, mid_price), option_type 'call' or 'put'.
    Returns strikes and call prices, sorted by strike, with quotes that fail basic checks dropped.
    """
    rows = {}
    for strike, kind, mid in quotes:
        if mid is None or not np.isfinite(mid) or mid <= 0 or strike <= 0:
            continue
        if kind == CALL and strike >= spot:
            call = float(mid)
        elif kind == PUT and strike < spot:
            call = float(mid) + spot * math.exp(-q * t) - strike * math.exp(-rate * t)
        else:
            continue
        rows[float(strike)] = call
    strikes = np.array(sorted(rows))
    return strikes, np.array([rows[k] for k in strikes])


@dataclass(frozen=True)
class OptionDensity:
    grid: np.ndarray  # price at expiry
    density: np.ndarray  # integrates to 1 over grid
    forward: float  # spot * e^{(r-q)T}
    rate: float
    q: float
    t: float
    fit_rmse_vol: float  # RMS error of the smooth smile against the quoted implied vols (in vol points)
    points: int  # strikes used

    def mean(self) -> float:
        return float(np.trapezoid(self.grid * self.density, self.grid))

    def cdf(self) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(0.5 * (self.density[1:] + self.density[:-1]) * np.diff(self.grid))])
        return c / c[-1]

    def quantiles(self, probs=(0.05, 0.5, 0.95)) -> dict[float, float]:
        c = self.cdf()
        return {p: float(np.interp(p, c, self.grid)) for p in probs}

    def prob_above(self, level: float) -> float:
        c = self.cdf()
        return float(1 - np.interp(level, self.grid, c))


def option_density(strikes, call_prices, spot: float, rate: float, t: float, q: float = 0.0,
                   degree: int = 2, grid_points: int = 2000) -> OptionDensity:
    """Risk-neutral density of the price at expiry from call prices across strikes."""
    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)
    if t <= 0:
        raise ValueError("expiry must be in the future")
    forward = spot * math.exp((rate - q) * t)
    ivs = []
    keep = []
    for k, c in zip(strikes, call_prices):
        iv = implied_vol_call(c, spot, k, t, rate, q)
        if iv is not None and 0.05 <= iv <= 2.5:
            ivs.append(iv)
            keep.append(k)
    if len(keep) < degree + 3:
        raise ValueError(f"need at least {degree + 3} usable strikes for a smile fit, got {len(keep)}")
    keep = np.array(keep)
    ivs = np.array(ivs)
    x = np.log(keep / forward)
    coef = np.polyfit(x, ivs, degree)
    resid = np.polyval(coef, x) - ivs
    # one robust pass: drop quotes that sit more than 3 robust sd off the first fit, then refit
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    good = np.abs(resid) <= 3 * scale
    if good.sum() >= degree + 3 and not good.all():
        keep, ivs, x = keep[good], ivs[good], x[good]
        coef = np.polyfit(x, ivs, degree)
        resid = np.polyval(coef, x) - ivs
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    lo, hi = keep.min() * 0.5, keep.max() * 1.5
    lo = max(lo, 1e-3 * spot)
    grid = np.linspace(lo, hi, grid_points)
    fitted = np.clip(np.polyval(coef, np.log(grid / forward)), 0.01, 3.0)
    prices = np.array([_bsm_call(spot, k, t, rate, v, q) for k, v in zip(grid, fitted)])
    second = np.gradient(np.gradient(prices, grid), grid)
    density = np.clip(math.exp(rate * t) * second, 0.0, None)
    total = np.trapezoid(density, grid)
    if total <= 0:
        raise ValueError("the fitted smile gives no positive density")
    return OptionDensity(grid, density / total, forward, rate, q, t, rmse, len(keep))


@dataclass(frozen=True)
class Combined:
    grid: np.ndarray  # fair value per share, today's scale
    dcf: np.ndarray
    option: np.ndarray
    product: np.ndarray | None  # normalised p_dcf * p_opt; None if the two densities do not overlap
    mixture: np.ndarray
    overlap: float  # mass of the unnormalised product: near 0 means the DCF and the market disagree
    market_price: float

    def summary(self, density: np.ndarray | None) -> dict[str, float]:
        if density is None:
            return {}
        c = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(self.grid))])
        c = c / c[-1]
        return {"p5": float(np.interp(0.05, c, self.grid)), "p50": float(np.interp(0.5, c, self.grid)),
                "p95": float(np.interp(0.95, c, self.grid)),
                "p_above_market": float(1 - np.interp(self.market_price, self.grid, c))}


def combine(dcf_draws, opt: OptionDensity, market_price: float, points: int = 1500) -> Combined:
    draws = np.asarray(dcf_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    factor = math.exp((opt.rate - opt.q) * opt.t)  # S_T -> today
    lo = min(np.percentile(draws, 0.1), opt.grid[0] / factor)
    hi = max(np.percentile(draws, 99.9), opt.grid[-1] / factor)
    x = np.linspace(max(lo, 1e-6), hi, points)
    dcf = gaussian_kde(draws)(x)
    option = np.interp(x * factor, opt.grid, opt.density, left=0.0, right=0.0) * factor
    dcf = dcf / np.trapezoid(dcf, x)
    option = option / np.trapezoid(option, x)
    raw = dcf * option
    overlap = float(np.trapezoid(raw, x))
    product = raw / overlap if overlap > 1e-12 else None
    mixture = 0.5 * dcf + 0.5 * option
    return Combined(x, dcf, option, product, mixture, overlap, float(market_price))


def chain_from_snapshot(snapshot, spot: float | None = None, max_spread: float = 1.0,
                        band: tuple[float, float] = (0.6, 1.6)):
    """Turn an options_platform Snapshot's rows into (strike, type, mid) quotes, keeping only usable ones.

    A quote is kept when both bid and ask are positive, the spread is at most `max_spread` of the mid, and the
    strike lies within `band` times spot. Stale zero-bid rows and far-away strikes otherwise swamp the smile.
    """
    spot = snapshot.spot if spot is None else spot
    quotes = []
    for r in snapshot.rows:
        if r.bid is None or r.ask is None or r.bid <= 0 or r.ask <= 0:
            continue
        mid = (r.bid + r.ask) / 2
        if (r.ask - r.bid) / mid > max_spread:
            continue
        if not (band[0] * spot <= float(r.strike) <= band[1] * spot):
            continue
        kind = CALL if str(r.option_type).lower().endswith("call") else PUT
        quotes.append((float(r.strike), kind, mid))
    return quotes


def fetch_chain(ticker: str, expiry: date | None, options_src: str):
    """Use the options project's market-data adapter (Yahoo via yfinance, disk cached)."""
    import sys
    if options_src not in sys.path:
        sys.path.insert(0, options_src)
    from options_platform.data.market_data import YFinanceAdapter

    return YFinanceAdapter().fetch(ticker, expiry)

def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from .company import analyse_company
    from .revenue_build import SegmentBuild

    p = argparse.ArgumentParser(description="DCF belief density combined with the option-implied price density.")
    p.add_argument("ticker")
    p.add_argument("--expiry", help="YYYY-MM-DD; default: the nearest listed expiry (may be very short)")
    p.add_argument("--options-src", default=str(Path.home() / "options-pricing-model" / "src"),
                   help="src folder of the options project")
    p.add_argument("--facts-file", help="XBRL company facts JSON (default: download from SEC)")
    p.add_argument("--build", help="segment revenue build JSON")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--wacc-build", action="store_true")
    p.add_argument("--per-year", action="store_true")
    p.add_argument("--range-width", type=float, default=1.0)
    p.add_argument("--draws", type=int, default=5000)
    p.add_argument("--market-price", type=float, help="default: the spot from the option snapshot")
    p.add_argument("--rate", type=float, help="risk-free rate as a decimal; default: from the snapshot")
    p.add_argument("--dividend", type=float, default=0.0, help="continuous dividend yield, decimal")
    p.add_argument("--json", help="also write the combined densities to this JSON file")
    args = p.parse_args(argv)

    expiry = date.fromisoformat(args.expiry) if args.expiry else None
    snap = fetch_chain(args.ticker, expiry, args.options_src)
    if snap.expiry is None or snap.spot is None:
        raise SystemExit("no option expiry or spot in the snapshot")
    t = (snap.expiry - date.today()).days / 365.0
    rate = args.rate if args.rate is not None else snap.risk_free_rate
    if rate is None:
        raise SystemExit("no risk-free rate in the snapshot; pass --rate")
    spot = snap.spot
    market = args.market_price if args.market_price is not None else spot

    raw = json.loads(Path(args.facts_file).read_text()) if args.facts_file else None
    analysis = analyse_company(
        args.ticker, build=SegmentBuild.from_json(args.build) if args.build else None, years=args.years,
        wacc_build=args.wacc_build, per_year=args.per_year, range_width=args.range_width, draws=args.draws,
        backtest=False, market_price=market, raw=raw)

    quotes = chain_from_snapshot(snap, spot)
    strikes, calls = otm_call_quotes(quotes, spot, rate, t, args.dividend)
    opt = option_density(strikes, calls, spot, rate, t, args.dividend)
    dcf = analysis.result.column("implied_price")
    out = combine(dcf, opt, market)

    print(f"{args.ticker}: option expiry {snap.expiry} ({t * 365:.0f} days), spot ${spot:,.2f}, "
          f"rate {rate:.2%}, market used for comparison ${market:,.2f}")
    print(f"option smile: {opt.points} OTM strikes, fit error {opt.fit_rmse_vol * 100:.2f} vol points")
    print(f"option-implied price at expiry, mean ${opt.mean():,.2f}; forward ${opt.forward:,.2f}")
    q_opt = opt.quantiles()
    print("option-implied price at expiry p5/p50/p95: " + "  ".join(f"${v:,.2f}" for v in q_opt.values()))
    print(f"option density mapped to today with the forward factor e^(-(r-q)T) = "
          f"{math.exp(-(rate - args.dividend) * t):.4f}")
    print("")
    for label, dens in (("DCF beliefs alone", out.dcf), ("option market alone", out.option),
                        ("combined (product)", out.product), ("mixture (50/50)", out.mixture)):
        if dens is None:
            print(f"{label:22s} no overlap: the DCF beliefs and the option market price do not meet")
            continue
        s = out.summary(dens)
        print(f"{label:22s} p5 ${s['p5']:,.2f}  p50 ${s['p50']:,.2f}  p95 ${s['p95']:,.2f}  "
              f"P(above ${market:,.2f}) {s['p_above_market']:.1%}")
    print(f"overlap of DCF and option densities before normalising: {out.overlap:.3g} "
          "(near zero means the DCF and the market disagree)")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "ticker": args.ticker, "expiry": str(snap.expiry), "t_years": t, "rate": rate, "spot": spot,
            "market_price": market, "grid": out.grid.tolist(), "dcf": out.dcf.tolist(),
            "option": out.option.tolist(), "product": None if out.product is None else out.product.tolist(),
            "mixture": out.mixture.tolist(), "overlap": out.overlap}))
        print(f"densities written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())