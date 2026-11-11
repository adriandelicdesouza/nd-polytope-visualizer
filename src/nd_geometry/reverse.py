"""Reverse DCF: given today's price, which driver values does the market imply, and how likely are they?

Two views. `solve_driver` moves one driver until the model price equals the market price, all
else at the base case. `reverse_dcf` works on the belief-weighted draws: it reports how much
belief mass sits at or above the market price, and, among draws whose price is close to the
market price, the mean and spread of every driver (the joint region the market is paying for).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .density import DensityResult
from .model import Model


@dataclass(frozen=True)
class DriverSolve:
    driver: str
    base: float
    low: float
    high: float
    value: float | None  # the driver value that makes the model price equal the target
    status: str  # "solved", "unreachable: needs more than high", "unreachable: needs less than low"


@dataclass(frozen=True)
class ImpliedDriver:
    name: str
    prior_mean: float
    prior_sd: float
    implied_mean: float
    implied_sd: float
    implied_p10: float
    implied_p90: float

    @property
    def shift_in_sd(self) -> float:
        """How far the market-implied mean sits from the belief mean, in belief standard deviations."""
        return (self.implied_mean - self.prior_mean) / self.prior_sd if self.prior_sd > 0 else 0.0


@dataclass(frozen=True)
class ReverseDCF:
    market_price: float
    base_price: float
    feasible_interval: tuple[float, float]
    attainable: bool
    belief_percentile: float  # share of belief mass with model price below the market price
    prob_at_or_above: float  # P(model price >= market price)
    prob_within: float  # P(|model price / market price - 1| <= tolerance)
    tolerance: float
    effective_draws: float
    bandwidth_used: float  # kernel width (share of the market price) after any widening
    implied: dict[str, ImpliedDriver]
    solves: dict[str, DriverSolve]
    implied_point_price: float  # model price at the implied means (the joint region is not a single point)

    def lines(self) -> list[str]:
        lo, hi = self.feasible_interval
        out = [f"Market price ${self.market_price:,.2f}; base-case model price ${self.base_price:,.2f}; "
               f"feasible interval ${lo:,.2f} to ${hi:,.2f}" + ("" if self.attainable else " (market price is OUTSIDE it)"),
               f"Belief mass at or above the market price: {self.prob_at_or_above:.1%} "
               f"(market sits at the {self.belief_percentile:.0%} percentile); "
               f"within +/-{self.tolerance:.0%} of it: {self.prob_within:.1%}; "
               f"{self.effective_draws:,.0f} effective draws at a {self.bandwidth_used:.0%} kernel"]
        out.append("Market-implied drivers (belief mean -> implied mean, p10 to p90, shift in belief sd):")
        for d in sorted(self.implied.values(), key=lambda d: -abs(d.shift_in_sd)):
            out.append(f"  {d.name:<20} {d.prior_mean:9.4f} -> {d.implied_mean:9.4f}  "
                       f"[{d.implied_p10:8.4f}, {d.implied_p90:8.4f}]  {d.shift_in_sd:+.2f} sd")
        out.append("One driver at a time (others at base):")
        for s in self.solves.values():
            text = f"needs {s.value:.4f}" if s.value is not None else s.status
            out.append(f"  {s.driver:<20} base {s.base:9.4f}  {text}  (range {s.low:.4f} to {s.high:.4f})")
        return out


def solve_driver(model: Model, driver: str, target: float, output: str = "implied_price",
                 at: dict[str, float] | None = None) -> DriverSolve:
    """The value of one driver, within its range, at which the model output equals `target`."""
    item = next(i for i in model.inputs if i.name == driver)
    point = dict(at or model.base_case())

    def gap(x: float) -> float:
        return model.evaluate({**point, driver: x})[output] - target

    lo, hi = gap(item.low), gap(item.high)
    if lo == 0:
        return DriverSolve(driver, item.base, item.low, item.high, item.low, "solved")
    if lo * hi > 0:
        # same sign at both ends: the target lies outside what this driver alone can reach
        increasing = hi > lo  # the model price rises with the driver
        target_above = lo < 0  # the model price is below the target at both ends
        needs_more = increasing == target_above
        return DriverSolve(driver, item.base, item.low, item.high, None,
                           "unreachable: needs more than high" if needs_more else "unreachable: needs less than low")
    return DriverSolve(driver, item.base, item.low, item.high, float(brentq(gap, item.low, item.high)), "solved")


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, x))


def reverse_dcf(result: DensityResult, market_price: float, output: str = "implied_price", tolerance: float = 0.05,
                bandwidth: float = 0.03, solve_drivers: list[str] | None = None,
                min_effective: float = 200.0) -> ReverseDCF:
    """Reverse DCF on belief draws. `bandwidth` is the width (as a share of the market price) of the kernel that
    selects draws near it, widened by half again until at least `min_effective` draws count (market prices far in
    the tail of the beliefs otherwise leave too few); `tolerance` is the half-width of the reported probability band."""
    model = result.model
    price = result.column(output)
    rel = price / market_price - 1.0
    used = bandwidth
    while True:
        weights = np.exp(-0.5 * (rel / used) ** 2)
        sq = float(np.sum(weights**2))
        ess = float(weights.sum() ** 2 / sq) if sq > 0 else 0.0
        if ess >= min_effective or used >= 0.5:
            break
        used *= 1.5
    implied: dict[str, ImpliedDriver] = {}
    for j, item in enumerate(model.inputs):
        column = result.inputs[:, j]
        prior_mean, prior_sd = float(column.mean()), float(column.std())
        if ess >= 5:
            mean = float(np.average(column, weights=weights))
            sd = float(np.sqrt(np.average((column - mean) ** 2, weights=weights)))
            p10, p90 = _weighted_quantile(column, weights, 0.1), _weighted_quantile(column, weights, 0.9)
        else:
            mean = sd = p10 = p90 = float("nan")
        implied[item.name] = ImpliedDriver(item.name, prior_mean, prior_sd, mean, sd, p10, p90)
    base_inputs = model.base_case()
    solves = {name: solve_driver(model, name, market_price, output) for name in (solve_drivers or model.input_names)}
    low, high = result.support(output)
    point = {n: d.implied_mean for n, d in implied.items()}
    point_price = model.evaluate(point)[output] if ess >= 5 else float("nan")
    return ReverseDCF(
        market_price=market_price, base_price=float(model.evaluate(base_inputs)[output]),
        feasible_interval=(low, high), attainable=low <= market_price <= high,
        belief_percentile=float(np.mean(price < market_price)), prob_at_or_above=float(np.mean(price >= market_price)),
        prob_within=float(np.mean(np.abs(rel) <= tolerance)), tolerance=tolerance, effective_draws=ess, bandwidth_used=used,
        implied=implied, solves=solves, implied_point_price=float(point_price))