"""Belief-weighted output polytope.

The output polytope is the feasible region: every outcome the model can produce
inside the input box. Beliefs say how likely each part of that region is. Draw the
inputs from belief distributions, push them through the model, and the draws give a
probability density over the polytope. Projecting onto one output (share price)
gives its feasible interval and its density within that interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde, triang, truncnorm

from .analysis import evaluate_rows
from .correlation import Correlation, copula_uniforms
from .model import Model
from .output_polytope import OutputPolytope, output_polytope


@dataclass(frozen=True)
class Belief:
    """How likely values of one input are within its range.

    kind: "uniform", "triangular" (mode = base case) or "normal" (mean = base case,
    truncated to the range). `spread` is the normal's sd as a share of the range.
    """

    kind: str = "triangular"
    spread: float = 0.25

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "triangular", "normal"):
            raise ValueError("kind must be uniform, triangular or normal")
        if self.spread <= 0:
            raise ValueError("spread must be positive")


def _inverse_cdf(item, belief: Belief, u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-12, 1 - 1e-12)
    width = item.high - item.low
    if belief.kind == "uniform":
        return item.low + u * width
    if belief.kind == "triangular":
        return triang.ppf(u, (item.base - item.low) / width, loc=item.low, scale=width)
    sd = belief.spread * width
    return truncnorm.ppf(u, (item.low - item.base) / sd, (item.high - item.base) / sd, loc=item.base, scale=sd)


def sample_beliefs(model: Model, beliefs: dict[str, Belief], draws: int, seed: int = 0,
                   correlation: Correlation | None = None) -> np.ndarray:
    """Draw inputs from each input's belief (default triangular on its base case).

    With `correlation`, the drivers move together through a Gaussian copula and every marginal belief is kept.
    """
    unknown = set(beliefs) - set(model.input_names)
    if unknown:
        raise ValueError(f"beliefs for unknown inputs: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    if correlation is not None:
        u = copula_uniforms(correlation.aligned(model.input_names), draws, rng)
        return np.column_stack([_inverse_cdf(item, beliefs.get(item.name, Belief()), u[:, j])
                                for j, item in enumerate(model.inputs)])
    columns = []
    for item in model.inputs:
        belief = beliefs.get(item.name, Belief())
        if belief.kind == "uniform":
            column = rng.uniform(item.low, item.high, draws)
        elif belief.kind == "triangular":
            column = rng.triangular(item.low, item.base, item.high, draws)
        else:
            sd = belief.spread * (item.high - item.low)
            a, b = (item.low - item.base) / sd, (item.high - item.base) / sd
            column = truncnorm.rvs(a, b, loc=item.base, scale=sd, size=draws, random_state=rng)
        columns.append(column)
    return np.column_stack(columns)


@dataclass(frozen=True)
class DensityResult:
    """Draws from the belief distribution, with the feasible polytope they live in."""

    model: Model
    beliefs: dict[str, Belief]
    inputs: np.ndarray
    outputs: np.ndarray
    feasible: OutputPolytope
    correlation: Correlation | None = None

    def column(self, output: str) -> np.ndarray:
        return self.outputs[:, self.model.output_names.index(output)]

    def support(self, output: str) -> tuple[float, float]:
        """Feasible interval of one output: the polytope projected onto that axis."""
        index = self.model.output_names.index(output)
        low, high = self.feasible.bounds()[index]
        return float(low), float(high)

    def quantiles(self, output: str, levels=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[float, float]:
        return {q: float(np.quantile(self.column(output), q)) for q in levels}

    def density(self, output: str, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Kernel density of one output on a grid spanning its feasible interval."""
        low, high = self.support(output)
        grid = np.linspace(low, high, points)
        return grid, gaussian_kde(self.column(output))(grid)

    def probability_between(self, output: str, low: float, high: float) -> float:
        column = self.column(output)
        return float(np.mean((column >= low) & (column <= high)))

    def joint_density(self, x: str, y: str, bins: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Probability mass per cell over two outputs: the density on a 2-D polytope projection."""
        h, xe, ye = np.histogram2d(self.column(x), self.column(y), bins=bins,
                                   range=[self.support(x), self.support(y)])
        return (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, (h / h.sum()).T

    def conditional_price(self, driver: str, output: str, bins: int = 10) -> list[tuple[float, float, float, float]]:
        """For each slice of one driver: (driver centre, p10, p50, p90) of the output."""
        x = self.inputs[:, self.model.input_names.index(driver)]
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        y = self.column(output)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = y[(x >= lo) & (x <= hi)]
            rows.append((float((lo + hi) / 2), *(float(np.quantile(sel, q)) for q in (0.1, 0.5, 0.9))))
        return rows


def belief_density(
    model: Model,
    beliefs: dict[str, Belief] | None = None,
    draws: int = 20_000,
    seed: int = 0,
    corner_limit: int = 12,
    correlation: Correlation | None = None,
) -> DensityResult:
    """Build the feasible polytope from the input box, then the density of beliefs over it.

    The polytope is the hull of the model at every corner of the box when the box has
    at most `corner_limit` inputs (2**n corners), otherwise of the belief draws plus
    the base case. Corner hulls are exact for models that are monotone in each input.
    """
    beliefs = beliefs or {}
    inputs = sample_beliefs(model, beliefs, draws, seed, correlation)
    outputs = evaluate_rows(model, inputs)
    n = len(model.inputs)
    if n <= corner_limit:
        lows = np.array([i.low for i in model.inputs])
        highs = np.array([i.high for i in model.inputs])
        picks = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
        corners = np.where(picks, highs, lows)
        feasible_points = np.vstack([evaluate_rows(model, corners), outputs])
    else:
        feasible_points = outputs
    feasible = output_polytope(feasible_points, output_names=model.output_names)
    return DensityResult(model, beliefs, inputs, outputs, feasible, correlation)