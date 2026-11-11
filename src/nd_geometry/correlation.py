"""Correlated driver beliefs: a Gaussian copula over each driver's marginal belief.

Each driver keeps its own belief (uniform, triangular or truncated normal). The copula only
sets how they move together: correlated normals are mapped to uniforms, and each uniform goes
through its driver's inverse belief CDF, so every marginal is preserved exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.stats import norm

# Assumed co-movement between DCF drivers (Gaussian-copula correlation). These are priors, not estimates:
# `historical_correlation` shrinks them toward what the filings show.
DCF_PRIORS: dict[tuple[str, str], float] = {
    ("wacc", "terminal_growth"): 0.30,  # both follow the rate environment
    ("capex_pct", "da_pct"): 0.60,  # depreciation follows past capex
    ("ebit_margin", "capex_pct"): -0.20,  # heavier investment, thinner margin
    ("ebit_margin", "other_reinvest_pct"): -0.20,
}
BUILD_TO_BUILD = 0.25  # revenue-build drivers share one demand environment
BUILD_TO_MARGIN = 0.25  # operating leverage: faster growth lifts the margin
FINANCIAL = {"ebit_margin", "tax_rate", "capex_pct", "da_pct", "nwc_pct", "other_reinvest_pct", "wacc", "terminal_growth"}


@dataclass(frozen=True)
class Correlation:
    """A validated correlation matrix over named drivers. Drivers not listed are independent."""

    names: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        n = len(self.names)
        if len(set(self.names)) != n:
            raise ValueError("correlation names must be unique")
        if m.shape != (n, n):
            raise ValueError(f"matrix must be {n} x {n}")
        if not np.allclose(m, m.T, atol=1e-9):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(m), 1.0, atol=1e-9) or np.any(np.abs(m) > 1 + 1e-9):
            raise ValueError("correlation matrix needs a unit diagonal and entries within [-1, 1]")
        smallest = float(np.linalg.eigvalsh(m).min()) if n else 0.0
        if smallest < -1e-8:
            raise ValueError(f"correlation matrix is not positive semi-definite (smallest eigenvalue {smallest:.3f}); "
                             "the requested pairwise correlations cannot all hold together")
        object.__setattr__(self, "matrix", m)

    @classmethod
    def from_pairs(cls, names: Sequence[str], pairs: Mapping[tuple[str, str], float]) -> Correlation:
        index = {n: i for i, n in enumerate(names)}
        m = np.eye(len(names))
        for (a, b), rho in pairs.items():
            if a not in index or b not in index:
                raise ValueError(f"unknown driver in pair ({a!r}, {b!r})")
            if a == b or not -1 < rho < 1:
                raise ValueError(f"pair ({a!r}, {b!r}): need two different drivers and |rho| < 1, got {rho}")
            m[index[a], index[b]] = m[index[b], index[a]] = rho
        return cls(tuple(names), m)

    @classmethod
    def independent(cls, names: Sequence[str]) -> Correlation:
        return cls(tuple(names), np.eye(len(names)))

    def rho(self, a: str, b: str) -> float:
        return float(self.matrix[self.names.index(a), self.names.index(b)])

    def aligned(self, names: Sequence[str]) -> np.ndarray:
        """The matrix over `names` (a model's inputs), with drivers this object does not list independent."""
        extra = set(self.names) - set(names)
        if extra:
            raise ValueError(f"correlation lists drivers the model does not have: {sorted(extra)}")
        full = np.eye(len(names))
        where = [list(names).index(n) for n in self.names]
        full[np.ix_(where, where)] = self.matrix
        return full


def nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues and rescale to a unit diagonal: the closest valid correlation matrix, in practice."""
    m = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2
    w, v = np.linalg.eigh(m)
    m = (v * np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(m))
    m = m / np.outer(d, d)
    np.fill_diagonal(m, 1.0)
    return m


def copula_uniforms(matrix: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """(draws, n) uniforms whose normal scores have the given correlation."""
    w, v = np.linalg.eigh(matrix)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    return norm.cdf(rng.standard_normal((draws, matrix.shape[0])) @ root.T)


def default_dcf_correlation(names: Iterable[str]) -> Correlation:
    """Prior correlations for a DCF model's inputs (revenue-build drivers are every input that is not a financial driver)."""
    names = tuple(names)
    build = [n for n in names if n not in FINANCIAL]
    pairs = {p: r for p, r in DCF_PRIORS.items() if p[0] in names and p[1] in names}
    for i, a in enumerate(build):
        for b in build[i + 1:]:
            pairs[(a, b)] = BUILD_TO_BUILD
        if "ebit_margin" in names:
            pairs[(a, "ebit_margin")] = BUILD_TO_MARGIN
    return Correlation.from_pairs(names, pairs)


def historical_correlation(series: Mapping[str, Sequence[tuple[date, float]]], names: Sequence[str],
                           prior: Correlation | None = None, prior_weight: float = 0.7,
                           min_obs: int = 5) -> Correlation:
    """Shrink prior correlations toward the sample correlation of the drivers' filed history.

    Annual history is short, so the sample estimate gets weight `1 - prior_weight` and is used only for
    pairs with at least `min_obs` shared fiscal years. The blend is repaired to a valid correlation matrix.
    """
    names = tuple(names)
    prior = prior or Correlation.independent(names)
    base = prior.aligned(names)
    blended = base.copy()
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            if a not in series or b not in series:
                continue
            sa, sb = dict(series[a]), dict(series[b])
            common = sorted(set(sa) & set(sb))
            if len(common) < min_obs:
                continue
            xa, xb = np.array([sa[d] for d in common]), np.array([sb[d] for d in common])
            if xa.std() == 0 or xb.std() == 0:
                continue
            sample = float(np.corrcoef(xa, xb)[0, 1])
            blended[i, j] = blended[j, i] = prior_weight * base[i, j] + (1 - prior_weight) * sample
    return Correlation(names, nearest_correlation(blended))