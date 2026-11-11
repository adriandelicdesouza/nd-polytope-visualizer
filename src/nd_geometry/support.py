"""Output polytope for any number of inputs, built from support points instead of box corners.

The corner method evaluates the model at all 2**n corners of the input box, which stops being practical
past about a dozen inputs. The output polytope only needs the points that stick out furthest in some
direction of output space. For a direction d, `support_polytope` maximises d . (scaled outputs) over the
input box by coordinate search (each input tried at its low, high, base and the two midpoints, with a
cache), starting from the base case and from the corner suggested by finite-difference signs. The cost
grows with n and the number of directions, not with 2**n. For a model that is monotone in each input
every maximiser is a box corner and the result is exact on those directions; otherwise it is a local
search, so the hull is an inner approximation (it never claims more than the model can produce).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Model
from .output_polytope import OutputPolytope, output_polytope


@dataclass(frozen=True)
class SupportResult:
    """Maximising input vectors and their outputs, one per distinct support point found."""

    model: Model
    inputs: np.ndarray  # (P, n)
    outputs: np.ndarray  # (P, m)
    directions: np.ndarray  # (K, m) unit directions used (in scaled output space)
    evaluations: int
    infeasible: int = 0  # input combinations the model refused (kept out of the hull)

    def polytope(self, extra: np.ndarray | None = None) -> OutputPolytope:
        """Hull of the support points, plus any extra output points (for example belief draws)."""
        points = self.outputs if extra is None else np.vstack([self.outputs, extra])
        return output_polytope(points, output_names=self.model.output_names)

    def vertex_inputs(self, polytope: OutputPolytope) -> list[dict[str, float]]:
        """For each hull vertex, the input values that produce it (empty if it came from an extra point)."""
        found = []
        for v in polytope.vertices:
            hit = np.where(np.all(np.isclose(self.outputs, v, rtol=1e-12, atol=1e-12), axis=1))[0]
            found.append(dict(zip(self.model.input_names, self.inputs[hit[0]])) if len(hit) else {})
        return found

    def input_roles(self, polytope: OutputPolytope) -> dict[str, str]:
        """How each input shapes the polytope: 'low' / 'high' if every vertex pins it to one end of its range
        (a one-sided input), 'trade-off' if vertices use both ends, 'interior' if it sits inside its range."""
        rows = [x for x in self.vertex_inputs(polytope) if x]
        roles = {}
        for item in self.model.inputs:
            values = np.array([x[item.name] for x in rows]) if rows else np.array([item.base])
            at_low, at_high = np.isclose(values, item.low), np.isclose(values, item.high)
            roles[item.name] = ("low" if at_low.all() else "high" if at_high.all()
                                else "trade-off" if at_low.any() and at_high.any() else "interior")
        return roles


def support_polytope(model: Model, directions: int = 300, seed: int = 0, sweeps: int = 8,
                     pilot: int = 64) -> SupportResult:
    """Find the support points of the model's output region over its input box, for `directions` directions."""
    rng = np.random.default_rng(seed)
    names = model.input_names
    lows = np.array([i.low for i in model.inputs])
    highs = np.array([i.high for i in model.inputs])
    base = np.array([i.base for i in model.inputs])
    n, m = len(names), len(model.output_names)
    cache: dict[tuple, np.ndarray] = {}

    def run(x: np.ndarray) -> np.ndarray:
        key = tuple(np.round(x, 12))
        if key not in cache:
            try:
                out = model.evaluate(dict(zip(names, x)))
                cache[key] = np.array([out[o] for o in model.output_names], dtype=float)
            except ValueError:  # an input combination the model refuses (for example WACC below growth)
                cache[key] = np.full(m, np.nan)
        return cache[key]

    sample = np.array([y for y in (run(rng.uniform(lows, highs)) for _ in range(pilot)) if np.all(np.isfinite(y))])
    centre, scale = sample.mean(axis=0), np.where(sample.std(axis=0) > 0, sample.std(axis=0), 1.0)

    axes = np.vstack([np.eye(m), -np.eye(m)])
    random = rng.standard_normal((max(directions - 2 * m, 0), m))
    dirs = np.vstack([axes, random / np.linalg.norm(random, axis=1, keepdims=True)]) if len(random) else axes

    def score(x: np.ndarray, d: np.ndarray) -> float:
        y = run(x)
        return float(d @ ((y - centre) / scale)) if np.all(np.isfinite(y)) else -np.inf

    def climb(x: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, float]:
        best = score(x, d)
        for _ in range(sweeps):
            moved = False
            for j in rng.permutation(n):
                for value in (lows[j], highs[j], base[j], (lows[j] + base[j]) / 2, (base[j] + highs[j]) / 2):
                    trial = x.copy()
                    trial[j] = value
                    s = score(trial, d)
                    if s > best + 1e-12:
                        x, best, moved = trial, s, True
            if not moved:
                break
        return x, best

    points: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for d in dirs:
        signs = np.empty(n)
        for j in range(n):
            lo_x, hi_x = base.copy(), base.copy()
            lo_x[j], hi_x[j] = lows[j], highs[j]
            signs[j] = highs[j] if score(hi_x, d) >= score(lo_x, d) else lows[j]
        winners = [climb(start.copy(), d) for start in (signs, base)]
        x = max(winners, key=lambda w: w[1])[0]
        points[tuple(np.round(x, 12))] = (x, run(x))
    inputs = np.array([p[0] for p in points.values()])
    outputs = np.array([p[1] for p in points.values()])
    bad = sum(1 for y in cache.values() if not np.all(np.isfinite(y)))
    return SupportResult(model, inputs, outputs, dirs, len(cache), bad)