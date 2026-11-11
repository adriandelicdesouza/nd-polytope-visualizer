"""Full n-dimensional report of an output polytope: size, shape, extremes, and what builds each vertex."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .output_polytope import OutputPolytope
from .support import SupportResult


@dataclass(frozen=True)
class HullReport:
    dimensions: int
    vertices: int
    facets: int
    volume: float
    bounds: dict[str, tuple[float, float]]
    support: dict[str, tuple[float, float]]  # (min, max) of each output with the input values that reach it
    extremes: dict[str, dict[str, dict[str, float]]]
    roles: dict[str, str]
    evaluations: int
    infeasible: int
    exact_for_monotone_models: bool = True

    def lines(self, top: int = 6) -> list[str]:
        out = [f"output polytope: {self.dimensions} outputs, {self.vertices} vertices, {self.facets} facets, "
               f"volume {self.volume:.4g}",
               f"{self.evaluations} model runs, {self.infeasible} infeasible input combinations skipped "
               "(vertices are support points: exact for models monotone in each input, an inner bound otherwise)"]
        for name, (lo, hi) in self.bounds.items():
            out.append(f"  {name}: {lo:.4g} .. {hi:.4g}")
        if self.roles:
            groups: dict[str, list[str]] = {}
            for k, r in self.roles.items():
                groups.setdefault(r, []).append(k)
            for r, ks in groups.items():
                out.append(f"  {r} inputs ({len(ks)}): " + ", ".join(ks[:top]) + (" ..." if len(ks) > top else ""))
        for name, ends in self.extremes.items():
            for end, x in ends.items():
                moved = sorted(x.items(), key=lambda kv: -abs(kv[1]))[:top]
                out.append(f"  {name} {end} reached with " + ", ".join(f"{k}={v:.4g}" for k, v in moved))
        return out


def hull_report(polytope: OutputPolytope, result: SupportResult | None = None) -> HullReport:
    names = polytope.output_names
    bounds = {n: (float(a), float(b)) for n, (a, b) in zip(names, polytope.bounds())}
    roles: dict[str, str] = {}
    extremes: dict[str, dict[str, dict[str, float]]] = {}
    evaluations = infeasible = 0
    if result is not None:
        roles = result.input_roles(polytope)
        evaluations, infeasible = result.evaluations, result.infeasible
        for j, n in enumerate(names):
            extremes[n] = {}
            for end, idx in (("min", int(np.argmin(result.outputs[:, j]))), ("max", int(np.argmax(result.outputs[:, j])))):
                extremes[n][end] = dict(zip(result.model.input_names, map(float, result.inputs[idx])))
    return HullReport(polytope.dimensions, len(polytope.vertices), len(polytope.equations), polytope.volume,
                      bounds, bounds, extremes, roles, evaluations, infeasible)