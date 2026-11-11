"""Revenue builds: how a revenue path is produced from named drivers.

A build exposes its drivers as model inputs, so beliefs about the revenue assumptions are
beliefs about the build's own drivers rather than about one growth number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .model import Input


class RevenueBuild(Protocol):
    name: str
    base_revenue: float
    years: int

    def inputs(self) -> tuple[Input, ...]: ...

    def revenues(self, v: Mapping[str, float]) -> list[float]:
        """Revenue for the base year followed by each forecast year."""
        ...


def _input(name: str, triple: tuple[float, float, float]) -> Input:
    low, base, high = triple
    return Input(name, "%", low, high, base)


@dataclass(frozen=True)
class TopDownBuild:
    """One growth rate that fades linearly to a target by the last forecast year."""

    base_revenue: float
    years: int
    growth: tuple[float, float, float]  # low, base, high
    fade: tuple[float, float, float]
    name: str = "top-down growth with fade"

    def inputs(self) -> tuple[Input, ...]:
        return (_input("revenue_growth", self.growth), _input("growth_fade_to", self.fade))

    def revenues(self, v: Mapping[str, float]) -> list[float]:
        g1, g_end = v["revenue_growth"], v.get("growth_fade_to", v["revenue_growth"])
        path = [self.base_revenue]
        for t in range(1, self.years + 1):
            g = g1 + (g_end - g1) * (t - 1) / (self.years - 1) if self.years > 1 else g1
            path.append(path[-1] * (1 + g))
        return path