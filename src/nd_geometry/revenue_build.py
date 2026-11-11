"""Revenue builds: how a revenue path is produced from named drivers.

A build exposes its drivers as model inputs, so beliefs about the revenue assumptions are
beliefs about the build's own drivers rather than about one growth number.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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

@dataclass(frozen=True)
class Factor:
    """One multiplicative growth factor of a stream, year by year.

    mode "add": the factor in year t is path[t] + x, with x a shift believed to lie in [low, high] (base usually 0).
    mode "scale": the factor in year t is path[t] * x, with x a multiplier believed to lie in [low, high].
    """

    name: str
    path: tuple[float, ...]
    mode: str
    low: float
    base: float
    high: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("add", "scale"):
            raise ValueError(f"factor {self.name!r}: mode must be 'add' or 'scale'")

    def value(self, t: int, x: float) -> float:
        return self.path[t] + x if self.mode == "add" else self.path[t] * x


@dataclass(frozen=True)
class Stream:
    """A revenue stream: base-year revenue grown each year by the product of its factors (1 + factor)."""

    name: str
    base_revenue: float
    factors: tuple[Factor, ...]
    source: str = ""


@dataclass(frozen=True)
class SegmentBuild:
    """Revenue as the sum of streams, each driven by named factors whose shifts or scales are the model inputs.

    Input names are `<stream>_<factor>`, so beliefs are held on the build's own assumptions
    (for example subscription volume, pricing, FX and advertising growth) rather than on one growth rate.
    """

    streams: tuple[Stream, ...]
    name: str = "segment revenue build"

    def __post_init__(self) -> None:
        lengths = {len(f.path) for s in self.streams for f in s.factors}
        if len(lengths) != 1:
            raise ValueError("every factor path must have the same number of years")
        names = [self._input_name(s, f) for s in self.streams for f in s.factors]
        if len(set(names)) != len(names):
            raise ValueError("stream and factor names must give unique input names")

    @staticmethod
    def _input_name(stream: Stream, factor: Factor) -> str:
        return f"{stream.name}_{factor.name}"

    @property
    def years(self) -> int:
        return len(self.streams[0].factors[0].path)

    @property
    def base_revenue(self) -> float:
        return sum(s.base_revenue for s in self.streams)

    def inputs(self) -> tuple[Input, ...]:
        return tuple(Input(self._input_name(s, f), "%", f.low, f.high, f.base) for s in self.streams for f in s.factors)

    def stream_revenues(self, v: Mapping[str, float]) -> dict[str, list[float]]:
        out = {}
        for s in self.streams:
            path = [s.base_revenue]
            for t in range(self.years):
                growth = 1.0
                for f in s.factors:
                    step = 1.0 + f.value(t, v[self._input_name(s, f)])
                    if step <= 0:
                        raise ValueError(f"{s.name}: factor {f.name} gives growth of {step - 1:.1%} or worse in year {t + 1}")
                    growth *= step
                path.append(path[-1] * growth)
            out[s.name] = path
        return out

    def revenues(self, v: Mapping[str, float]) -> list[float]:
        return [sum(col) for col in zip(*self.stream_revenues(v).values())]

    @classmethod
    def from_dict(cls, spec: Mapping) -> SegmentBuild:
        streams = tuple(
            Stream(s["name"], float(s["base_revenue"]), tuple(
                Factor(f["name"], tuple(float(x) for x in f["path"]), f["mode"], float(f["range"]["low"]),
                       float(f["range"]["base"]), float(f["range"]["high"]), f.get("source", ""))
                for f in s["factors"]), s.get("source", ""))
            for s in spec["streams"])
        return cls(streams, spec.get("name", "segment revenue build"))

    @classmethod
    def from_json(cls, path) -> SegmentBuild:
        return cls.from_dict(json.loads(Path(path).read_text()))