"""The cheapest story that justifies a price, for a model with any number of inputs.

Single-driver reverse DCF asks "what must one input be if all others stay at base". With dozens of inputs that is
usually unreachable. Here every input may move: we find the input vector inside the box that reproduces the target
output and moves the inputs the least, measured in half-widths of their ranges (a stand-in for belief surprise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .model import Model


@dataclass(frozen=True)
class Story:
    target: float
    output: str
    reachable: bool
    achieved: float
    cost: float  # sum of squared moves, in half-width units
    moves: dict[str, float]  # input -> (value - base) / half-width, so +-1 is the edge of the range
    values: dict[str, float]

    def lines(self, top: int = 8) -> list[str]:
        if not self.reachable:
            return [f"{self.output} {self.target:.4g} is outside what the input box can produce "
                    f"(closest {self.achieved:.4g})"]
        big = sorted(self.moves.items(), key=lambda kv: -abs(kv[1]))[:top]
        return [f"cheapest story for {self.output} = {self.target:.4g}: total surprise {self.cost:.2f} "
                f"(sum of squared half-width moves)"] + [
            f"  {k}: {self.values[k]:.4g} ({m:+.2f} half-widths)" for k, m in big]


def cheapest_story(model: Model, output: str, target: float, tolerance: float = 1e-6) -> Story:
    names = model.input_names
    lows = np.array([i.low for i in model.inputs])
    highs = np.array([i.high for i in model.inputs])
    base = np.array([i.base for i in model.inputs])
    half = np.maximum((highs - lows) / 2, 1e-12)

    def value(z: np.ndarray) -> float:
        try:
            return model.evaluate(dict(zip(names, base + z * half)))[output]
        except ValueError:
            return float("nan")

    bounds = [((lo - b) / h, (hi - b) / h) for lo, hi, b, h in zip(lows, highs, base, half)]
    z0 = np.zeros(len(names))
    scale = max(abs(target), abs(value(z0)), 1.0)
    res = minimize(lambda z: float(z @ z), z0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda z: (value(z) - target) / scale}],
                   options={"maxiter": 300, "ftol": 1e-12})
    z = res.x
    achieved = value(z)
    reachable = bool(np.isfinite(achieved) and abs(achieved - target) <= max(tolerance * scale, 1e-4 * scale))
    if not reachable:  # find the closest output the box can make, to say how far off we are
        far = minimize(lambda z: ((value(z) - target) / scale) ** 2 if np.isfinite(value(z)) else 1e6, z0,
                       method="SLSQP", bounds=bounds)
        z, achieved = far.x, value(far.x)
    values = dict(zip(names, map(float, base + z * half)))
    return Story(target, output, reachable, float(achieved), float(z @ z), dict(zip(names, map(float, z))), values)