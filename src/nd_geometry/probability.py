from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import evaluate_rows
from .model import Model


@dataclass(frozen=True)
class RiskEvent:
    """A risk that occurs with some probability and, when it occurs, sets an input.

    When it does not occur, the input takes the `quiet` value. Severity is drawn
    uniformly from [low, high] when it does occur.
    """

    input_name: str
    probability: float
    low: float
    high: float
    quiet: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not self.low <= self.high:
            raise ValueError("event low must be <= high")


@dataclass(frozen=True)
class MonteCarloResult:
    """Joint draws of model inputs and outputs, with which risk events occurred."""

    model: Model
    events: tuple[RiskEvent, ...]
    inputs: np.ndarray
    outputs: np.ndarray
    occurred: np.ndarray
    seed: int

    def quantiles(self, output: str, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> dict[float, float]:
        """Return quantiles of one output across all draws."""
        column = self.outputs[:, self.model.output_names.index(output)]
        return {q: float(np.quantile(column, q)) for q in quantiles}

    def event_frequencies(self) -> dict[str, float]:
        """Return the share of draws in which each event occurred."""
        return {
            event.input_name: float(self.occurred[:, index].mean())
            for index, event in enumerate(self.events)
        }


def monte_carlo(
    model: Model,
    events: tuple[RiskEvent, ...] = (),
    draws: int = 10_000,
    seed: int = 0,
    fixed: dict[str, float] | None = None,
) -> MonteCarloResult:
    """Sample market inputs uniformly and risk events independently, then evaluate the model.

    Events are treated as independent: the chance that two occur together is the
    product of their probabilities. `fixed` pins inputs to a value for every draw.
    """
    names = model.input_names
    event_inputs = [event.input_name for event in events]
    if len(set(event_inputs)) != len(event_inputs):
        raise ValueError("each event must drive a different input")
    for event in events:
        if event.input_name not in names:
            raise ValueError(f"event input {event.input_name!r} is not a model input")
        spec = model.inputs[names.index(event.input_name)]
        if not spec.low <= event.quiet <= spec.high:
            raise ValueError(f"quiet value for {event.input_name!r} must lie in the input range")
    for name in fixed or {}:
        if name not in names:
            raise ValueError(f"fixed input {name!r} is not a model input")
        if name in event_inputs:
            raise ValueError(f"cannot fix {name!r}: a risk event drives it")
    if draws < 1:
        raise ValueError("draws must be >= 1")

    rng = np.random.default_rng(seed)
    lows = np.array([item.low for item in model.inputs])
    highs = np.array([item.high for item in model.inputs])
    inputs = lows + rng.uniform(size=(draws, len(names))) * (highs - lows)

    occurred = np.zeros((draws, len(events)), dtype=bool)
    for index, event in enumerate(events):
        column = names.index(event.input_name)
        occurred[:, index] = rng.uniform(size=draws) < event.probability
        severity = event.low + rng.uniform(size=draws) * (event.high - event.low)
        inputs[:, column] = np.where(occurred[:, index], severity, event.quiet)

    for name, value in (fixed or {}).items():
        inputs[:, names.index(name)] = value

    outputs = evaluate_rows(model, inputs)
    return MonteCarloResult(
        model=model,
        events=tuple(events),
        inputs=inputs,
        outputs=outputs,
        occurred=occurred,
        seed=seed,
    )


def quantile_profile(
    model: Model,
    events: tuple[RiskEvent, ...],
    vary_input: str,
    output: str,
    points: int = 9,
    draws: int = 4000,
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, list[float]]:
    """Trace output quantiles as one input is fixed at each of `points` values.

    A risk event on `vary_input` is ignored, because the profile sets that input itself.
    """
    spec = model.inputs[model.input_names.index(vary_input)]
    values = np.linspace(spec.low, spec.high, points)
    # The profile sets this input directly, so any event driving it is removed.
    events = tuple(event for event in events if event.input_name != vary_input)
    profile: dict[str, list[float]] = {"value": [float(v) for v in values]}
    for q in quantiles:
        profile[f"p{round(q * 100)}"] = []

    for value in values:
        result = monte_carlo(model, events, draws=draws, seed=seed, fixed={vary_input: float(value)})
        for q in quantiles:
            profile[f"p{round(q * 100)}"].append(result.quantiles(output, (q,))[q])
    return profile
