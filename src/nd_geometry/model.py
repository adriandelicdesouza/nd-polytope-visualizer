from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Input:
    """A named model input with its unit, range and base-case value."""

    name: str
    unit: str
    low: float
    high: float
    base: float

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"input name {self.name!r} must be a valid identifier")
        if not self.low < self.high:
            raise ValueError(f"input {self.name!r}: low must be less than high")
        if not self.low <= self.base <= self.high:
            raise ValueError(f"input {self.name!r}: base must lie within [low, high]")


@dataclass(frozen=True)
class Output:
    """A named model output with its unit."""

    name: str
    unit: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"output name {self.name!r} must be a valid identifier")


ModelFunction = Callable[[dict[str, float]], Mapping[str, float]]


@dataclass(frozen=True)
class Model:
    """A model template: named inputs, named outputs, and a function linking them."""

    name: str
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    function: ModelFunction

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("a model needs at least one input")
        if len(self.outputs) < 2:
            raise ValueError("a model needs at least two outputs for an output polytope")
        if len({item.name for item in self.inputs}) != len(self.inputs):
            raise ValueError("input names must be unique")
        if len({item.name for item in self.outputs}) != len(self.outputs):
            raise ValueError("output names must be unique")

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.outputs)

    def base_case(self) -> dict[str, float]:
        """Return the base-case input values."""
        return {item.name: item.base for item in self.inputs}

    def evaluate(self, values: Mapping[str, float]) -> dict[str, float]:
        """Run the model on named inputs and return named, validated outputs."""
        missing = set(self.input_names) - set(values)
        unknown = set(values) - set(self.input_names)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown inputs: {sorted(unknown)}")

        result = dict(self.function({name: float(values[name]) for name in self.input_names}))
        if set(result) != set(self.output_names):
            raise ValueError(f"model must return exactly the outputs {sorted(self.output_names)}")

        for name in self.output_names:
            if not np.isfinite(result[name]):
                raise ValueError(f"output {name!r} is not finite at these inputs")

        return {name: float(result[name]) for name in self.output_names}
