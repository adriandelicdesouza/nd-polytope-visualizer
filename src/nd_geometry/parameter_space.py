from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterSpace:
    """An N-dimensional bounded parameter space."""

    names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        dimensions = len(self.names)

        if dimensions < 1:
            raise ValueError("parameter space must have at least one dimension")

        if len(self.lower_bounds) != dimensions:
            raise ValueError("lower_bounds must match the number of parameters")

        if len(self.upper_bounds) != dimensions:
            raise ValueError("upper_bounds must match the number of parameters")

        if len(set(self.names)) != dimensions:
            raise ValueError("parameter names must be unique")

        if any(
            lower >= upper
            for lower, upper in zip(
                self.lower_bounds,
                self.upper_bounds,
            )
        ):
            raise ValueError("each lower bound must be less than its upper bound")

    @property
    def dimensions(self) -> int:
        """Return the number of parameters."""
        return len(self.names)

    @property
    def bounds(self) -> np.ndarray:
        """Return bounds as an array of shape (N, 2)."""
        return np.column_stack(
            (self.lower_bounds, self.upper_bounds)
        )

    def sample_grid(self, points_per_dimension: int) -> np.ndarray:
        """Generate a Cartesian grid of parameter values."""
        if points_per_dimension < 2:
            raise ValueError("points_per_dimension must be >= 2")

        axes = [
            np.linspace(lower, upper, points_per_dimension)
            for lower, upper in zip(
                self.lower_bounds,
                self.upper_bounds,
            )
        ]

        grids = np.meshgrid(*axes, indexing="ij")

        return np.stack(grids, axis=-1).reshape(
            -1,
            self.dimensions,
        )

    def sample_random(
        self,
        count: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate uniformly distributed random parameter values."""
        if count < 1:
            raise ValueError("count must be >= 1")

        rng = np.random.default_rng(seed)

        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)

        return rng.uniform(
            lower,
            upper,
            size=(count, self.dimensions),
        )