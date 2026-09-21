from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hypercube:
    """An N-dimensional hypercube."""

    dimensions: int
    size: float = 1.0

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")

        if self.size <= 0:
            raise ValueError("size must be > 0")

    @property
    def vertices(self) -> np.ndarray:
        """Return all hypercube vertices as an array of shape (2**N, N)."""
        values = np.array([-self.size, self.size])

        grids = np.meshgrid(
            *([values] * self.dimensions),
            indexing="ij",
        )

        return np.stack(grids, axis=-1).reshape(-1, self.dimensions)
