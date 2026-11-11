from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class OutputPolytope:
    """The convex polytope spanned by model outputs over a parameter region."""

    points: np.ndarray
    vertices: np.ndarray
    equations: np.ndarray
    volume: float
    output_names: tuple[str, ...]

    @property
    def dimensions(self) -> int:
        """Return the number of outputs (dimensions of the polytope)."""
        return self.points.shape[1]

    def bounds(self) -> np.ndarray:
        """Return (min, max) of each output across the polytope, shape (M, 2)."""
        return np.column_stack(
            (self.vertices.min(axis=0), self.vertices.max(axis=0))
        )

    def contains(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        """Return True if the output point lies inside or on the polytope."""
        point = np.asarray(point, dtype=float)

        if point.shape != (self.dimensions,):
            raise ValueError("point must have one value per output")

        # Each facet row is (normal, offset) with normal . x + offset <= 0 inside.
        values = self.equations[:, :-1] @ point + self.equations[:, -1]

        return bool(np.all(values <= tolerance))


def output_polytope(
    outputs: np.ndarray,
    output_names: tuple[str, ...] | None = None,
) -> OutputPolytope:
    """Build the convex hull of model output points (requires 2 or more outputs)."""
    points = np.asarray(outputs, dtype=float)

    if points.ndim != 2:
        raise ValueError("outputs must be a 2D array")

    count, dimensions = points.shape

    if dimensions < 2:
        raise ValueError(
            "an output polytope needs at least 2 outputs; use the min and max for one output"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("outputs must be finite")

    if count < dimensions + 1:
        raise ValueError(
            f"need at least {dimensions + 1} output points for {dimensions} outputs"
        )

    if output_names is None:
        names = tuple(f"output_{index}" for index in range(dimensions))
    else:
        names = tuple(output_names)

        if len(names) != dimensions or len(set(names)) != dimensions:
            raise ValueError("output_names must be unique and match the number of outputs")

    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise ValueError(
            "outputs do not span a full-dimensional polytope (points are degenerate)"
        ) from error

    return OutputPolytope(
        points=points,
        vertices=points[hull.vertices],
        equations=hull.equations.copy(),
        volume=float(hull.volume),
        output_names=names,
    )
