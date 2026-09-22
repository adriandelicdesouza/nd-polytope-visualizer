from __future__ import annotations

from dataclasses import dataclass
from .evaluation import EvaluationResult
import numpy as np


@dataclass(frozen=True)
class SensitivityData:
    """Model outputs associated with an N-dimensional parameter grid."""

    values: np.ndarray
    parameter_names: tuple[str, ...]
    axes: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        dimensions = len(self.parameter_names)

        if dimensions < 1:
            raise ValueError("sensitivity data must have at least one dimension")

        if len(self.axes) != dimensions:
            raise ValueError("axes must match the number of parameters")

        if self.values.ndim != dimensions:
            raise ValueError("values dimensions must match the number of parameters")

        expected_shape = tuple(len(axis) for axis in self.axes)

        if self.values.shape != expected_shape:
            raise ValueError("values shape must match the axis lengths")

        if len(set(self.parameter_names)) != dimensions:
            raise ValueError("parameter names must be unique")

    @classmethod
    def from_evaluation(
        cls,
        result: EvaluationResult,
        parameter_names: tuple[str, ...],
        axes: tuple[np.ndarray, ...],
    ) -> "SensitivityData":
        """Create sensitivity data from model evaluation results."""
        values = np.asarray(result.outputs, dtype=float)

        expected_count = np.prod(
            [len(axis) for axis in axes],
            dtype=int,
        )

        if values.size != expected_count:
            raise ValueError(
                "evaluation output count must match the parameter grid size"
            )

        shape = tuple(len(axis) for axis in axes)

        return cls(
            values=values.reshape(shape),
            parameter_names=parameter_names,
            axes=axes,
        )

    @property
    def dimensions(self) -> int:
        """Return the number of parameter dimensions."""
        return len(self.parameter_names)

    def value_at(self, indices: tuple[int, ...]) -> float:
        """Return the model output at a grid index."""
        if len(indices) != self.dimensions:
            raise ValueError(
                "number of indices must match the number of parameters"
            )

        return float(self.values[indices])

    def value_at_parameters(
        self,
        parameters: tuple[float, ...],
    ) -> float:
        """Return the model output at the nearest grid point."""
        if len(parameters) != self.dimensions:
            raise ValueError(
                "number of parameters must match the number of dimensions"
            )

        indices = tuple(
            int(np.argmin(np.abs(axis - value)))
            for axis, value in zip(self.axes, parameters)
        )

        return self.value_at(indices)

    @property
    def minimum(self) -> float:
        """Return the minimum model output."""
        return float(np.min(self.values))


    @property
    def maximum(self) -> float:
        """Return the maximum model output."""
        return float(np.max(self.values))


    @property
    def mean(self) -> float:
        """Return the mean model output."""
        return float(np.mean(self.values))