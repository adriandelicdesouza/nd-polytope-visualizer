from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Model outputs associated with parameter samples."""

    samples: np.ndarray
    outputs: np.ndarray


def evaluate_model(
    samples: np.ndarray,
    model: Callable[[np.ndarray], float],
) -> EvaluationResult:
    """Evaluate a model for every parameter sample."""
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")

    outputs = np.asarray(
        [model(sample) for sample in samples],
        dtype=float,
    )

    return EvaluationResult(
        samples=samples,
        outputs=outputs,
    )

@dataclass(frozen=True)
class VectorEvaluationResult:
    """Multi-output model results associated with parameter samples."""

    samples: np.ndarray
    outputs: np.ndarray


def evaluate_vector_model(
    samples: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
) -> VectorEvaluationResult:
    """Evaluate a model that returns a vector of outputs for every sample."""
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")

    if samples.shape[0] == 0:
        raise ValueError("samples must contain at least one point")

    rows = []

    for sample in samples:
        output = np.asarray(model(sample), dtype=float)

        if output.ndim != 1:
            raise ValueError("model must return a 1D vector of outputs")

        rows.append(output)

    if len({len(row) for row in rows}) != 1:
        raise ValueError("model must return the same number of outputs for every sample")

    return VectorEvaluationResult(
        samples=samples,
        outputs=np.vstack(rows),
    )
