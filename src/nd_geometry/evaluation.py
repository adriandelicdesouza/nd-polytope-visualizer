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