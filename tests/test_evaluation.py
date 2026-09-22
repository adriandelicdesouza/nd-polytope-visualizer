from __future__ import annotations

import numpy as np
import pytest

from nd_geometry.evaluation import evaluate_model

def test_evaluate_model_returns_evaluation_result():
    samples = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    def model(sample: np.ndarray) -> float:
        return sample[0] + sample[1]

    result = evaluate_model(samples, model)

    np.testing.assert_allclose(
        result.samples,
        samples,
    )

    np.testing.assert_allclose(
        result.outputs,
        np.array([3.0, 7.0, 11.0]),
    )


def test_evaluate_model_output_shape():
    samples = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result = evaluate_model(
        samples,
        lambda sample: sample[0] * sample[1],
    )

    assert result.outputs.shape == (2,)

def test_evaluate_model_rejects_non_2d_samples():
    samples = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="samples must be a 2D array"):
        evaluate_model(
            samples,
            lambda sample: sample[0],
        )

