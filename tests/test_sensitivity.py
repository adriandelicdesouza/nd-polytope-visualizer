from __future__ import annotations

import numpy as np

from nd_geometry.sensitivity import SensitivityData
from nd_geometry.evaluation import evaluate_model

def test_sensitivity_data_dimensions():
    data = SensitivityData(
        values=np.zeros((3, 3, 3)),
        parameter_names=("growth", "margin", "wacc"),
        axes=(
            np.array([0.02, 0.06, 0.10]),
            np.array([0.10, 0.175, 0.25]),
            np.array([0.06, 0.09, 0.12]),
        ),
    )

    assert data.dimensions == 3


def test_sensitivity_data_stores_values():
    values = np.arange(8).reshape(2, 2, 2)

    data = SensitivityData(
        values=values,
        parameter_names=("growth", "margin", "wacc"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.25]),
            np.array([0.06, 0.12]),
        ),
    )

    np.testing.assert_array_equal(data.values, values)

import pytest


def test_sensitivity_data_rejects_mismatched_axes():
    with pytest.raises(
        ValueError,
        match="axes must match the number of parameters",
    ):
        SensitivityData(
            values=np.zeros((2, 2)),
            parameter_names=("growth", "margin"),
            axes=(np.array([0.02, 0.10]),),
        )


def test_sensitivity_data_rejects_mismatched_values_shape():
    with pytest.raises(
        ValueError,
        match="values shape must match the axis lengths",
    ):
        SensitivityData(
            values=np.zeros((2, 3)),
            parameter_names=("growth", "margin"),
            axes=(
                np.array([0.02, 0.10]),
                np.array([0.10, 0.25]),
            ),
        )


def test_sensitivity_data_rejects_duplicate_parameter_names():
    with pytest.raises(
        ValueError,
        match="parameter names must be unique",
    ):
        SensitivityData(
            values=np.zeros((2, 2)),
            parameter_names=("growth", "growth"),
            axes=(
                np.array([0.02, 0.10]),
                np.array([0.10, 0.25]),
            ),
        )

def test_sensitivity_data_from_evaluation():
    samples = np.array([
        [0.02, 0.10],
        [0.02, 0.20],
        [0.10, 0.10],
        [0.10, 0.20],
    ])

    result = evaluate_model(
        samples,
        lambda sample: sample[0] + sample[1],
    )

    data = SensitivityData.from_evaluation(
        result=result,
        parameter_names=("growth", "margin"),
        axes=(
            np.array([0.02, 0.10]),
            np.array([0.10, 0.20]),
        ),
    )

    assert data.values.shape == (2, 2)

    np.testing.assert_allclose(
        data.values,
        np.array([
            [0.12, 0.22],
            [0.20, 0.30],
        ]),
    )

def test_sensitivity_data_value_at():
    values = np.arange(8).reshape(2, 2, 2)

    data = SensitivityData(
        values=values,
        parameter_names=("a", "b", "c"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    assert data.value_at((0, 0, 0)) == 0.0
    assert data.value_at((1, 0, 1)) == 5.0


def test_sensitivity_data_value_at_rejects_wrong_dimension():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("a", "b"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    with pytest.raises(
        ValueError,
        match="number of indices must match",
    ):
        data.value_at((0,))

def test_sensitivity_data_value_at_parameters():
    values = np.arange(8).reshape(2, 2, 2)

    data = SensitivityData(
        values=values,
        parameter_names=("a", "b", "c"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    assert data.value_at_parameters((0.0, 0.0, 0.0)) == 0.0
    assert data.value_at_parameters((1.0, 0.0, 1.0)) == 5.0


def test_sensitivity_data_value_at_parameters_rejects_wrong_dimension():
    data = SensitivityData(
        values=np.zeros((2, 2)),
        parameter_names=("a", "b"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    with pytest.raises(
        ValueError,
        match="number of parameters must match",
    ):
        data.value_at_parameters((0.0,))

def test_sensitivity_data_statistics():
    data = SensitivityData(
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        parameter_names=("a", "b"),
        axes=(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
    )

    assert data.minimum == 1.0
    assert data.maximum == 4.0
    assert data.mean == 2.5