import numpy as np
import pytest

from nd_geometry.parameter_space import ParameterSpace


def test_parameter_space_dimensions():
    space = ParameterSpace(
        names=("growth", "margin", "wacc"),
        lower_bounds=(0.00, 0.10, 0.07),
        upper_bounds=(0.20, 0.25, 0.12),
    )

    assert space.dimensions == 3


def test_parameter_space_bounds():
    space = ParameterSpace(
        names=("growth", "margin"),
        lower_bounds=(0.00, 0.10),
        upper_bounds=(0.20, 0.25),
    )

    np.testing.assert_allclose(
        space.bounds,
        np.array([
            [0.00, 0.20],
            [0.10, 0.25],
        ]),
    )


def test_parameter_space_rejects_empty_space():
    with pytest.raises(
        ValueError,
        match="at least one dimension",
    ):
        ParameterSpace(
            names=(),
            lower_bounds=(),
            upper_bounds=(),
        )


def test_parameter_space_requires_matching_bounds():
    with pytest.raises(
        ValueError,
        match="lower_bounds must match",
    ):
        ParameterSpace(
            names=("growth", "margin"),
            lower_bounds=(0.00,),
            upper_bounds=(0.20, 0.25),
        )


def test_parameter_space_requires_unique_names():
    with pytest.raises(
        ValueError,
        match="parameter names must be unique",
    ):
        ParameterSpace(
            names=("growth", "growth"),
            lower_bounds=(0.00, 0.00),
            upper_bounds=(0.20, 0.20),
        )


def test_parameter_space_requires_valid_bounds():
    with pytest.raises(
        ValueError,
        match="lower bound must be less",
    ):
        ParameterSpace(
            names=("growth",),
            lower_bounds=(0.20,),
            upper_bounds=(0.10,),
        )

def test_sample_grid_shape():
    space = ParameterSpace(
        names=("growth", "margin", "wacc"),
        lower_bounds=(0.00, 0.10, 0.07),
        upper_bounds=(0.20, 0.25, 0.12),
    )

    samples = space.sample_grid(points_per_dimension=3)

    assert samples.shape == (27, 3)


def test_sample_grid_contains_bounds():
    space = ParameterSpace(
        names=("growth", "margin"),
        lower_bounds=(0.00, 0.10),
        upper_bounds=(0.20, 0.25),
    )

    samples = space.sample_grid(points_per_dimension=2)

    np.testing.assert_allclose(
        samples,
        np.array([
            [0.00, 0.10],
            [0.00, 0.25],
            [0.20, 0.10],
            [0.20, 0.25],
        ]),
    )


def test_sample_grid_respects_bounds():
    space = ParameterSpace(
        names=("growth", "margin", "wacc"),
        lower_bounds=(0.00, 0.10, 0.07),
        upper_bounds=(0.20, 0.25, 0.12),
    )

    samples = space.sample_grid(points_per_dimension=5)

    assert np.all(samples >= np.array(space.lower_bounds))
    assert np.all(samples <= np.array(space.upper_bounds))


def test_sample_grid_requires_at_least_two_points():
    space = ParameterSpace(
        names=("growth",),
        lower_bounds=(0.00,),
        upper_bounds=(0.20,),
    )

    with pytest.raises(
        ValueError,
        match="points_per_dimension must be >= 2",
    ):
        space.sample_grid(points_per_dimension=1)

def test_sample_random_shape():
    space = ParameterSpace(
        names=("growth", "margin", "wacc"),
        lower_bounds=(0.00, 0.10, 0.07),
        upper_bounds=(0.20, 0.25, 0.12),
    )

    samples = space.sample_random(
        count=100,
        seed=42,
    )

    assert samples.shape == (100, 3)


def test_sample_random_is_reproducible():
    space = ParameterSpace(
        names=("growth", "margin"),
        lower_bounds=(0.00, 0.10),
        upper_bounds=(0.20, 0.25),
    )

    first = space.sample_random(
        count=10,
        seed=42,
    )

    second = space.sample_random(
        count=10,
        seed=42,
    )

    np.testing.assert_allclose(first, second)


def test_sample_random_respects_bounds():
    space = ParameterSpace(
        names=("growth", "margin", "wacc"),
        lower_bounds=(0.00, 0.10, 0.07),
        upper_bounds=(0.20, 0.25, 0.12),
    )

    samples = space.sample_random(
        count=1000,
        seed=42,
    )

    assert np.all(
        samples >= np.asarray(space.lower_bounds)
    )

    assert np.all(
        samples <= np.asarray(space.upper_bounds)
    )


def test_sample_random_requires_positive_count():
    space = ParameterSpace(
        names=("growth",),
        lower_bounds=(0.00,),
        upper_bounds=(0.20,),
    )

    with pytest.raises(
        ValueError,
        match="count must be >= 1",
    ):
        space.sample_random(count=0)