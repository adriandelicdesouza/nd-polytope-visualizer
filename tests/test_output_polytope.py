from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from nd_geometry.evaluation import evaluate_vector_model
from nd_geometry.output_polytope import output_polytope
from nd_geometry.parameter_space import ParameterSpace


def test_vector_model_returns_one_row_per_sample():
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])

    def model(sample: np.ndarray) -> np.ndarray:
        return np.array([sample[0] + sample[1], sample[0] * sample[1]])

    result = evaluate_vector_model(samples, model)

    np.testing.assert_allclose(result.outputs, [[3.0, 2.0], [7.0, 12.0]])


def test_vector_model_rejects_inconsistent_output_counts():
    samples = np.array([[1.0], [2.0]])

    def model(sample: np.ndarray) -> np.ndarray:
        return np.ones(int(sample[0]))

    with pytest.raises(ValueError, match="same number of outputs"):
        evaluate_vector_model(samples, model)


def test_square_output_polytope_has_unit_area_and_four_vertices():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])

    polytope = output_polytope(points, output_names=("x", "y"))

    assert polytope.volume == pytest.approx(1.0)
    assert len(polytope.vertices) == 4
    assert polytope.output_names == ("x", "y")
    np.testing.assert_allclose(polytope.bounds(), [[0.0, 1.0], [0.0, 1.0]])


def test_contains_distinguishes_inside_boundary_and_outside():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    polytope = output_polytope(points)

    assert polytope.contains(np.array([0.5, 0.5]))
    assert polytope.contains(np.array([1.0, 1.0]))  # on the hypotenuse
    assert not polytope.contains(np.array([2.0, 2.0]))


def test_three_output_polytope_volume_matches_scipy():
    points = np.random.default_rng(0).uniform(size=(40, 3))

    polytope = output_polytope(points)

    assert polytope.dimensions == 3
    assert polytope.volume == pytest.approx(ConvexHull(points).volume)


def test_linear_model_image_is_exact_polytope_of_box_corners():
    matrix = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    space = ParameterSpace(
        names=("a", "b", "c"),
        lower_bounds=(0.0, 0.0, 0.0),
        upper_bounds=(1.0, 1.0, 1.0),
    )

    grid = space.sample_grid(points_per_dimension=4)
    result = evaluate_vector_model(grid, lambda x: matrix @ x)
    polytope = output_polytope(result.outputs)

    corners = np.array(
        [[a, b, c] for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0)]
    )
    exact = ConvexHull(corners @ matrix.T)

    assert polytope.volume == pytest.approx(exact.volume)
    np.testing.assert_allclose(polytope.bounds(), [[0.0, 3.0], [0.0, 4.0]])


def test_degenerate_points_raise_clear_error():
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    with pytest.raises(ValueError, match="linearly dependent"):
        output_polytope(collinear)


def test_linearly_dependent_outputs_are_reported():
    values = np.random.default_rng(1).uniform(size=(30, 1))
    points = np.hstack([values, 2.0 * values + 1.0])

    with pytest.raises(ValueError, match="span 1 of 2"):
        output_polytope(points)


def test_single_output_is_rejected():
    with pytest.raises(ValueError, match="at least 2 outputs"):
        output_polytope(np.array([[1.0], [2.0], [3.0]]))
