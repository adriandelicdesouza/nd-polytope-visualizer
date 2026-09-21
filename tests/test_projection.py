import numpy as np
import pytest

from nd_geometry.projection import project


def test_4d_to_3d():
    points = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])

    projected = project(points, 3)

    expected = np.array([
        [1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0],
    ])

    np.testing.assert_allclose(projected, expected)


def test_6d_to_3d():
    points = np.arange(18, dtype=float).reshape(3, 6)

    projected = project(points, 3)

    assert projected.shape == (3, 3)

    np.testing.assert_allclose(
        projected,
        points[:, :3],
    )


def test_custom_axes():
    points = np.array([
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    ])

    projected = project(
        points,
        target_dimensions=3,
        axes=(0, 3, 5),
    )

    np.testing.assert_allclose(
        projected,
        [[10.0, 40.0, 60.0]],
    )


def test_projection_does_not_modify_input():
    points = np.array([
        [1.0, 2.0, 3.0, 4.0],
    ])

    original = points.copy()

    project(points, 3)

    np.testing.assert_array_equal(points, original)


def test_identity_projection():
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    projected = project(points, 3)

    np.testing.assert_allclose(projected, points)


def test_invalid_target_dimension():
    points = np.zeros((2, 4))

    with pytest.raises(ValueError):
        project(points, 5)


def test_invalid_target_dimension_zero():
    points = np.zeros((2, 4))

    with pytest.raises(ValueError):
        project(points, 0)


def test_invalid_axis():
    points = np.zeros((2, 4))

    with pytest.raises(ValueError):
        project(
            points,
            target_dimensions=3,
            axes=(0, 1, 4),
        )


def test_duplicate_axes():
    points = np.zeros((2, 4))

    with pytest.raises(ValueError):
        project(
            points,
            target_dimensions=3,
            axes=(0, 0, 1),
        )


def test_wrong_number_of_axes():
    points = np.zeros((2, 4))

    with pytest.raises(ValueError):
        project(
            points,
            target_dimensions=3,
            axes=(0, 1),
        )

@pytest.mark.parametrize(
    "source_dimensions,target_dimensions",
    [
        (3, 2),
        (4, 3),
        (5, 3),
        (6, 3),
        (8, 4),
        (10, 5),
        (20, 3),
    ],
)
def test_arbitrary_dimension_reduction(
    source_dimensions,
    target_dimensions,
):
    points = np.zeros((10, source_dimensions))

    projected = project(
        points,
        target_dimensions,
    )

    assert projected.shape == (
        10,
        target_dimensions,
    )