import numpy as np
import pytest

from nd_geometry.rotations import rotate


def test_zero_rotation():
    points = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, 4.0],
    ])

    rotated = rotate(points, 0, 1, 0.0)

    np.testing.assert_allclose(rotated, points)


def test_90_degree_rotation():
    points = np.array([[1.0, 0.0]])

    rotated = rotate(
        points,
        axis_a=0,
        axis_b=1,
        angle=np.pi / 2,
    )

    np.testing.assert_allclose(
        rotated,
        [[0.0, 1.0]],
        atol=1e-12,
    )


def test_rotation_preserves_distance():
    points = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [-2.0, 1.0, 0.5, 3.0],
    ])

    original_distance = np.linalg.norm(points[0] - points[1])

    rotated = rotate(
        points,
        axis_a=0,
        axis_b=3,
        angle=0.73,
    )

    rotated_distance = np.linalg.norm(rotated[0] - rotated[1])

    np.testing.assert_allclose(
        rotated_distance,
        original_distance,
    )


def test_rotation_only_changes_selected_axes():
    points = np.array([[1.0, 2.0, 3.0, 4.0]])

    rotated = rotate(
        points,
        axis_a=0,
        axis_b=2,
        angle=np.pi / 4,
    )

    np.testing.assert_allclose(
        rotated[:, [1, 3]],
        points[:, [1, 3]],
    )


def test_arbitrary_6d_rotation():
    points = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    rotated = rotate(
        points,
        axis_a=2,
        axis_b=5,
        angle=np.pi / 2,
    )

    np.testing.assert_allclose(
        rotated,
        [[1.0, 2.0, -6.0, 4.0, 5.0, 3.0]],
        atol=1e-12,
    )


def test_high_dimensional_rotation():
    points = np.arange(20, dtype=float).reshape(1, 20)

    rotated = rotate(
        points,
        axis_a=3,
        axis_b=17,
        angle=0.42,
    )

    assert rotated.shape == points.shape

    # Every axis other than 3 and 17 must remain unchanged.
    unaffected_axes = [
        i for i in range(20)
        if i not in (3, 17)
    ]

    np.testing.assert_allclose(
        rotated[:, unaffected_axes],
        points[:, unaffected_axes],
    )


@pytest.mark.parametrize(
    "axis_a, axis_b",
    [
        (0, 1),
        (0, 3),
        (1, 4),
        (2, 5),
    ],
)
def test_multiple_rotation_planes(axis_a, axis_b):
    points = np.ones((10, 6))

    rotated = rotate(
        points,
        axis_a,
        axis_b,
        0.5,
    )

    assert rotated.shape == points.shape


def test_invalid_axis_a():
    points = np.zeros((1, 4))

    with pytest.raises(ValueError):
        rotate(points, -1, 2, 0.5)


def test_invalid_axis_b():
    points = np.zeros((1, 4))

    with pytest.raises(ValueError):
        rotate(points, 0, 4, 0.5)


def test_same_axes():
    points = np.zeros((1, 4))

    with pytest.raises(ValueError):
        rotate(points, 2, 2, 0.5)


def test_invalid_points_shape():
    points = np.zeros(4)

    with pytest.raises(ValueError):
        rotate(points, 0, 1, 0.5)
