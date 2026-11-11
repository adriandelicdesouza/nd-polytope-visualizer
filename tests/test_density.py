import numpy as np
import pytest

from nd_geometry.density import Belief, belief_density, sample_beliefs
from nd_geometry.model import Input, Model, Output


def toy() -> Model:
    return Model(
        name="toy",
        inputs=(Input("x", "u", 0.0, 4.0, 1.0), Input("y", "u", 0.0, 2.0, 1.0)),
        outputs=(Output("s", "u"), Output("d", "u")),
        function=lambda v: {"s": v["x"] + v["y"], "d": v["x"] - v["y"]},
    )


def test_triangular_mean_matches_hand_calc():
    # Triangular mean = (low + mode + high) / 3: x -> 5/3, y -> 1, so E[s] = 8/3.
    result = belief_density(toy(), draws=40_000, seed=1)
    assert result.column("s").mean() == pytest.approx(8 / 3, abs=0.03)


def test_support_is_polytope_projection_and_contains_draws():
    result = belief_density(toy(), draws=2_000)
    assert result.support("s") == pytest.approx((0.0, 6.0))
    assert result.column("s").min() >= 0.0 and result.column("s").max() <= 6.0
    assert all(result.feasible.contains(row) for row in result.outputs[:200])


def test_density_integrates_to_about_one_over_support():
    grid, dens = belief_density(toy(), draws=5_000).density("s", points=400)
    assert np.trapezoid(dens, grid) == pytest.approx(1.0, abs=0.05)


def test_joint_density_sums_to_one():
    _, _, mass = belief_density(toy(), draws=5_000).joint_density("s", "d", bins=20)
    assert mass.sum() == pytest.approx(1.0)


def test_tighter_normal_belief_narrows_the_distribution():
    wide = belief_density(toy(), {"x": Belief("uniform")}, draws=5_000)
    tight = belief_density(toy(), {"x": Belief("normal", 0.05)}, draws=5_000)
    assert tight.column("s").std() < wide.column("s").std()


def test_draws_stay_inside_input_ranges_and_unknown_beliefs_fail():
    inputs = sample_beliefs(toy(), {"x": Belief("normal", 0.5)}, 5_000)
    assert inputs[:, 0].min() >= 0.0 and inputs[:, 0].max() <= 4.0
    with pytest.raises(ValueError):
        sample_beliefs(toy(), {"nope": Belief()}, 10)


def test_conditional_price_rises_with_driver():
    rows = belief_density(toy(), draws=10_000).conditional_price("x", "s", bins=5)
    medians = [r[2] for r in rows]
    assert medians == sorted(medians)