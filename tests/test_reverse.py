import numpy as np
import pytest

from company_fixtures import make_raw
from nd_geometry.dcf import company_model
from nd_geometry.density import belief_density
from nd_geometry.filings import company_from_facts
from nd_geometry.reverse import reverse_dcf, solve_driver


@pytest.fixture(scope="module")
def setup():
    co = company_from_facts(make_raw(), ticker="STDY", overrides={"wacc": (0.09, 0.10, 0.12)})
    model, _ = company_model(co)
    result = belief_density(model, draws=6000, seed=1)
    return model, result


def test_solve_round_trips_to_the_target(setup):
    model, _ = setup
    base = model.base_case()
    target = model.evaluate({**base, "ebit_margin": 0.22})["implied_price"]
    solved = solve_driver(model, "ebit_margin", target)
    assert solved.status == "solved" and solved.value == pytest.approx(0.22, abs=1e-6)


def test_solve_reports_unreachable_targets(setup):
    model, _ = setup
    assert solve_driver(model, "tax_rate", 1e9).value is None
    assert "more than high" in solve_driver(model, "ebit_margin", 1e9).status
    assert "less than low" in solve_driver(model, "ebit_margin", -1e9).status


def test_market_at_the_median_leaves_beliefs_roughly_unchanged(setup):
    model, result = setup
    median = float(np.median(result.column("implied_price")))
    rev = reverse_dcf(result, median)
    assert rev.attainable and rev.belief_percentile == pytest.approx(0.5, abs=0.01)
    assert rev.prob_at_or_above == pytest.approx(0.5, abs=0.01)
    assert all(abs(d.shift_in_sd) < 0.35 for d in rev.implied.values())


def test_a_high_market_price_pulls_value_drivers_up_and_cuts_probability(setup):
    model, result = setup
    median = float(np.median(result.column("implied_price")))
    high = reverse_dcf(result, 1.5 * median)
    assert high.prob_at_or_above < 0.5 < reverse_dcf(result, 0.8 * median).prob_at_or_above
    assert high.implied["ebit_margin"].shift_in_sd > 0.3 and high.implied["wacc"].shift_in_sd < -0.3
    assert high.implied_point_price == pytest.approx(1.5 * median, rel=0.25)


def test_price_outside_the_feasible_interval_is_flagged(setup):
    _, result = setup
    top = result.support("implied_price")[1]
    rev = reverse_dcf(result, 3 * top)
    assert not rev.attainable and rev.prob_at_or_above == 0.0
    assert any("OUTSIDE" in line for line in rev.lines())

def test_far_tail_price_widens_the_kernel_to_keep_enough_draws(setup):
    _, result = setup
    top = result.support("implied_price")[1]
    near = reverse_dcf(result, float(np.median(result.column("implied_price"))))
    tail = reverse_dcf(result, 0.97 * top)
    assert near.bandwidth_used == 0.03 and tail.bandwidth_used > 0.03
    assert tail.effective_draws >= 200 or tail.bandwidth_used >= 0.5
