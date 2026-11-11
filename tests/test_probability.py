from __future__ import annotations

import numpy as np
import pytest

from nd_geometry.model import Input, Model, Output
from nd_geometry.probability import RiskEvent, monte_carlo, quantile_profile
from nd_geometry.probability import RiskEvent as Event

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from sea_freight_model import EVENTS, MODEL  # noqa: E402


def _toy():
    return Model(
        name="toy",
        inputs=(Input("a", "u", 0.0, 1.0, 0.0), Input("b", "u", 0.0, 10.0, 0.0)),
        outputs=(Output("x", "u"), Output("y", "u")),
        function=lambda v: {"x": v["a"] + v["b"], "y": v["a"]},
    )


def test_event_rejects_bad_probability_and_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        RiskEvent("a", probability=1.5, low=0.0, high=1.0)
    with pytest.raises(ValueError, match="low must be"):
        RiskEvent("a", probability=0.5, low=2.0, high=1.0)


def test_event_frequency_matches_probability():
    model = _toy()
    result = monte_carlo(model, (RiskEvent("b", 0.3, 5.0, 9.0),), draws=20000, seed=4)

    assert result.event_frequencies()["b"] == pytest.approx(0.3, abs=0.02)


def test_quiet_value_applies_when_event_does_not_occur():
    model = _toy()
    result = monte_carlo(model, (RiskEvent("b", 0.0, 5.0, 9.0, quiet=2.0),), draws=50, seed=1)

    np.testing.assert_allclose(result.inputs[:, 1], 2.0)


def test_independent_events_combine_by_product():
    model = _toy()
    events = (RiskEvent("a", 0.5, 0.0, 1.0), RiskEvent("b", 0.2, 1.0, 2.0))
    result = monte_carlo(model, events, draws=40000, seed=9)

    both = (result.occurred[:, 0] & result.occurred[:, 1]).mean()
    assert both == pytest.approx(0.5 * 0.2, abs=0.01)


def test_monte_carlo_is_reproducible_with_seed():
    first = monte_carlo(_toy(), draws=200, seed=3)
    second = monte_carlo(_toy(), draws=200, seed=3)

    np.testing.assert_allclose(first.outputs, second.outputs)


def test_fixed_input_is_held_for_every_draw():
    result = monte_carlo(_toy(), draws=100, seed=2, fixed={"b": 4.0})

    assert np.all(result.inputs[:, 1] == 4.0)


def test_invalid_event_setups_are_rejected():
    model = _toy()
    with pytest.raises(ValueError, match="not a model input"):
        monte_carlo(model, (RiskEvent("zzz", 0.5, 0.0, 1.0),))
    with pytest.raises(ValueError, match="quiet value"):
        monte_carlo(model, (RiskEvent("b", 0.5, 0.0, 1.0, quiet=20.0),))
    with pytest.raises(ValueError, match="cannot fix"):
        monte_carlo(model, (RiskEvent("b", 0.5, 0.0, 1.0),), fixed={"b": 1.0})


def test_quantiles_are_ordered():
    result = monte_carlo(_toy(), draws=2000, seed=5)
    q = result.quantiles("x")

    assert q[0.1] <= q[0.5] <= q[0.9]


def test_sea_freight_model_runs_with_risk_events():
    result = monte_carlo(MODEL, EVENTS, draws=2000, seed=1)
    stats = result.quantiles("cost_per_teu")

    assert result.outputs.shape == (2000, 2)
    assert stats[0.5] > 0
    assert result.event_frequencies()["port_delay_days"] == pytest.approx(44 / 1827, abs=0.005)


def test_delays_raise_transit_time_in_the_profile():
    profile = quantile_profile(MODEL, EVENTS, "port_delay_days", "transit_days", points=3, draws=800, seed=2)

    assert profile["p50"][-1] > profile["p50"][0]
    assert len(profile["value"]) == 3


def test_positive_correlation_raises_joint_occurrence_and_keeps_marginals():
    model = _toy()
    events = (RiskEvent("a", 0.3, 0.0, 1.0), RiskEvent("b", 0.3, 1.0, 2.0))
    independent = monte_carlo(model, events, draws=60000, seed=11)
    correlated = monte_carlo(model, events, draws=60000, seed=11, correlation=np.array([[1.0, 0.7], [0.7, 1.0]]))

    joint_independent = (independent.occurred[:, 0] & independent.occurred[:, 1]).mean()
    joint_correlated = (correlated.occurred[:, 0] & correlated.occurred[:, 1]).mean()

    assert joint_independent == pytest.approx(0.09, abs=0.01)
    assert joint_correlated > 0.15
    assert correlated.event_frequencies()["a"] == pytest.approx(0.3, abs=0.01)
    assert correlated.event_frequencies()["b"] == pytest.approx(0.3, abs=0.01)


def test_identity_correlation_matches_independent_sampling_in_distribution():
    model = _toy()
    events = (RiskEvent("a", 0.4, 0.0, 1.0), RiskEvent("b", 0.2, 1.0, 2.0))
    result = monte_carlo(model, events, draws=40000, seed=5, correlation=np.eye(2))

    joint = (result.occurred[:, 0] & result.occurred[:, 1]).mean()
    assert joint == pytest.approx(0.08, abs=0.01)


def test_negative_correlation_lowers_joint_occurrence():
    model = _toy()
    events = (RiskEvent("a", 0.5, 0.0, 1.0), RiskEvent("b", 0.5, 1.0, 2.0))
    result = monte_carlo(model, events, draws=40000, seed=8, correlation=np.array([[1.0, -0.8], [-0.8, 1.0]]))

    # Gaussian copula at the median thresholds: P(both) = 1/4 + arcsin(rho) / (2 pi).
    expected = 0.25 + np.arcsin(-0.8) / (2 * np.pi)
    joint = (result.occurred[:, 0] & result.occurred[:, 1]).mean()
    assert joint == pytest.approx(expected, abs=0.01)


def test_invalid_correlation_matrices_are_rejected():
    model = _toy()
    events = (RiskEvent("a", 0.3, 0.0, 1.0), RiskEvent("b", 0.3, 1.0, 2.0))
    with pytest.raises(ValueError, match="k x k"):
        monte_carlo(model, events, correlation=np.eye(3))
    with pytest.raises(ValueError, match="ones on the diagonal"):
        monte_carlo(model, events, correlation=np.array([[2.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semi-definite"):
        monte_carlo(model, events, correlation=np.array([[1.0, 1.5], [1.5, 1.0]]))


def test_sea_freight_base_case_matches_hand_calculation():
    base = MODEL.evaluate(MODEL.base_case())

    # 10,600 nm at 20 knots is 22.08 sea days. Fuel 225 * (20/24)^3 t/day at 745.3 USD/t VLSFO.
    # Canal 500,000. Port fees: 2 calls x (40,000 + 0.50 x 90,000 GT). Carried: 8,000 x 0.85 TEU.
    sea_days = 10600 / (20 * 24)
    fuel = 225 * (20 / 24) ** 3 * sea_days * (90 * 7.53 * 1.10)
    charter = 50000 * sea_days
    ports = 2 * (40000 + 0.50 * 90000)
    assert base["transit_days"] == pytest.approx(sea_days)
    assert base["cost_per_teu"] == pytest.approx((fuel + charter + 500000 + ports) / (8000 * 0.85))


def test_port_fees_and_canal_toll_are_added_per_teu():
    base = MODEL.base_case()
    no_fees = MODEL.evaluate({**base, "canal_toll_usd": 0.0, "port_fixed_fee_usd": 20000.0, "port_dues_usd_per_gt": 0.25})
    with_fees = MODEL.evaluate(base)
    assert with_fees["cost_per_teu"] > no_fees["cost_per_teu"]


def test_sea_freight_rises_with_crude_and_falls_with_speed():
    base = MODEL.base_case()
    assert MODEL.evaluate({**base, "brent_price": 110.0})["cost_per_teu"] > MODEL.evaluate(base)["cost_per_teu"]
    assert MODEL.evaluate({**base, "ship_speed_knots": 16.0})["transit_days"] > MODEL.evaluate(base)["transit_days"]


def test_historical_severities_are_resampled_and_bounded():
    from nd_geometry.probability import RiskEvent as Ev

    event = Ev.from_history("b", probability=1.0, severities=(2.0, 4.0))
    result = monte_carlo(_toy(), (event,), draws=500, seed=6)

    assert set(np.unique(result.inputs[:, 1])) <= {2.0, 4.0}
    assert event.low == 2.0 and event.high == 4.0


def test_history_severity_outside_input_range_is_rejected():
    with pytest.raises(ValueError, match="within"):
        RiskEvent("b", probability=0.5, low=0.0, high=1.0, severities=(0.5, 3.0))


def test_price_shock_history_counts_rises_at_or_above_threshold(tmp_path):
    from nd_geometry.history import price_shock_history

    csv_path = tmp_path / "p.csv"
    csv_path.write_text("observation_date,P\n2020-01-01,100\n2020-01-08,100\n2020-01-15,100\n2020-01-22,100\n2020-01-29,110\n2020-02-05,110\n")
    history = price_shock_history(csv_path, "P", since="2020-01-01", lag=4, threshold=0.05)

    assert history.observations == 2
    assert history.probability == pytest.approx(1.0)
    assert history.severities == (0.1, 0.1)


def test_fuel_shock_history_matches_recorded_estimate():
    from nd_geometry.history import price_shock_history

    history = price_shock_history(
        pathlib.Path(__file__).resolve().parents[1] / "data" / "us_diesel_retail_weekly_fred_GASDESW.csv",
        "GASDESW",
        since="2015-01-01",
    )
    assert history.probability == pytest.approx(0.1333, abs=0.0005)
    assert len(history.severities) == 82
