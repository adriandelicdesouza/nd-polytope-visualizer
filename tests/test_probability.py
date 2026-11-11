from __future__ import annotations

import numpy as np
import pytest

from nd_geometry.model import Input, Model, Output
from nd_geometry.probability import RiskEvent, monte_carlo, quantile_profile
from nd_geometry.probability import RiskEvent as Event

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from freight_oil_model import EVENTS, MODEL  # noqa: E402


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


def test_freight_oil_model_runs_with_risk_events():
    result = monte_carlo(MODEL, EVENTS, draws=2000, seed=1)
    stats = result.quantiles("cost_per_tonne")

    assert result.outputs.shape == (2000, 2)
    assert stats[0.5] > 0
    assert result.event_frequencies()["port_delay_days"] == pytest.approx(0.25, abs=0.03)


def test_delays_raise_transit_time_in_the_profile():
    profile = quantile_profile(MODEL, EVENTS, "port_delay_days", "transit_days", points=3, draws=800, seed=2)

    assert profile["p50"][-1] > profile["p50"][0]
    assert len(profile["value"]) == 3
