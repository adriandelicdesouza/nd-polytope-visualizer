import numpy as np
import pytest
from scipy.stats import kstest, spearmanr

from company_fixtures import make_raw
from nd_geometry.correlation import (Correlation, default_dcf_correlation, historical_correlation,
                                     nearest_correlation)
from nd_geometry.dcf import company_model
from nd_geometry.density import Belief, belief_density, sample_beliefs
from nd_geometry.filings import company_from_facts
from nd_geometry.model import Input, Model, Output


def two_input_model() -> Model:
    return Model("toy", (Input("a", "", 0.0, 1.0, 0.5), Input("b", "", 0.0, 2.0, 0.5), Input("c", "", 0.0, 1.0, 0.2)),
                 (Output("sum", ""), Output("diff", "")),
                 lambda v: {"sum": v["a"] + v["b"] + v["c"], "diff": v["a"] - v["b"]})


def test_independent_draws_are_unchanged_by_the_new_code_path():
    model = two_input_model()
    assert np.array_equal(sample_beliefs(model, {}, 500, seed=4), sample_beliefs(model, {}, 500, seed=4, correlation=None))


def test_copula_hits_the_target_rank_correlation_and_keeps_marginals():
    model = two_input_model()
    beliefs = {"a": Belief("triangular"), "b": Belief("normal", 0.2), "c": Belief("uniform")}
    corr = Correlation.from_pairs(["a", "b"], {("a", "b"): 0.6})
    x = sample_beliefs(model, beliefs, 40_000, seed=2, correlation=corr)
    assert spearmanr(x[:, 0], x[:, 1])[0] == pytest.approx(6 / np.pi * np.arcsin(0.6 / 2), abs=0.02)
    assert abs(spearmanr(x[:, 0], x[:, 2])[0]) < 0.02  # c is not listed, so independent
    assert kstest(x[:, 2], "uniform").pvalue > 0.01  # uniform marginal preserved
    assert x[:, 0].mean() == pytest.approx((0 + 0.5 + 1) / 3, abs=0.01)  # triangular marginal preserved
    assert x[:, 0].min() >= 0 and x[:, 1].max() <= 2


def test_positive_correlation_widens_a_sum_and_negative_narrows_it():
    model = two_input_model()
    sd = lambda rho: belief_density(model, draws=8000, seed=1,
                                    correlation=Correlation.from_pairs(["a", "b"], {("a", "b"): rho})
                                    ).column("sum").std()
    assert sd(0.7) > sd(0.0) > sd(-0.7)


def test_invalid_matrices_are_refused():
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="positive semi-definite"):
        Correlation.from_pairs(names, {("a", "b"): 0.9, ("b", "c"): 0.9, ("a", "c"): -0.9})
    with pytest.raises(ValueError):
        Correlation.from_pairs(names, {("a", "b"): 1.0})
    with pytest.raises(ValueError):
        Correlation.from_pairs(names, {("a", "z"): 0.1})
    with pytest.raises(ValueError, match="does not have"):
        sample_beliefs(two_input_model(), {}, 10, correlation=Correlation.from_pairs(["a", "zzz"], {("a", "zzz"): 0.2}))


def test_nearest_correlation_repairs_an_invalid_matrix():
    bad = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])
    fixed = nearest_correlation(bad)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-9 and np.allclose(np.diag(fixed), 1)
    Correlation(("a", "b", "c"), fixed)  # now valid


def test_default_dcf_priors_and_historical_shrinkage():
    names = ["rev_a", "rev_b", "ebit_margin", "capex_pct", "da_pct", "wacc", "terminal_growth"]
    prior = default_dcf_correlation(names)
    assert prior.rho("capex_pct", "da_pct") == 0.6 and prior.rho("rev_a", "rev_b") == 0.25 and prior.rho("rev_a", "wacc") == 0
    years = [__import__("datetime").date(2016 + i, 12, 31) for i in range(8)]
    up = [(d, float(i)) for i, d in enumerate(years)]
    down = [(d, float(-i)) for i, d in enumerate(years)]
    series = {"capex_pct": up, "da_pct": down}  # perfectly negative in the data
    blended = historical_correlation(series, names, prior, prior_weight=0.7)
    assert blended.rho("capex_pct", "da_pct") == pytest.approx(0.7 * 0.6 + 0.3 * -1.0, abs=0.05)
    assert historical_correlation(series, names, prior, min_obs=20).rho("capex_pct", "da_pct") == pytest.approx(0.6)


def test_correlation_moves_dcf_price_dispersion_the_way_the_economics_say():
    co = company_from_facts(make_raw(), ticker="STDY", overrides={"wacc": (0.09, 0.10, 0.12)})
    model, _ = company_model(co)
    sd = lambda pairs: belief_density(
        model, draws=6000, seed=3,
        correlation=Correlation.from_pairs(model.input_names, pairs) if pairs else None).column("implied_price").std()
    independent = sd({})
    # a higher discount rate cuts value and higher terminal growth raises it, so moving together damps the price
    assert sd({("wacc", "terminal_growth"): 0.9}) < independent
    # growth and margin both raise value, so moving together widens the price
    assert sd({("revenue_growth", "ebit_margin"): 0.9}) > independent
    prior = default_dcf_correlation(model.input_names)
    assert belief_density(model, draws=500, seed=3, correlation=prior).correlation is prior
