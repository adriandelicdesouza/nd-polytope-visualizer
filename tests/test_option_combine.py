import math

import numpy as np

from nd_geometry.option_combine import (CALL, PUT, combine, implied_vol_call, option_density,
                                        otm_call_quotes)
from nd_geometry.density import belief_density  # noqa: F401  (import check)
from nd_geometry.option_combine import _bsm_call


def lognormal_chain(spot=100.0, vol=0.30, rate=0.04, t=0.5, q=0.0):
    strikes = np.arange(60.0, 151.0, 2.5)
    quotes = []
    for k in strikes:
        if k >= spot:
            quotes.append((k, CALL, _bsm_call(spot, k, t, rate, vol, q)))
        else:
            call = _bsm_call(spot, k, t, rate, vol, q)
            put = call - spot * math.exp(-q * t) + k * math.exp(-rate * t)
            quotes.append((k, PUT, put))
    return quotes


def test_implied_vol_round_trips():
    price = _bsm_call(100, 105, 0.5, 0.04, 0.27, 0.0)
    assert abs(implied_vol_call(price, 100, 105, 0.5, 0.04) - 0.27) < 1e-6
    assert implied_vol_call(-1.0, 100, 105, 0.5, 0.04) is None


def test_constant_smile_recovers_lognormal_density():
    spot, vol, rate, t = 100.0, 0.30, 0.04, 0.5
    k, c = otm_call_quotes(lognormal_chain(spot, vol, rate, t), spot, rate, t)
    opt = option_density(k, c, spot, rate, t)
    assert abs(opt.density.sum() * (opt.grid[1] - opt.grid[0]) - 1) < 1e-6
    # risk-neutral mean of S_T is the forward
    assert abs(opt.mean() - spot * math.exp(rate * t)) / opt.forward < 0.005
    # lognormal median
    median = spot * math.exp((rate - 0.5 * vol * vol) * t)
    assert abs(opt.quantiles((0.5,))[0.5] - median) / median < 0.01
    assert opt.fit_rmse_vol < 1e-3


def test_combined_density_centres_where_dcf_and_market_agree():
    spot, vol, rate, t = 100.0, 0.30, 0.04, 0.5
    k, c = otm_call_quotes(lognormal_chain(spot, vol, rate, t), spot, rate, t)
    opt = option_density(k, c, spot, rate, t)
    rng = np.random.default_rng(0)
    draws = rng.normal(spot, 8.0, 4000)  # DCF belief centred on the market price
    out = combine(draws, opt, market_price=spot)
    assert out.product is not None and out.overlap > 1e-3
    s = out.summary(out.product)
    assert abs(s["p50"] - spot) < 4.0
    assert 0 < s["p_above_market"] < 1


def test_no_overlap_is_reported_not_hidden():
    spot, vol, rate, t = 100.0, 0.30, 0.04, 0.5
    k, c = otm_call_quotes(lognormal_chain(spot, vol, rate, t), spot, rate, t)
    opt = option_density(k, c, spot, rate, t)
    far = np.random.default_rng(1).normal(900.0, 5.0, 2000)  # DCF value far above anything the market prices
    out = combine(far, opt, market_price=spot)
    assert out.product is None and out.overlap < 1e-6
    assert out.summary(out.mixture)["p50"] > 0

def test_bad_quotes_are_filtered_and_do_not_break_the_fit():
    spot, vol, rate, t = 100.0, 0.30, 0.04, 0.5
    good = lognormal_chain(spot, vol, rate, t)
    junk = [(2000.0, CALL, 30.0)]  # a far strike with an impossible price: outside the band and the IV range
    k, c = otm_call_quotes(good + junk, spot, rate, t)
    opt = option_density(k, c, spot, rate, t)
    assert opt.fit_rmse_vol < 1e-3
