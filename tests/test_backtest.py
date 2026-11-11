import json
from datetime import date

import numpy as np
import pytest

from company_fixtures import make_raw
from nd_geometry.backtest import (backtest_basket, backtest_company, calibrate_width, pit, price_check,
                                  price_rank_correlation)
from nd_geometry.filings import SEC_FACTS_URL, SEC_TICKERS_URL
from nd_geometry.prices import PriceHistory

KW = dict(draws=400, seed=1)


def rows_for(result, origin, metric="revenue", horizon=1):
    return [r for r in result.rows if r.origin == origin and r.metric == metric and r.horizon == horizon]


def test_pit_counts_ties_as_half():
    d = np.array([1.0, 2.0, 2.0, 3.0])
    assert pit(d, 0.0) == 0.0 and pit(d, 9.0) == 1.0 and pit(d, 2.0) == pytest.approx(0.5)


def test_forecasts_use_only_filings_available_at_the_origin():
    calm = backtest_company(make_raw(), "X", **KW)
    later_shock = backtest_company(make_raw(growths={2023: -0.30, 2024: -0.30}), "X", **KW)
    a, b = rows_for(calm, date(2021, 12, 31), horizon=2)[0], rows_for(later_shock, date(2021, 12, 31), horizon=2)[0]
    assert (a.p10, a.p50, a.p90) == (b.p10, b.p50, b.p90)  # the forecast cannot see 2023
    assert b.realized < a.realized  # but the realised outcome does


def test_a_stable_company_is_centred_and_a_collapse_is_flagged():
    calm = backtest_company(make_raw(), "X", **KW)
    assert 0.35 < rows_for(calm, date(2021, 12, 31))[0].pit < 0.65
    assert calm.coverage(0.8, "revenue", 1) == 1.0
    broken = backtest_company(make_raw(growths={2022: -0.20, 2023: -0.20, 2024: -0.20}), "X", **KW)
    assert rows_for(broken, date(2021, 12, 31))[0].pit == 0.0
    assert broken.mean_pit("revenue") < 0.1 and broken.coverage(0.8, "revenue") < 0.2


def test_every_row_has_ordered_quantiles_and_a_valid_pit():
    result = backtest_company(make_raw(growths={2018: .04, 2019: .12, 2020: .03, 2021: .11, 2022: .05}), "X", **KW)
    assert len(result.rows) > 10
    for r in result.rows:
        assert r.p10 <= r.p50 <= r.p90 and 0 <= r.pit <= 1 and r.as_of > r.origin
    assert {r.metric for r in result.rows} == {"revenue", "ebit", "cash_flow_proxy"}


def test_too_little_history_gives_no_forecasts():
    raw = make_raw()
    for rows in raw["facts"]["us-gaap"]["Revenues"]["units"].values():
        del rows[:4]
    assert backtest_company(raw, "X", **KW).rows == ()


def test_wider_beliefs_cover_more_and_are_less_sharp():
    noisy = make_raw(growths={2017: .03, 2018: .12, 2019: .05, 2020: .11, 2021: .02, 2022: .13, 2023: .01, 2024: .14})
    narrow, wide = calibrate_width(noisy, widths=(0.5, 3.0), **KW)
    assert wide.coverage_80 >= narrow.coverage_80 and wide.median_relative_width > narrow.median_relative_width


def test_basket_pools_companies_and_reports_failures():
    facts = make_raw()
    table = {"0": {"cik_str": 1, "ticker": "STDY", "title": "Steady"}}

    def fetch(url):
        if url == SEC_TICKERS_URL:
            return json.dumps(table).encode()
        if url == SEC_FACTS_URL.format(cik=1):
            return json.dumps(facts).encode()
        raise OSError(url)

    result = backtest_basket(["STDY", "NOPE"], fetch=fetch, **KW)
    assert {r.ticker for r in result.rows} == {"STDY"} and "NOPE" in result.skipped


def test_price_check_restates_shares_for_later_splits_and_records_forward_return():
    days = [date(2016 + i // 2, 1 + 6 * (i % 2), 15) for i in range(20)]
    prices = PriceHistory("X", "USD", 50.0, days[-1], tuple(days), tuple(10.0 + 2 * i for i in range(20)),
                          ((date(2022, 6, 1), 10.0),))
    rows = price_check(make_raw(), prices, "X", draws=300, seed=2)
    assert rows and all(0 <= r.pit <= 1 and r.p10 <= r.p50 <= r.p90 for r in rows)
    assert any(r.forward_return is not None for r in rows)
    early = next(r for r in rows if r.as_of < date(2022, 6, 1))
    # the split multiplies the share count by 10 for filings before it, so the per-share value is a tenth
    flat = price_check(make_raw(), PriceHistory("X", "USD", 50.0, days[-1], tuple(days), prices.closes, ()), "X",
                       draws=300, seed=2)
    same = next(r for r in flat if r.as_of == early.as_of)
    assert early.p50 == pytest.approx(same.p50 / 10)
    corr, n = price_rank_correlation(rows)
    assert n >= 0