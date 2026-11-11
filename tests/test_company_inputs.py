import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

from company_fixtures import LAST_YEAR, fiscal_end, make_raw
from nd_geometry.dcf import company_model, dcf_value
from nd_geometry.filings import cached_get, company_from_facts, make_range
from nd_geometry.prices import parse_chart
from nd_geometry.revenue_build import TopDownBuild

_path = Path(__file__).resolve().parents[1] / "examples" / "dcf_v2_model.py"
_spec = importlib.util.spec_from_file_location("dcf_v2_model", _path)
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def test_ratios_recovered_from_filings():
    co = company_from_facts(make_raw(), ticker="STDY")
    assert co.fiscal_year_end == fiscal_end(LAST_YEAR)
    assert co.base_revenue == pytest.approx(1000 * 1.08 ** 8)
    for name, truth in (("revenue_growth", 0.08), ("ebit_margin", 0.20), ("tax_rate", 0.25),
                        ("capex_pct", 0.05), ("da_pct", 0.045)):
        spec = co.drivers[name]
        assert spec.source == "filings"
        assert spec.base == pytest.approx(truth, abs=1e-9)
        assert spec.low < spec.base < spec.high  # minimum width is enforced on a flat history


def test_cash_conversion_gap_matches_construction():
    co = company_from_facts(make_raw(cfo_gap=0.02))
    assert co.drivers["other_reinvest_pct"].base == pytest.approx(0.02, abs=1e-9)


def test_net_debt_and_shares():
    co = company_from_facts(make_raw())
    assert co.net_debt == pytest.approx(900 + 100 - 300 - 100)  # debt less cash and short-term investments, $m
    assert co.shares == pytest.approx(98.0)  # cover page
    assert any("cover page" in n for n in co.notes)


def test_as_of_hides_later_filings():
    full = company_from_facts(make_raw())
    early = company_from_facts(make_raw(), as_of=date(2021, 6, 1))  # FY2020 10-K filed Feb 2021
    assert early.fiscal_year_end == fiscal_end(2020)
    assert early.base_revenue < full.base_revenue
    assert early.net_debt == 0.0  # balance sheet facts only exist from the FY2024 filing


def test_overrides_are_labelled_and_validated():
    co = company_from_facts(make_raw(), overrides={"wacc": (0.09, 0.115, 0.14)})
    assert co.drivers["wacc"].source == "override" and co.drivers["wacc"].base == 0.115
    with pytest.raises(KeyError):
        company_from_facts(make_raw(), overrides={"nonsense": (0, 0, 1)})
    with pytest.raises(ValueError):
        company_from_facts(make_raw(), overrides={"wacc": (0.02, 0.03, 0.04)})


def test_too_little_history_is_refused():
    raw = make_raw()
    raw["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = raw["facts"]["us-gaap"]["Revenues"]["units"]["USD"][:2]
    with pytest.raises(ValueError):
        company_from_facts(raw)


def test_make_range_contains_base_and_respects_clip():
    low, base, high = make_range([0.01, 0.02, 0.03], clip=(0.015, 0.025))
    assert 0.015 <= low <= base <= high <= 0.025


def test_dcf_matches_v2_on_same_inputs():
    drivers = {"revenue_growth": 0.06, "growth_fade_to": 0.03, "ebit_margin": 0.20, "tax_rate": 0.25,
               "capex_pct": 0.05, "da_pct": 0.045, "nwc_pct": 0.10, "wacc": 0.10, "terminal_growth": 0.02}
    build = TopDownBuild(1000.0, 5, (0.02, 0.06, 0.10), (0.0, 0.03, 0.05))
    mine = dcf_value(drivers, build.revenues(drivers), shares=100.0, net_debt=0.0)
    ref = v2.dcf_v2(drivers)
    for key in ("implied_price", "enterprise_value", "tv_share", "exit_multiple"):
        assert mine[key] == pytest.approx(ref[key])


def test_company_model_evaluates_and_orders_on_wacc():
    co = company_from_facts(make_raw(), ticker="STDY", overrides={"wacc": (0.09, 0.10, 0.12)})
    model, build = company_model(co)
    assert model.input_names[:2] == ("revenue_growth", "growth_fade_to")
    base = model.base_case()
    assert model.evaluate(base)["implied_price"] > 0
    cheap = model.evaluate({**base, "wacc": 0.09})["implied_price"]
    dear = model.evaluate({**base, "wacc": 0.12})["implied_price"]
    assert cheap > model.evaluate(base)["implied_price"] > dear


def test_cached_get_reuses_the_cache(tmp_path):
    calls = []
    fetch = lambda url: calls.append(url) or b"payload"
    assert cached_get("http://x/y", fetch, tmp_path) == b"payload"
    assert cached_get("http://x/y", fetch, tmp_path) == b"payload"
    assert len(calls) == 1


def test_parse_chart_gives_last_price_splits_and_lookup():
    payload = json.dumps({"chart": {"result": [{
        "meta": {"symbol": "ABC", "currency": "USD", "regularMarketPrice": 12.5, "regularMarketTime": 1_700_000_000},
        "timestamp": [1_600_000_000, 1_650_000_000, 1_700_000_000],
        "indicators": {"quote": [{"close": [10.0, None, 12.5]}]},
        "events": {"splits": {"1": {"date": 1_620_000_000, "numerator": 10, "denominator": 1}}}}]}})
    hist = parse_chart(payload)
    assert hist.last_price == 12.5 and len(hist.dates) == 2  # the null close is dropped
    assert hist.price_on(date(2021, 1, 1)) == 10.0 and hist.price_on(date(2019, 1, 1)) is None
    assert hist.split_factor_after(date(2020, 12, 1)) == 10.0
    assert hist.split_factor_after(date(2022, 1, 1)) == 1.0

def test_negative_growth_history_still_gives_a_valid_fade_range():
    co = company_from_facts(make_raw(growth=-0.04))
    fade = co.drivers["growth_fade_to"]
    assert fade.low <= fade.base <= fade.high and co.drivers["revenue_growth"].base < 0


def test_missing_share_count_is_flagged_not_fatal():
    raw = make_raw()
    del raw["facts"]["dei"]["EntityCommonStockSharesOutstanding"]
    del raw["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]
    co = company_from_facts(raw, ticker="NOSH")
    assert co.shares != co.shares and any("NO SHARE COUNT" in n for n in co.notes)
    with pytest.raises(ValueError, match="share count"):
        company_model(co)
