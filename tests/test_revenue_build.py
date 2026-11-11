import json
from pathlib import Path

import pytest

from company_fixtures import make_raw
from nd_geometry.dcf import company_model
from nd_geometry.filings import company_from_facts
from nd_geometry.revenue_build import Factor, SegmentBuild, Stream

SPEC = Path(__file__).resolve().parents[1] / "examples" / "specs" / "nflx_revenue_build.json"


def tiny() -> SegmentBuild:
    subs = Stream("subs", 100.0, (Factor("volume", (0.10, 0.05), "add", -0.02, 0.0, 0.02),
                                   Factor("price", (0.03, 0.03), "add", -0.01, 0.0, 0.01)))
    ads = Stream("ads", 10.0, (Factor("growth", (0.50, 0.20), "scale", 0.5, 1.0, 1.5),))
    return SegmentBuild((subs, ads), "tiny")


def test_growth_is_the_product_of_factors_and_streams_sum():
    b = tiny()
    v = {"subs_volume": 0.0, "subs_price": 0.0, "ads_growth": 1.0}
    paths = b.stream_revenues(v)
    assert paths["subs"][1] == pytest.approx(100 * 1.10 * 1.03)
    assert paths["subs"][2] == pytest.approx(100 * 1.10 * 1.03 * 1.05 * 1.03)
    assert paths["ads"] == pytest.approx([10.0, 15.0, 18.0])
    assert b.revenues(v) == pytest.approx([110.0, paths["subs"][1] + 15.0, paths["subs"][2] + 18.0])


def test_shifts_and_scales_move_every_year():
    b = tiny()
    base = b.revenues({"subs_volume": 0.0, "subs_price": 0.0, "ads_growth": 1.0})
    more = b.revenues({"subs_volume": 0.02, "subs_price": 0.0, "ads_growth": 1.5})
    assert all(m > x for m, x in zip(more[1:], base[1:])) and more[0] == base[0]


def test_inputs_are_named_stream_factor_with_ranges():
    inputs = {i.name: i for i in tiny().inputs()}
    assert set(inputs) == {"subs_volume", "subs_price", "ads_growth"}
    assert (inputs["ads_growth"].low, inputs["ads_growth"].base, inputs["ads_growth"].high) == (0.5, 1.0, 1.5)


def test_bad_builds_are_refused():
    with pytest.raises(ValueError):
        SegmentBuild((Stream("a", 1.0, (Factor("x", (0.1,), "add", 0, 0, 0.1), Factor("y", (0.1, 0.1), "add", 0, 0, 0.1))),))
    with pytest.raises(ValueError):
        Factor("x", (0.1,), "multiply", 0, 0, 1)
    with pytest.raises(ValueError):  # growth of -100% or worse is not a revenue path
        tiny().revenues({"subs_volume": 0.0, "subs_price": -2.0, "ads_growth": 1.0})


def test_netflix_build_reproduces_the_workbook_scenarios():
    spec = json.loads(SPEC.read_text())
    build = SegmentBuild.from_dict(spec)
    names = {i.name: i for i in build.inputs()}
    base = {n: i.base for n, i in names.items()}
    bull = {"subscription_volume": names["subscription_volume"].high, "subscription_pricing": names["subscription_pricing"].high,
            "subscription_fx": 0.0, "advertising_growth": names["advertising_growth"].high}
    bear = {"subscription_volume": names["subscription_volume"].low, "subscription_pricing": names["subscription_pricing"].low,
            "subscription_fx": 1.0, "advertising_growth": names["advertising_growth"].low}
    assert build.base_revenue == pytest.approx(51125.871)
    assert build.revenues(base)[1:] == pytest.approx(spec["workbook_total"], abs=0.01)
    assert build.revenues(bull)[1:] == pytest.approx(spec["workbook_scenarios"]["bull"], abs=0.01)
    assert build.revenues(bear)[1:] == pytest.approx(spec["workbook_scenarios"]["bear"], abs=0.01)


def test_company_model_takes_the_build_drivers_first():
    build = tiny()
    co = company_from_facts(make_raw(), ticker="STDY", overrides={"wacc": (0.09, 0.10, 0.12)})
    model, used = company_model(co, build)
    assert used is build and model.input_names[:3] == ("subs_volume", "subs_price", "ads_growth")
    assert model.evaluate(model.base_case())["implied_price"] != 0
    assert model.evaluate({**model.base_case(), "ads_growth": 1.5})["implied_price"] > \
        model.evaluate({**model.base_case(), "ads_growth": 0.5})["implied_price"]