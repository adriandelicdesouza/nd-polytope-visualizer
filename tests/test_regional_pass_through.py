import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "examples" / "regional_pass_through.py"
_spec = importlib.util.spec_from_file_location("regional_pass_through", _path)
rpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpt)

BASE = {"brent_price": 80.0, "diesel_crack": 30.0, "usd_per_eur": 1.12, "freight_per_litre": 0.02}


def test_base_case_matches_hand_calculation():
    out = rpt.regional_pass_through(BASE)
    assert out["eu_pump_ex_tax"] == pytest.approx(0.172 + (110 / 158.987 + 0.02) / 1.12)
    assert out["us_pump_ex_tax"] == pytest.approx(1.197 + 110 / 42 + 0.02 * 3.785411784)


def test_crude_long_run_pass_through_is_one_in_each_region():
    base = rpt.regional_pass_through(BASE)
    up = rpt.regional_pass_through({**BASE, "brent_price": 90.0})
    assert up["eu_pump_ex_tax"] - base["eu_pump_ex_tax"] == pytest.approx(10 / (158.987 * 1.12))
    assert up["us_pump_ex_tax"] - base["us_pump_ex_tax"] == pytest.approx(10 / 42)


def test_stronger_euro_lowers_eu_price_only():
    base = rpt.regional_pass_through(BASE)
    strong = rpt.regional_pass_through({**BASE, "usd_per_eur": 1.25})
    assert strong["eu_pump_ex_tax"] < base["eu_pump_ex_tax"]
    assert strong["us_pump_ex_tax"] == pytest.approx(base["us_pump_ex_tax"])
