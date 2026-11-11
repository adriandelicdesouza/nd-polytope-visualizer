import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "examples" / "dcf_v2_model.py"
_spec = importlib.util.spec_from_file_location("dcf_v2_model", _path)
dcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcf)

BASE = {"revenue_growth": 0.06, "ebit_margin": 0.20, "tax_rate": 0.25, "capex_pct": 0.05,
        "da_pct": 0.045, "nwc_pct": 0.10, "wacc": 0.10, "terminal_growth": 0.02}


def test_wacc_must_exceed_terminal_growth():
    with pytest.raises(ValueError):
        dcf.dcf_v2({**BASE, "wacc": 0.02, "terminal_growth": 0.02})


def test_mid_year_values_higher_than_end_year(monkeypatch):
    mid = dcf.dcf_v2(BASE)["enterprise_value"]
    monkeypatch.setattr(dcf, "MID_YEAR", False)
    end = dcf.dcf_v2(BASE)["enterprise_value"]
    assert mid > end


def test_no_tax_credit_on_negative_ebit():
    loss = dcf.dcf_v2({**BASE, "ebit_margin": -0.05, "tax_rate": 0.25})
    no_tax = dcf.dcf_v2({**BASE, "ebit_margin": -0.05, "tax_rate": 0.0})
    assert loss["enterprise_value"] == pytest.approx(no_tax["enterprise_value"])


def test_no_growth_means_no_working_capital_drag():
    flat = dcf.dcf_v2({**BASE, "revenue_growth": 0.0, "nwc_pct": 0.10})
    flat_no_nwc = dcf.dcf_v2({**BASE, "revenue_growth": 0.0, "nwc_pct": 0.0})
    assert flat["enterprise_value"] == pytest.approx(flat_no_nwc["enterprise_value"])


def test_tv_share_between_zero_and_one():
    out = dcf.dcf_v2(BASE)
    assert 0 < out["tv_share"] < 1


def test_flat_growth_fade_matches_no_fade():
    flat = dcf.dcf_v2({**BASE, "growth_fade_to": BASE["revenue_growth"]})
    default = dcf.dcf_v2(BASE)
    assert flat["implied_price"] == pytest.approx(default["implied_price"])


def test_lower_fade_growth_lowers_value():
    high = dcf.dcf_v2({**BASE, "growth_fade_to": 0.05})["implied_price"]
    low = dcf.dcf_v2({**BASE, "growth_fade_to": 0.00})["implied_price"]
    assert low < high


def test_exit_multiple_is_positive_for_base_case():
    assert dcf.dcf_v2(BASE)["exit_multiple"] > 0
