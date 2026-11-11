import json

import pytest

from company_fixtures import make_raw
from nd_geometry.company import analyse_company, build_report, main
from nd_geometry.revenue_build import SegmentBuild

OVERRIDE = {"wacc": (0.09, 0.10, 0.12)}


def test_offline_workflow_runs_end_to_end_and_reports(tmp_path):
    a = analyse_company("STDY", raw=make_raw(), overrides=OVERRIDE, years=5, draws=2000, market_price=30.0)
    assert a.company.ticker == "STDY" and a.price_source == "supplied" and a.independent is not None
    assert a.reverse.market_price == 30.0 and 0 <= a.reverse.prob_at_or_above <= 1
    assert a.backtest is not None and a.backtest.rows
    text = "\n".join(a.lines())
    assert "Reverse DCF" in text and "Backtest" in text and "Market-implied drivers" in text
    page = build_report(a, tmp_path / "r" / "report.html")
    assert page.exists() and page.stat().st_size > 10_000 and "market $30.00" in page.read_text()


def test_correlation_modes_and_bad_mode():
    none = analyse_company("S", raw=make_raw(), overrides=OVERRIDE, draws=500, market_price=30, backtest=False, correlation="none")
    assert none.correlation is None and none.independent is None
    hist = analyse_company("S", raw=make_raw(growths={2018: .03, 2019: .12, 2020: .05, 2021: .11, 2022: .02}), overrides=OVERRIDE,
                           draws=500, market_price=30, backtest=False, correlation="historical")
    assert hist.correlation is not None
    with pytest.raises(ValueError):
        analyse_company("S", raw=make_raw(), overrides=OVERRIDE, draws=100, market_price=30, correlation="magic")


def test_command_line_with_a_segment_build(tmp_path, capsys):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(make_raw()))
    spec = tmp_path / "build.json"
    spec.write_text(json.dumps({"name": "two-stream", "streams": [
        {"name": "core", "base_revenue": 1700.0, "factors": [
            {"name": "volume", "mode": "add", "path": [0.06, 0.05, 0.04], "range": {"low": -0.02, "base": 0.0, "high": 0.02}}]},
        {"name": "new", "base_revenue": 100.0, "factors": [
            {"name": "growth", "mode": "scale", "path": [0.5, 0.3, 0.2], "range": {"low": 0.5, "base": 1.0, "high": 1.5}}]}]}))
    out = tmp_path / "page.html"
    code = main(["STDY", "--facts-file", str(facts), "--build", str(spec), "--set", "wacc=0.09,0.10,0.12", "--draws", "1000",
                 "--market-price", "25", "--no-backtest", "--out", str(out)])
    shown = capsys.readouterr().out
    assert code == 0 and out.exists() and "core_volume" in shown and "new_growth" in shown
    with pytest.raises(SystemExit):
        main(["STDY", "--facts-file", str(facts), "--set", "wacc=oops", "--market-price", "25"])