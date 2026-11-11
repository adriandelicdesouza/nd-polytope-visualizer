import json
import pathlib

import pytest

from nd_geometry.cli import main
from nd_geometry.spec import load_model

SPECS = pathlib.Path(__file__).resolve().parents[1] / "examples" / "specs"


def test_dcf_spec_loads_and_gives_a_sensible_base_case():
    model = load_model(SPECS / "dcf.json")
    base = model.evaluate(model.base_case())
    assert len(model.input_names) == 9
    assert base["implied_price"] > 0
    assert 0 < base["tv_share"] < 1


def test_regional_spec_matches_the_python_model():
    from regional_pass_through import MODEL as python_model

    model = load_model(SPECS / "regional_pass_through.json")
    assert model.evaluate(model.base_case()) == pytest.approx(python_model.evaluate(python_model.base_case()))


def test_sea_spec_has_all_eleven_inputs():
    model = load_model(SPECS / "sea_freight.json")
    assert len(model.input_names) == 11
    assert model.evaluate(model.base_case())["cost_per_teu"] > 0


def test_spec_rejects_bad_function_reference(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "function": "nomodule", "inputs": [], "outputs": []}))
    with pytest.raises(ValueError):
        load_model(bad)


def test_cli_scenario_prints_base_and_scenario(capsys):
    assert main(["scenario", str(SPECS / "regional_pass_through.json"), "--set", "brent_price=100"]) == 0
    out = capsys.readouterr().out
    assert "eu_pump_ex_tax" in out and "us_pump_ex_tax" in out


def test_cli_scenario_rejects_unknown_input():
    with pytest.raises(SystemExit):
        main(["scenario", str(SPECS / "regional_pass_through.json"), "--set", "oil=1"])


def test_cli_analyse_writes_results(tmp_path, capsys):
    assert main(["analyse", str(SPECS / "regional_pass_through.json"), "--samples", "300", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "explain.txt").exists()
    assert "Regional" in capsys.readouterr().out


def test_cli_surface_returns_grid_of_expected_shape(tmp_path):
    assert main(["surface", str(SPECS / "dcf.json"), "--x", "wacc", "--y", "terminal_growth",
                 "--output", "implied_price", "--points", "5", "--samples", "300", "--out", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "surface_implied_price.json").read_text())
    assert len(payload["z"]) == len(payload["y"])
    assert len(payload["z"][0]) == len(payload["x"])
