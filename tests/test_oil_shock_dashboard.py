import importlib.util
import json
import pathlib
import sys

_examples = pathlib.Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(_examples))
_spec = importlib.util.spec_from_file_location("oil_shock_dashboard", _examples / "oil_shock_dashboard.py")
osd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osd)


def test_payload_base_scenario_is_zero_change():
    base = osd.build_payload()["scen"][0]
    assert base["name"] == "Base"
    assert abs(base["deu"]) < 1e-9 and abs(base["dteu"]) < 1e-9


def test_pump_path_starts_at_zero_and_rises():
    path = osd.build_payload()["path"]
    assert path[0]["eu"] == 0
    assert path[-1]["eu"] > path[1]["eu"] > 0


def test_html_is_self_contained_with_no_placeholders():
    html = osd.build_html()
    assert "__PAYLOAD__" not in html
    embedded = html.split("const D = ", 1)[1].split(";\nconst base", 1)[0]
    assert "scen" in json.loads(embedded)
