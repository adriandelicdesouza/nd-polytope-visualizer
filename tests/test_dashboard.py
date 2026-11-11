from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

from freight_oil_model import EVENTS, MODEL  # noqa: E402

from nd_geometry.dashboard import dashboard_payload, write_dashboard  # noqa: E402
from nd_geometry.probability import monte_carlo  # noqa: E402


def test_payload_has_columns_for_inputs_outputs_and_event_flags():
    result = monte_carlo(MODEL, EVENTS, draws=50, seed=2)
    payload = dashboard_payload(result)

    expected = list(MODEL.input_names) + list(MODEL.output_names) + ["occurred:port_delay_days", "occurred:fuel_shock"]
    assert payload["columns"] == expected
    assert len(payload["data"]) == 50
    assert all(len(row) == len(expected) for row in payload["data"])


def test_html_embeds_data_and_title_without_leftover_placeholders(tmp_path):
    result = monte_carlo(MODEL, EVENTS, draws=20, seed=2)
    path = tmp_path / "dash.html"

    write_dashboard(result, path, title="Test dashboard")
    html = path.read_text()

    assert "<title>Test dashboard</title>" in html
    assert "__DATA__" not in html and "__TITLE__" not in html
    embedded = html.split("const D = ", 1)[1].split(";\nconst cols", 1)[0]
    assert json.loads(embedded)["draws"] == 20


def test_payload_values_are_rounded_to_six_significant_digits():
    result = monte_carlo(MODEL, EVENTS, draws=5, seed=2)
    payload = dashboard_payload(result)

    flat = [value for row in payload["data"] for value in row]
    assert all(float(f"{value:.6g}") == value for value in flat)
