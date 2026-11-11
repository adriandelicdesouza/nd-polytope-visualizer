from __future__ import annotations

import json

import numpy as np
import pytest

from nd_geometry.analysis import analyse
from nd_geometry.model import Input, Model, Output


def _linear_model(base_b: float = 1.0):
    # x = a + b and y = a - b: the image of the input box is a parallelogram.
    return Model(
        name="linear",
        inputs=(Input("a", "u", 0.0, 1.0, 0.5), Input("b", "u", 0.0, 2.0, base_b)),
        outputs=(Output("x", "u"), Output("y", "u")),
        function=lambda v: {"x": v["a"] + v["b"], "y": v["a"] - v["b"]},
    )


def test_output_bounds_match_exact_range_of_linear_model():
    result = analyse(_linear_model(), points_per_dimension=5)

    assert result.output_bounds["x"] == pytest.approx((0.0, 3.0))
    assert result.output_bounds["y"] == pytest.approx((-2.0, 1.0))


def test_base_case_inside_and_position_are_reported():
    result = analyse(_linear_model(base_b=1.0), points_per_dimension=5)

    assert result.base_outputs == {"x": 1.5, "y": -0.5}
    assert result.base_inside is True
    assert result.base_position()["x"] == pytest.approx(0.5)
    assert result.base_position()["y"] == pytest.approx(0.5)


def test_grid_outputs_have_one_row_per_grid_point():
    result = analyse(_linear_model(), points_per_dimension=4)

    assert result.outputs.shape == (16, 2)
    assert result.method == "grid"
    assert result.points_per_dimension == 4


def test_runner_works_on_a_nonlinear_model_without_special_casing():
    model = Model(
        name="curved",
        inputs=(Input("a", "u", 0.0, 1.0, 0.5), Input("b", "u", 0.0, 3.0, 1.0)),
        outputs=(Output("p", "u"), Output("q", "u")),
        function=lambda v: {"p": v["a"] ** 2 + 1.0, "q": np.sin(v["b"]) + v["a"]},
    )

    result = analyse(model, points_per_dimension=6)

    assert result.polytope.dimensions == 2
    assert result.output_bounds["p"][0] == pytest.approx(1.0)
    assert result.base_inside is True


def test_to_dict_and_save_round_trip(tmp_path):
    result = analyse(_linear_model(), points_per_dimension=3)
    path = tmp_path / "summary.json"

    result.save(path)
    loaded = json.loads(path.read_text())

    assert loaded["model"] == "linear"
    assert loaded["sample_count"] == 9
    assert loaded["base_case_inside_envelope"] is True
    assert [o["name"] for o in loaded["outputs"]] == ["x", "y"]
    assert loaded["outputs"][0]["low"] == pytest.approx(0.0)


def test_runner_fails_clearly_when_model_returns_non_finite_output():
    model = Model(
        name="bad",
        inputs=(Input("a", "u", 0.0, 1.0, 0.5),),
        outputs=(Output("x", "u"), Output("y", "u")),
        function=lambda v: {"x": v["a"], "y": float("inf")},
    )

    with pytest.raises(ValueError, match="not finite"):
        analyse(model, points_per_dimension=3)
