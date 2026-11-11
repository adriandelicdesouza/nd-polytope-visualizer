from __future__ import annotations

import pytest

from nd_geometry.model import Input, Model, Output


def _model(function=None):
    return Model(
        name="test",
        inputs=(Input("a", "u", 0.0, 1.0, 0.5), Input("b", "u", 0.0, 2.0, 1.0)),
        outputs=(Output("x", "u"), Output("y", "u")),
        function=function or (lambda v: {"x": v["a"] + v["b"], "y": v["a"] * v["b"]}),
    )


def test_input_rejects_bad_range_and_base():
    with pytest.raises(ValueError, match="low must be less"):
        Input("a", "u", 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="base must lie"):
        Input("a", "u", 0.0, 1.0, 2.0)


def test_names_must_be_identifiers():
    with pytest.raises(ValueError, match="valid identifier"):
        Input("wacc rate", "%", 0.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="valid identifier"):
        Output("ev-m", "$m")


def test_model_needs_two_outputs_and_unique_names():
    with pytest.raises(ValueError, match="at least two outputs"):
        Model("m", (Input("a", "u", 0, 1, 0.5),), (Output("x", "u"),), lambda v: {"x": 1.0})
    with pytest.raises(ValueError, match="unique"):
        Model(
            "m",
            (Input("a", "u", 0, 1, 0.5), Input("a", "u", 0, 1, 0.5)),
            (Output("x", "u"), Output("y", "u")),
            lambda v: {"x": 1.0, "y": 1.0},
        )


def test_evaluate_returns_named_outputs():
    assert _model().evaluate({"a": 0.5, "b": 2.0}) == {"x": 2.5, "y": 1.0}


def test_evaluate_rejects_missing_and_unknown_inputs():
    model = _model()
    with pytest.raises(ValueError, match="missing inputs"):
        model.evaluate({"a": 0.5})
    with pytest.raises(ValueError, match="unknown inputs"):
        model.evaluate({"a": 0.5, "b": 1.0, "c": 3.0})


def test_evaluate_rejects_wrong_output_names():
    model = _model(lambda v: {"x": 1.0, "z": 2.0})
    with pytest.raises(ValueError, match="exactly the outputs"):
        model.evaluate({"a": 0.5, "b": 1.0})


def test_evaluate_rejects_non_finite_output():
    model = _model(lambda v: {"x": float("nan"), "y": 1.0})
    with pytest.raises(ValueError, match="not finite"):
        model.evaluate({"a": 0.5, "b": 1.0})


def test_base_case_uses_declared_base_values():
    assert _model().base_case() == {"a": 0.5, "b": 1.0}
