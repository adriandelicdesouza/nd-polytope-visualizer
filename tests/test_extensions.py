from __future__ import annotations

import json

import numpy as np
import pytest

from nd_geometry.analysis import analyse
from nd_geometry.baseline import compare_to_baseline
from nd_geometry.explain import explain, input_influence
from nd_geometry.model import Input, Model, Output
from nd_geometry.refine import refine
from nd_geometry.views import envelope_projection, input_surface, surface_figure


def _linear(scale: float = 1.0):
    # x = scale*a + 2b, y = a: wacc-like input b moves x twice as much as a.
    return Model(
        name="lin",
        inputs=(Input("a", "u", 0.0, 1.0, 0.5), Input("b", "u", 0.0, 1.0, 0.5)),
        outputs=(Output("x", "u"), Output("y", "%")),
        function=lambda v: {"x": scale * v["a"] + 2.0 * v["b"], "y": v["a"]},
    )


def test_random_method_includes_corners_for_exact_bounds():
    result = analyse(_linear(), method="random", samples=50, seed=3)

    assert result.method == "random"
    assert result.output_bounds["x"] == pytest.approx((0.0, 3.0))
    assert result.output_bounds["y"] == pytest.approx((0.0, 1.0))


def test_random_method_is_reproducible_with_seed():
    first = analyse(_linear(), method="random", samples=40, seed=7)
    second = analyse(_linear(), method="random", samples=40, seed=7)

    np.testing.assert_allclose(first.outputs, second.outputs)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be"):
        analyse(_linear(), method="sobol")


def test_refine_never_shrinks_and_reports_history():
    result = analyse(_linear(), points_per_dimension=3)

    refined, volumes = refine(result, rounds=3)

    assert volumes[0] == pytest.approx(result.polytope.volume)
    assert all(later >= earlier - 1e-9 for earlier, later in zip(volumes, volumes[1:]))
    assert refined.polytope.volume >= result.polytope.volume - 1e-9
    assert refined.base_inside is True


def test_refine_finds_interior_extreme_that_grid_misses():
    # The output peaks at the middle of the input range, which a 3-point grid misses.
    peaked = Model(
        name="peak",
        inputs=(Input("a", "u", 0.0, 1.0, 0.5),),
        outputs=(Output("x", "u"), Output("y", "u")),
        function=lambda v: {"x": 1.0 - (v["a"] - 0.5) ** 2 * 4.0 + v["a"], "y": v["a"]},
    )
    coarse = analyse(peaked, points_per_dimension=3)
    refined, _ = refine(coarse, rounds=4)

    assert refined.output_bounds["x"][1] > coarse.output_bounds["x"][1]


def test_refine_rejects_zero_rounds():
    result = analyse(_linear(), points_per_dimension=3)
    with pytest.raises(ValueError, match="rounds"):
        refine(result, rounds=0)


def test_input_influence_ranks_stronger_input_first():
    result = analyse(_linear(), points_per_dimension=3)

    ranking = input_influence(result)["x"]

    assert ranking[0][0] == "b"
    assert ranking[0][1] == pytest.approx(2.0)
    assert sum(share for _, _, share in ranking) == pytest.approx(1.0)


def test_explain_mentions_outputs_top_driver_and_formats_percent():
    lines = explain(analyse(_linear(), points_per_dimension=3))
    text = "\n".join(lines)

    assert "x ranges from 0.00 u to 3.00 u" in text
    assert "b moves x most" in text
    assert "y ranges from 0.0% to 100.0%" in text
    assert "one input at a time" in text


def test_input_surface_varies_two_inputs_and_fixes_the_rest():
    result = analyse(_linear(), points_per_dimension=3)

    surface = input_surface(result, "a", "b", "x", fixed=None, points=5)

    assert surface["z"].shape == (5, 5)
    assert surface["z"][0, 0] == pytest.approx(0.0)  # a=0, b=0
    assert surface["z"][-1, -1] == pytest.approx(3.0)  # a=1, b=1


def test_input_surface_rejects_bad_names():
    result = analyse(_linear(), points_per_dimension=3)
    with pytest.raises(ValueError, match="two different model inputs"):
        input_surface(result, "a", "a", "x")
    with pytest.raises(ValueError, match="unknown input"):
        input_surface(result, "a", "b", "x", fixed={"zzz": 1.0})


def test_figures_build_without_error():
    result = analyse(_linear(), points_per_dimension=3)

    assert surface_figure(input_surface(result, "a", "b", "x", points=4)).data
    assert envelope_projection(result, "x", "y").data


def test_baseline_passes_for_identical_model_and_flags_changed_model(tmp_path):
    path = tmp_path / "baseline.json"
    analyse(_linear(), points_per_dimension=3).save(path)

    same = compare_to_baseline(analyse(_linear(), points_per_dimension=3), path)
    changed = compare_to_baseline(analyse(_linear(scale=1.5), points_per_dimension=3), path)

    assert same == {"ok": True, "changes": []}
    assert changed["ok"] is False
    assert any(change.get("item") == "output x.high" for change in changed["changes"])


def test_baseline_flags_changed_input_bounds(tmp_path):
    path = tmp_path / "baseline.json"
    analyse(_linear(), points_per_dimension=3).save(path)
    wider = Model(
        name="lin",
        inputs=(Input("a", "u", 0.0, 2.0, 0.5), Input("b", "u", 0.0, 1.0, 0.5)),
        outputs=(Output("x", "u"), Output("y", "%")),
        function=lambda v: {"x": v["a"] + 2.0 * v["b"], "y": v["a"]},
    )

    report = compare_to_baseline(analyse(wider, points_per_dimension=3), path)

    assert report["ok"] is False
    assert any(change["item"] == "input a.high" for change in report["changes"])
    assert json.loads(path.read_text())["model"] == "lin"
