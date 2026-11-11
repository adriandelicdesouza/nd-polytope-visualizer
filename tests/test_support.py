import time

import numpy as np

from company_fixtures import make_raw
from nd_geometry.density import Belief, belief_density
from nd_geometry.dcf import company_model
from nd_geometry.filings import company_from_facts
from nd_geometry.model import Input, Model, Output
from nd_geometry.output_polytope import output_polytope
from nd_geometry.support import support_polytope


def linear_model(n: int, seed: int = 0) -> tuple[Model, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(2, n))
    inputs = tuple(Input(f"x{i}", "u", -1.0, 2.0, 0.5) for i in range(n))
    names = [i.name for i in inputs]
    model = Model(
        "lin", inputs, (Output("a", "u"), Output("b", "u")),
        lambda v: {"a": float(w[0] @ [v[k] for k in names]), "b": float(w[1] @ [v[k] for k in names])},
    )
    return model, w


def test_linear_40_inputs_matches_exact_box_support():
    model, w = linear_model(40)
    res = support_polytope(model, directions=120, seed=1)
    poly = res.polytope()
    # exact support of the image of a box under a linear map: sum of max(w*low, w*high) per direction
    for d in (np.array([1.0, 0.0]), np.array([0.0, -1.0]), np.array([1.0, 1.0])):
        exact = np.sum(np.maximum(d @ w * -1.0, d @ w * 2.0))
        found = max(res.outputs @ d)
        assert found <= exact + 1e-9 and found >= exact - 1e-9
    assert poly.vertices.shape[1] == 2 and len(poly.vertices) >= 6


def test_matches_corner_hull_on_a_ten_input_dcf():
    model, _ = company_model(company_from_facts(make_raw(), ticker="T"))
    n = len(model.inputs)
    assert n >= 9
    res = support_polytope(model, directions=400, seed=2)
    approx = res.polytope()
    exact = belief_density(model, {}, draws=200, seed=0, corner_limit=n + 1).feasible
    ratio = approx.volume / exact.volume
    assert 0.95 <= ratio <= 1.0 + 1e-6


def test_non_monotone_model_stays_inside_true_region():
    inputs = tuple(Input(f"x{i}", "u", -1.0, 1.0, 0.0) for i in range(6))
    model = Model(
        "bowl", inputs, (Output("a", "u"), Output("b", "u")),
        lambda v: {"a": sum(v[f"x{i}"] ** 2 for i in range(6)), "b": v["x0"] - v["x1"]},
    )
    res = support_polytope(model, directions=100, seed=3)
    assert res.outputs[:, 0].max() > 5.5  # the bowl's rim, about 6
    assert res.outputs[:, 0].min() >= 0.0


def test_sixty_inputs_scale_without_corners():
    model, _ = linear_model(60)
    t = time.time()
    res = support_polytope(model, directions=100, seed=0)
    assert time.time() - t < 60 and res.evaluations < 100_000
    assert len(res.polytope().vertices) >= 4


def test_belief_density_switches_to_support_above_corner_limit():
    model, _ = linear_model(30)
    out = belief_density(model, {}, draws=500, seed=0, corner_limit=12, directions=60)
    pts = out.feasible.vertices
    assert pts.shape[1] == 2 and len(pts) >= 4


def test_vertex_inputs_and_roles_for_one_sided_inputs():
    inputs = (Input("up", "u", 0.0, 1.0, 0.5), Input("down", "u", 0.0, 1.0, 0.5))
    model = Model("m", inputs, (Output("a", "u"), Output("b", "u")),
                  lambda v: {"a": v["up"] + v["down"], "b": v["up"] - v["down"]})
    res = support_polytope(model, directions=40, seed=0)
    poly = res.polytope()
    assert len(res.vertex_inputs(poly)) == len(poly.vertices)
    roles = res.input_roles(poly)
    assert set(roles.values()) <= {"low", "high", "trade-off", "interior"}
    assert roles["up"] == "trade-off"

def test_wacc_build_and_per_year_inputs_give_a_big_dcf_that_runs():
    co = company_from_facts(make_raw(), ticker="T")
    small, _ = company_model(co)
    big, _ = company_model(co, years=5, wacc_build=True, per_year=True)
    assert len(big.inputs) > 3 * len(small.inputs) and "wacc" not in big.input_names
    assert {"rf", "equity_beta", "erp", "ebit_margin_y3"} <= set(big.input_names)
    out = big.evaluate(big.base_case())
    ref = small.evaluate(small.base_case())
    assert out["implied_price"] > 0 and np.isfinite(out["tv_share"])
    res = support_polytope(big, directions=80, seed=0)
    assert len(res.polytope().vertices) >= 4 and res.evaluations < 200_000
    # per-year inputs set to the single-input value reproduce the single-input model
    flat, _ = company_model(co, years=5, per_year=True)
    assert abs(flat.evaluate(flat.base_case())["implied_price"] - ref["implied_price"]) < 1e-9


def test_hull_report_gives_full_shape_and_extremes():
    from nd_geometry.hull_report import hull_report
    inputs = tuple(Input(f"x{i}", "u", 0.0, 1.0, 0.5) for i in range(3))
    def f(v):
        if v["x0"] > 0.99 and v["x1"] > 0.99:
            raise ValueError("refused")
        return {"a": v["x0"] + v["x1"], "b": v["x0"] - v["x2"]}
    model = Model("m", inputs, (Output("a", "u"), Output("b", "u")), f)
    res = support_polytope(model, directions=60, seed=0)
    rep = hull_report(res.polytope(), res)
    text = "\n".join(rep.lines())
    assert rep.vertices >= 3 and rep.facets >= 3 and rep.volume > 0 and rep.infeasible > 0
    assert set(rep.extremes) == {"a", "b"} and "inputs (" in text and "polytope" in text


def test_cheapest_story_spreads_the_move_across_inputs_and_flags_unreachable():
    from nd_geometry.story import cheapest_story
    model, w = linear_model(40)
    base_a = model.evaluate(model.base_case())["a"]
    st = cheapest_story(model, "a", base_a + 3.0)
    assert st.reachable and abs(st.achieved - (base_a + 3.0)) < 1e-3
    # least-squares move is proportional to the weights, so every input moves a little rather than one a lot
    assert max(abs(m) for m in st.moves.values()) < 1.0
    assert abs(st.cost - 9.0 / np.sum((1.5 * w[0]) ** 2)) < 1e-4  # closed-form minimum-norm move
    far = cheapest_story(model, "a", base_a + 1e6)
    assert not far.reachable and "outside" in far.lines()[0]


def test_input_overrides_reach_wacc_parts_and_per_year_inputs():
    import pytest

    co = company_from_facts(make_raw(), ticker="T")
    model, _ = company_model(co, years=5, wacc_build=True, per_year=True,
                             input_overrides={"rf": (0.02, 0.03, 0.04), "ebit_margin_y2": (0.1, 0.2, 0.3)})
    by = {i.name: i for i in model.inputs}
    assert (by["rf"].low, by["rf"].base, by["rf"].high) == (0.02, 0.03, 0.04)
    assert by["ebit_margin_y2"].high == 0.3 and by["ebit_margin_y3"].base != 0.2
    with pytest.raises(ValueError, match="unknown inputs"):
        company_model(co, input_overrides={"nope": (0, 1, 2)})
    with pytest.raises(ValueError, match="low <= base <= high"):
        company_model(co, input_overrides={"wacc": (0.1, 0.05, 0.12)})
