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
    assert 0.9 <= ratio <= 1.0 + 1e-6


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