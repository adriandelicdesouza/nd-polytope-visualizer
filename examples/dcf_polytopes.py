"""Polytope views of the DCF: driver hypercube -> output polytope, tesseract of drivers,
and probability surfaces from Monte Carlo. Writes outputs/dcf_polytopes.html."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull

from nd_geometry import Geometry, Hypercube, project, rotate
from nd_geometry.output_polytope import output_polytope
from nd_geometry.probability import monte_carlo
from dcf_v2_model import MODEL

OUT = Path(__file__).resolve().parent.parent / "outputs"
AMBER = [[0.0, "#0f1115"], [0.5, "#a07200"], [1.0, "#f0a500"]]
NAMES = MODEL.input_names
OUT_NAMES = MODEL.output_names
RANGE = {i.name: (i.low, i.high) for i in MODEL.inputs}
BASE = {i.name: i.base for i in MODEL.inputs}
TESSERACT = ("wacc", "terminal_growth", "ebit_margin", "revenue_growth")
THRESHOLD = 20.0  # $/share for the exceedance surface
MC_DRAWS, BINS = 60_000, 20


def to_driver(unit: float, name: str) -> float:
    """Map a hypercube coordinate in [-1, 1] onto the driver's low..high range."""
    low, high = RANGE[name]
    return low + (unit + 1) / 2 * (high - low)


# 1. Driver hypercube: all 2^9 corners of the 9-D input box -> 3-D output polytope.
corners = Hypercube(len(NAMES)).vertices
corner_out = np.array([
    [MODEL.evaluate({n: to_driver(u[k], n) for k, n in enumerate(NAMES)})[o] for o in OUT_NAMES]
    for u in corners
])
poly = output_polytope(corner_out, output_names=OUT_NAMES)
base_out = np.array([MODEL.evaluate(BASE)[o] for o in OUT_NAMES])
hull = ConvexHull(corner_out)

# 2. Tesseract: 4 drivers as a 4-D hypercube, other drivers at base, projected to 3-D.
tess = Hypercube(len(TESSERACT)).vertices
tess_edges = np.array([(a, b) for a, b in itertools.combinations(range(len(tess)), 2)
                       if np.sum(tess[a] != tess[b]) == 1])
tess_price = np.array([
    MODEL.evaluate({**BASE, **{n: to_driver(u[k], n) for k, n in enumerate(TESSERACT)}})["implied_price"]
    for u in tess
])
rotated = rotate(tess, axis_a=0, axis_b=3, angle=np.pi / 8)
rotated = rotate(rotated, axis_a=1, axis_b=3, angle=np.pi / 10)
tgeo = project(Geometry(vertices=rotated, edges=tess_edges), target_dimensions=3)

# 3. Monte Carlo over all drivers -> probability and median surfaces over two drivers.
mc = monte_carlo(MODEL, draws=MC_DRAWS, seed=0)
ix, iy = NAMES.index("wacc"), NAMES.index("revenue_growth")
ip = OUT_NAMES.index("implied_price")
xe = np.linspace(*RANGE["wacc"], BINS + 1)
ye = np.linspace(*RANGE["revenue_growth"], BINS + 1)
xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
xb = np.clip(np.digitize(mc.inputs[:, ix], xe) - 1, 0, BINS - 1)
yb = np.clip(np.digitize(mc.inputs[:, iy], ye) - 1, 0, BINS - 1)
prob = np.full((BINS, BINS), np.nan)
median = np.full((BINS, BINS), np.nan)
for r in range(BINS):
    for c in range(BINS):
        sel = mc.outputs[(yb == r) & (xb == c), ip]
        if sel.size >= 10:
            prob[r, c] = np.mean(sel > THRESHOLD)
            median[r, c] = np.median(sel)
def _segments(vertices: np.ndarray, edges: np.ndarray) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [vertices[a, 0], vertices[b, 0], None]
        ys += [vertices[a, 1], vertices[b, 1], None]
        zs += [vertices[a, 2], vertices[b, 2], None]
    return xs, ys, zs


fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "surface"}, {"type": "xy"}]],
    subplot_titles=(
        "Output polytope: hull of all 512 driver-cube corners",
        "Tesseract: 4 drivers as a 4-D cube, projected to 3-D, coloured by price",
        f"P(implied price > ${THRESHOLD:.0f}) over WACC x revenue growth",
        "Median implied price over WACC x revenue growth ($/share)",
    ),
    vertical_spacing=0.08,
)

fig.add_trace(go.Mesh3d(
    x=corner_out[:, 0], y=corner_out[:, 1], z=corner_out[:, 2],
    i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
    color="#f0a500", opacity=0.25, flatshading=True, hoverinfo="skip",
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=corner_out[:, 0], y=corner_out[:, 1], z=corner_out[:, 2], mode="markers",
    marker=dict(size=2, color="#f0a500"), name="corners", hoverinfo="skip",
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=[base_out[0]], y=[base_out[1]], z=[base_out[2]], mode="markers",
    marker=dict(size=7, color="#3498db"), name="base case",
), row=1, col=1)

sx, sy, sz = _segments(tgeo.vertices, tgeo.edges)
fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
                           line=dict(color="#5c6370", width=3), hoverinfo="skip"), row=1, col=2)
fig.add_trace(go.Scatter3d(
    x=tgeo.vertices[:, 0], y=tgeo.vertices[:, 1], z=tgeo.vertices[:, 2], mode="markers",
    marker=dict(size=7, color=tess_price, colorscale=AMBER, showscale=False),
    text=[f"price ${p:.2f}" for p in tess_price], hoverinfo="text", name="tesseract",
), row=1, col=2)

fig.add_trace(go.Surface(x=xc, y=yc, z=prob, colorscale=AMBER, cmin=0, cmax=1,
                         colorbar=dict(title="P", len=0.4), name="prob"), row=2, col=1)
fig.add_trace(go.Heatmap(x=xc, y=yc, z=median, colorscale=AMBER, colorbar=dict(title="$/sh", len=0.4),
                         name="median"), row=2, col=2)

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#0a0b0d", plot_bgcolor="#0f1115",
    font=dict(family="IBM Plex Mono, monospace", color="#c8cdd6", size=11),
    title=dict(text="DCF polytopes and probability surfaces (dcf_v2)", font=dict(color="#f0a500")),
    height=1500, margin=dict(l=10, r=10, t=90, b=10), showlegend=False,
)
fig.layout.scene.xaxis.title.text = "implied price"
fig.layout.scene.yaxis.title.text = "TV share"
fig.layout.scene.zaxis.title.text = "exit multiple"
fig.layout.scene2.xaxis.title.text = "x"
fig.layout.scene2.yaxis.title.text = "y"
fig.layout.scene2.zaxis.title.text = "z"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "dcf_polytopes.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    print(f"Input hypercube: {len(corners)} corners of {len(NAMES)}-D driver space")
    print(f"Output polytope: {len(poly.vertices)} vertices, volume {poly.volume:.4f}")
    for name, (lo, hi) in zip(poly.output_names, poly.bounds()):
        print(f"  {name}: {lo:.4f} to {hi:.4f}")
    print("Base case inside polytope:", poly.contains(base_out))
    print(f"Tesseract: {len(tess)} vertices, {len(tess_edges)} edges")
    print(f"Monte Carlo draws: {MC_DRAWS}; P(price > ${THRESHOLD:.0f}) overall = {np.mean(mc.outputs[:, ip] > THRESHOLD):.3f}")
    print(f"Written {path}")