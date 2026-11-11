"""Where the share price could be, and how likely: belief-weighted DCF polytope.
Writes outputs/dcf_belief_density.html."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull

from nd_geometry.density import Belief, belief_density
from dcf_v2_model import MODEL

OUT = Path(__file__).resolve().parent.parent / "outputs"
AMBER = [[0.0, "#0f1115"], [0.5, "#a07200"], [1.0, "#f0a500"]]
PRICE, TV, EXIT = "implied_price", "tv_share", "exit_multiple"
DRAWS = 40_000

# Three sets of beliefs about the same feasible region.
SCENARIOS = {
    "Agnostic (uniform)": {n: Belief("uniform") for n in MODEL.input_names},
    "Base-centred (triangular)": {},
    "Conviction (tight normal)": {n: Belief("normal", 0.12) for n in MODEL.input_names},
}
COLOURS = ["#5c6370", "#f0a500", "#3498db"]
results = {name: belief_density(MODEL, b, draws=DRAWS, seed=3) for name, b in SCENARIOS.items()}
main = results["Base-centred (triangular)"]
low, high = main.support(PRICE)
base_price = MODEL.evaluate(MODEL.base_case())[PRICE]

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
    subplot_titles=(
        f"Share price density inside its feasible interval (${low:.2f} to ${high:.2f})",
        "Polytope projection: price vs exit multiple, probability mass per cell",
        "Polytope projection: price vs terminal value share, probability mass per cell",
        "Price fan by WACC (p10 / p50 / p90 within each WACC slice)",
        "Price fan by revenue growth (p10 / p50 / p90 within each slice)",
    ),
    vertical_spacing=0.09,
)

for (name, res), colour in zip(results.items(), COLOURS):
    grid, dens = res.density(PRICE)
    q = res.quantiles(PRICE)
    fig.add_trace(go.Scatter(x=grid, y=dens, mode="lines", name=name, line=dict(color=colour, width=2),
                             fill="tozeroy", fillcolor="rgba(0,0,0,0)",
                             hovertemplate="$%{x:.2f}<br>density %{y:.4f}<extra>" + name + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[q[0.05], q[0.5], q[0.95]], y=[0, 0, 0], mode="markers", showlegend=False,
                             marker=dict(color=colour, size=9, symbol=["triangle-up", "diamond", "triangle-up"]),
                             hovertemplate="$%{x:.2f}<extra>" + name + " p5/p50/p95</extra>"), row=1, col=1)
fig.add_vline(x=base_price, line=dict(color="#2ecc71", dash="dot"), row=1, col=1)


def hull_outline(res, a, b):
    pts = res.feasible.points[:, [res.model.output_names.index(a), res.model.output_names.index(b)]]
    h = ConvexHull(pts)
    ring = np.append(h.vertices, h.vertices[0])
    return pts[ring, 0], pts[ring, 1]


for col, other in ((1, EXIT), (2, TV)):
    xs, ys, mass = main.joint_density(PRICE, other)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=mass, colorscale=AMBER, showscale=False,
                             hovertemplate="price $%{x:.2f}<br>" + other + " %{y:.3f}<br>mass %{z:.4f}<extra></extra>"),
                  row=2, col=col)
    hx, hy = hull_outline(main, PRICE, other)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", line=dict(color="#e74c3c", width=1.5),
                             showlegend=False, hoverinfo="skip"), row=2, col=col)

for col, driver in ((1, "wacc"), (2, "revenue_growth")):
    rows = main.conditional_price(driver, PRICE)
    x = [r[0] for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[r[3] for r in rows], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=3, col=col)
    fig.add_trace(go.Scatter(x=x, y=[r[1] for r in rows], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(240,165,0,0.25)", showlegend=False, hoverinfo="skip"), row=3, col=col)
    fig.add_trace(go.Scatter(x=x, y=[r[2] for r in rows], mode="lines+markers", line=dict(color="#f0a500"),
                             showlegend=False, hovertemplate=driver + " %{x:.4f}<br>median $%{y:.2f}<extra></extra>"),
                  row=3, col=col)

fig.update_xaxes(title_text="implied price $/share", row=1, col=1)
fig.update_yaxes(title_text="density", row=1, col=1)
fig.update_xaxes(title_text="implied price", row=2)
fig.update_yaxes(title_text="exit multiple (x EBITDA)", row=2, col=1)
fig.update_yaxes(title_text="terminal value share", row=2, col=2)
fig.update_xaxes(title_text="WACC", row=3, col=1)
fig.update_xaxes(title_text="revenue growth", row=3, col=2)
fig.update_yaxes(title_text="implied price", row=3)
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#0a0b0d", plot_bgcolor="#0f1115",
    font=dict(family="IBM Plex Mono, monospace", color="#c8cdd6", size=11),
    title=dict(text="Where the share price could be and how likely (dcf_v2, belief-weighted polytope)",
               font=dict(color="#f0a500")),
    height=1400, margin=dict(l=60, r=20, t=90, b=40), legend=dict(orientation="h", y=1.04),
)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "dcf_belief_density.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    print(f"Feasible price interval: ${low:.2f} to ${high:.2f}; base case ${base_price:.2f}")
    for name, res in results.items():
        q = res.quantiles(PRICE)
        near = res.probability_between(PRICE, 0.8 * base_price, 1.2 * base_price)
        print(f"{name}: p5 ${q[0.05]:.2f}  p50 ${q[0.5]:.2f}  p95 ${q[0.95]:.2f}  P(base +/-20%) {near:.2f}")
    top = main.probability_between(PRICE, 0.5 * high, high)
    print(f"Mass in the top half of the feasible interval (base-centred): {top:.4f}")
    print(f"Written {path}")