"""Six DCF surfaces: each output over two drivers, the rest held at base. Writes outputs/dcf_surfaces.html."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dcf_v2_model import MODEL

OUT = Path(__file__).resolve().parent.parent / "outputs"
N = 25
AMBER = [[0.0, "#0f1115"], [0.5, "#a07200"], [1.0, "#f0a500"]]

PANELS = [
    ("implied_price", "wacc", "terminal_growth", "Implied price $/share: WACC vs terminal growth"),
    ("implied_price", "revenue_growth", "ebit_margin", "Implied price: revenue growth vs EBIT margin"),
    ("implied_price", "wacc", "ebit_margin", "Implied price: WACC vs EBIT margin"),
    ("implied_price", "growth_fade_to", "revenue_growth", "Implied price: growth fade vs starting growth"),
    ("tv_share", "wacc", "terminal_growth", "Terminal value share: WACC vs terminal growth"),
    ("exit_multiple", "wacc", "terminal_growth", "Implied exit multiple: WACC vs terminal growth"),
]


def _range(name: str) -> tuple[float, float]:
    item = next(i for i in MODEL.inputs if i.name == name)
    return item.low, item.high


def surface(output: str, x: str, y: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = MODEL.base_case()
    xs = np.linspace(*_range(x), N)
    ys = np.linspace(*_range(y), N)
    z = np.full((N, N), np.nan)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            try:
                z[j, i] = MODEL.evaluate({**base, x: xv, y: yv})[output]
            except ValueError:  # WACC at or below terminal growth: no valuation
                pass
    return xs, ys, z


def build() -> go.Figure:
    fig = make_subplots(rows=3, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]] * 3,
                        subplot_titles=[p[3] for p in PANELS])
    for idx, (output, x, y, title) in enumerate(PANELS):
        xs, ys, z = surface(output, x, y)
        row, col = idx // 2 + 1, idx % 2 + 1
        fig.add_trace(go.Surface(x=xs, y=ys, z=z, colorscale=AMBER, showscale=False,
                                 hovertemplate=f"{x}: %{{x:.4f}}<br>{y}: %{{y:.4f}}<br>{output}: %{{z:.3f}}<extra></extra>"),
                      row=row, col=col)
        scene = fig.layout[f"scene{idx + 1 if idx else ''}"]
        scene.xaxis.title.text = x
        scene.yaxis.title.text = y
        scene.zaxis.title.text = output
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0b0d", plot_bgcolor="#0f1115",
                      font=dict(family="IBM Plex Mono, monospace", color="#c8cdd6", size=11),
                      title=dict(text="DCF surfaces (dcf_v2), other drivers at base", font=dict(color="#f0a500")),
                      height=1500, margin=dict(l=10, r=10, t=80, b=10))
    return fig


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "dcf_surfaces.html"
    build().write_html(path, include_plotlyjs=True, full_html=True)
    print(f"Written {path}")
