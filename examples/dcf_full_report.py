from __future__ import annotations

from pathlib import Path

from dcf_full_model import MODEL
from nd_geometry.analysis import analyse
from nd_geometry.baseline import compare_to_baseline
from nd_geometry.explain import explain
from nd_geometry.refine import refine
from nd_geometry.views import envelope_projection, input_surface, surface_figure, write_html

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

result = analyse(MODEL, method="random", samples=3000, seed=1)
refined, volumes = refine(result, rounds=6)

print("Envelope volume by refinement round:", [round(v, 1) for v in volumes])
print("Base case inside envelope:", refined.base_inside)
for line in explain(refined):
    print(line)

surface = input_surface(refined, "wacc", "terminal_growth", "enterprise_value", points=25)
write_html(surface_figure(surface), OUT / "dcf_wacc_terminal_value.html")
write_html(envelope_projection(refined, "enterprise_value", "year5_free_cash_flow"), OUT / "dcf_envelope.html")

refined.save(OUT / "dcf_baseline.json")
print("Baseline check against itself:", compare_to_baseline(refined, OUT / "dcf_baseline.json"))
