from __future__ import annotations

from pathlib import Path

from freight_oil_model import CORRELATION, EVENTS, MODEL
from nd_geometry.dashboard import write_dashboard
from nd_geometry.probability import monte_carlo, quantile_profile

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

result = monte_carlo(MODEL, EVENTS, draws=10_000, seed=1, correlation=CORRELATION)

print("Event frequencies:", {k: round(v, 3) for k, v in result.event_frequencies().items()})
for output in MODEL.output_names:
    quantiles = result.quantiles(output)
    print(f"{output}: P10 {quantiles[0.1]:.2f}, P50 {quantiles[0.5]:.2f}, P90 {quantiles[0.9]:.2f}")

profile = quantile_profile(MODEL, EVENTS, "brent_price", "cost_per_tonne", points=5, draws=3000, seed=1, correlation=CORRELATION)
print("cost_per_tonne by Brent price ($/bbl):")
for index, value in enumerate(profile["value"]):
    print(f"  {value:6.1f}: P10 {profile['p10'][index]:.2f}, P50 {profile['p50'][index]:.2f}, P90 {profile['p90'][index]:.2f}")

write_dashboard(result, OUT / "freight_oil_dashboard.html", title="Freight cost under oil and disruption risk")
print("Dashboard written.")
