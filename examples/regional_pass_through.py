from __future__ import annotations

from nd_geometry.model import Input, Model, Output

# Steady-state diesel pump prices (ex-tax) from crude, crack, FX and freight.
# Long-run ULSD pass-through is fixed at 1; margins and adjustment speeds are estimated in
# docs/regional_pass_through.md (error-correction regression, Newey-West SE, 2015-2026 samples).
LITRES_PER_BARREL = 158.987
GALLONS_PER_BARREL = 42.0
LITRES_PER_GALLON = 3.785411784

EU_MARGIN_EUR_PER_L = 0.172  # since 2015, EU aggregate, pre-tax
US_MARGIN_USD_PER_GAL = 1.197  # since 2015, US retail diesel

# Assumption, not estimated: freight is passed to the pump in full (1.0).
FREIGHT_PASS_THROUGH = 1.0


def regional_pass_through(values: dict[str, float]) -> dict[str, float]:
    """Long-run EU and US pump prices (ex-tax) from crude, diesel crack, FX and freight."""
    ulsd_usd_per_bbl = values["brent_price"] + values["diesel_crack"]
    freight = values["freight_per_litre"] * FREIGHT_PASS_THROUGH  # USD per litre

    eu = EU_MARGIN_EUR_PER_L + (ulsd_usd_per_bbl / LITRES_PER_BARREL + freight) / values["usd_per_eur"]
    us = US_MARGIN_USD_PER_GAL + ulsd_usd_per_bbl / GALLONS_PER_BARREL + freight * LITRES_PER_GALLON
    return {"eu_pump_ex_tax": eu, "us_pump_ex_tax": us}


MODEL = Model(
    name="Regional diesel pump pass-through",
    inputs=(
        Input("brent_price", "$/bbl", 50.0, 120.0, 80.0),
        Input("diesel_crack", "$/bbl", 10.0, 60.0, 30.0),
        Input("usd_per_eur", "USD/EUR", 1.00, 1.25, 1.12),
        Input("freight_per_litre", "USD/L", 0.00, 0.10, 0.02),
    ),
    outputs=(Output("eu_pump_ex_tax", "EUR/L"), Output("us_pump_ex_tax", "USD/gal")),
    function=regional_pass_through,
)

if __name__ == "__main__":
    print(regional_pass_through({"brent_price": 80.0, "diesel_crack": 30.0, "usd_per_eur": 1.12, "freight_per_litre": 0.02}))
