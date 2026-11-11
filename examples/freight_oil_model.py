from __future__ import annotations

import numpy as np

from pathlib import Path

from nd_geometry.history import price_shock_history
from nd_geometry.model import Input, Model, Output
from nd_geometry.probability import RiskEvent

# All money is USD. Sources are listed in docs/freight_assumptions.md.
LITRES_PER_BARREL = 158.987
DRIVING_HOURS_PER_DAY = 13.0  # Canadian daily driving limit south of 60N (US limit: 11)
NON_FUEL_COST_PER_KM = 1.854 / 1.609344  # ATRI 2025 non-fuel operating cost, USD/mile to USD/km
DELAY_COST_PER_DAY = (1.028 + 0.40) * 500.0  # ATRI driver pay+benefits and truck/trailer per mile x 500 miles/day


def freight_oil(values: dict[str, float]) -> dict[str, float]:
    """Road freight cost per tonne and minimum transit time, linked to the oil price."""
    diesel_per_litre = (values["brent_price"] + values["diesel_crack"]) / LITRES_PER_BARREL
    fuel_per_km = diesel_per_litre * (1 + values["fuel_shock"]) * values["fuel_consumption"] / 100.0
    distance = values["distance_km"]

    trip_cost = (fuel_per_km + NON_FUEL_COST_PER_KM) * distance + values["port_delay_days"] * DELAY_COST_PER_DAY
    tonnes_carried = values["capacity_tonnes"] * values["load_factor"]
    transit_days = distance / (values["average_speed_kmh"] * DRIVING_HOURS_PER_DAY) + values["port_delay_days"]

    return {"cost_per_tonne": trip_cost / tonnes_carried, "transit_days": transit_days}


MODEL = Model(
    name="Road freight linked to oil",
    inputs=(
        Input("brent_price", "$/bbl", 50.0, 120.0, 80.0),
        Input("diesel_crack", "$/bbl", 15.0, 90.0, 30.0),
        Input("fuel_consumption", "L/100km", 30.0, 40.0, 35.0),
        Input("distance_km", "km", 300.0, 1500.0, 800.0),
        Input("capacity_tonnes", "t", 10.0, 40.0, 20.0),
        Input("load_factor", "%", 0.60, 1.00, 0.85),
        Input("average_speed_kmh", "km/h", 70.0, 95.0, 80.0),
        Input("port_delay_days", "days", 0.0, 15.0, 0.0),
        Input("fuel_shock", "%", 0.0, 0.45, 0.0),
    ),
    outputs=(Output("cost_per_tonne", "USD"), Output("transit_days", "days")),
    function=freight_oil,
)

DATA = Path(__file__).resolve().parents[1] / "data"

# Fuel shock: diesel up 5% or more over 4 weeks, from US weekly retail diesel (FRED GASDESW, EIA data) since 2015.
_FUEL = price_shock_history(DATA / "us_diesel_retail_weekly_fred_GASDESW.csv", "GASDESW", since="2015-01-01")

# Port delay: days lost to North American gateway strikes and lockouts, 2020-2024 (see docs/freight_assumptions.md).
# Probability is disrupted days / days in the period: 44 / 1827.
_PORT_DISRUPTION_DAYS = (12.0, 5.0, 13.0, 3.0, 3.0, 8.0)

EVENTS = (
    RiskEvent.from_history("port_delay_days", probability=44 / 1827, severities=_PORT_DISRUPTION_DAYS),
    RiskEvent.from_history("fuel_shock", probability=_FUEL.probability, severities=_FUEL.severities),
)

# Port delays and fuel shocks often share a cause (e.g. a supply disruption). Illustrative 0.5.
CORRELATION = np.array([[1.0, 0.5], [0.5, 1.0]])
