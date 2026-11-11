from __future__ import annotations

from nd_geometry.model import Input, Model, Output
from nd_geometry.probability import RiskEvent

LITRES_PER_BARREL = 158.987
LITRES_PER_KM = 0.35
OTHER_COST_PER_KM = 0.45
DELAY_COST_PER_DAY = 800.0
DRIVING_HOURS_PER_DAY = 10.0


def freight_oil(values: dict[str, float]) -> dict[str, float]:
    """Road freight cost per tonne and transit time, linked to the oil price."""
    diesel_per_litre = (values["brent_price"] + values["diesel_crack"]) / LITRES_PER_BARREL
    fuel_per_km = diesel_per_litre * (1 + values["fuel_shock"]) * LITRES_PER_KM
    distance = values["distance_km"]

    trip_cost = (fuel_per_km + OTHER_COST_PER_KM) * distance + values["port_delay_days"] * DELAY_COST_PER_DAY
    tonnes_carried = values["capacity_tonnes"] * values["load_factor"]
    transit_days = distance / (values["average_speed_kmh"] * DRIVING_HOURS_PER_DAY) + values["port_delay_days"]

    return {"cost_per_tonne": trip_cost / tonnes_carried, "transit_days": transit_days}


MODEL = Model(
    name="Road freight linked to oil",
    inputs=(
        Input("brent_price", "$/bbl", 50.0, 120.0, 80.0),
        Input("diesel_crack", "$/bbl", 15.0, 45.0, 30.0),
        Input("distance_km", "km", 300.0, 1500.0, 800.0),
        Input("capacity_tonnes", "t", 10.0, 40.0, 20.0),
        Input("load_factor", "%", 0.60, 1.00, 0.85),
        Input("average_speed_kmh", "km/h", 70.0, 95.0, 80.0),
        Input("port_delay_days", "days", 0.0, 15.0, 0.0),
        Input("fuel_shock", "%", 0.0, 0.30, 0.0),
    ),
    outputs=(Output("cost_per_tonne", "$"), Output("transit_days", "days")),
    function=freight_oil,
)

# Illustrative probabilities only: replace with your own risk assessment.
EVENTS = (
    RiskEvent("port_delay_days", probability=0.25, low=1.0, high=10.0),
    RiskEvent("fuel_shock", probability=0.15, low=0.05, high=0.30),
)
