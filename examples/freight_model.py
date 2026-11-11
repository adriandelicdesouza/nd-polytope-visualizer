from __future__ import annotations

from nd_geometry.analysis import analyse
from nd_geometry.model import Input, Model, Output

TRUCK_CAPACITY_TONNES = 20.0
LITRES_PER_KM = 0.35
DRIVER_COST_PER_KM = 0.45
DRIVING_HOURS_PER_DAY = 10.0


def freight(values: dict[str, float]) -> dict[str, float]:
    distance = values["distance_km"]
    cost_per_km = values["diesel_price"] * LITRES_PER_KM + DRIVER_COST_PER_KM
    cost_per_tonne = cost_per_km * distance / (TRUCK_CAPACITY_TONNES * values["load_factor"])
    transit_days = distance / (values["average_speed_kmh"] * DRIVING_HOURS_PER_DAY)
    return {"cost_per_tonne": cost_per_tonne, "transit_days": transit_days}


MODEL = Model(
    name="Road freight (simplified)",
    inputs=(
        Input("distance_km", "km", 300.0, 1500.0, 800.0),
        Input("diesel_price", "$/litre", 1.40, 2.00, 1.70),
        Input("load_factor", "%", 0.60, 1.00, 0.85),
        Input("average_speed_kmh", "km/h", 70.0, 95.0, 80.0),
    ),
    outputs=(Output("cost_per_tonne", "$"), Output("transit_days", "days")),
    function=freight,
)

if __name__ == "__main__":
    result = analyse(MODEL)
    for name, (low, high) in result.output_bounds.items():
        print(f"{name}: {low:.2f} to {high:.2f}")
    print("Base case inside envelope:", result.base_inside)
