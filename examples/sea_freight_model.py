from __future__ import annotations

from pathlib import Path

import numpy as np

from nd_geometry.history import price_shock_history
from nd_geometry.model import Input, Model, Output
from nd_geometry.probability import RiskEvent

# Sea freight: one 8,000 TEU container ship, Shanghai to Rotterdam, cost per TEU carried.
# Covers fuel and time-charter hire only. Canal tolls, port and terminal charges, cargo handling and
# overhead are not included. All money is USD. Sources are in docs/freight_assumptions.md.
SHIP_TEU = 8000
REF_BURN_T_PER_DAY = 225.0  # 8,000 TEU at 24 knots, Notteboom via transportgeography.org
REF_SPEED_KN = 24.0
BURN_EXPONENT = 3.0  # cubic law: 225 t/day at 24 kn vs 150 t/day at 21 kn gives about 3.0
BARRELS_PER_TONNE = 7.53  # Ship & Bunker conversion for VLSFO prices
HOURS_PER_DAY = 24.0


def sea_freight(values: dict[str, float]) -> dict[str, float]:
    """Cost per TEU carried and transit days for one container-ship voyage."""
    vlsfo_usd_per_t = values["brent_price"] * BARRELS_PER_TONNE * values["vlsfo_premium"] * (1 + values["fuel_shock"])
    speed = values["ship_speed_knots"]

    sea_days = values["route_distance_nm"] / (speed * HOURS_PER_DAY)
    burn_t_per_day = REF_BURN_T_PER_DAY * (speed / REF_SPEED_KN) ** BURN_EXPONENT
    fuel_cost = burn_t_per_day * sea_days * vlsfo_usd_per_t

    # Charter hire runs while the ship is at sea and while it waits in port. Idle-day fuel is not modelled.
    charter_days = sea_days + values["port_delay_days"]
    charter_cost = values["time_charter_usd_per_day"] * charter_days

    teu_carried = SHIP_TEU * values["load_factor"]
    return {"cost_per_teu": (fuel_cost + charter_cost) / teu_carried, "transit_days": charter_days}


MODEL = Model(
    name="Sea freight (container ship) linked to oil",
    inputs=(
        Input("brent_price", "$/bbl", 60.0, 120.0, 90.0),
        Input("vlsfo_premium", "x Brent", 0.95, 1.25, 1.10),
        Input("ship_speed_knots", "knots", 16.0, 24.0, 20.0),
        Input("route_distance_nm", "nm", 10600.0, 13800.0, 10600.0),
        Input("time_charter_usd_per_day", "USD/day", 40000.0, 60000.0, 50000.0),
        Input("load_factor", "%", 0.60, 0.95, 0.85),
        Input("port_delay_days", "days", 0.0, 15.0, 0.0),
        Input("fuel_shock", "%", 0.0, 0.45, 0.0),
    ),
    outputs=(Output("cost_per_teu", "USD/TEU"), Output("transit_days", "days")),
    function=sea_freight,
)

DATA = Path(__file__).resolve().parents[1] / "data"

# Fuel shock: Brent up 5% or more over 4 weeks, weekly means of FRED DCOILBRENTEU since 2015.
_FUEL = price_shock_history(DATA / "brent_weekly_fred_DCOILBRENTEU.csv", "DCOILBRENTEU", since="2015-01-01")

# Port delay: days lost to gateway strikes and lockouts, 2020-2024 (see docs/freight_assumptions.md).
_PORT_DISRUPTION_DAYS = (12.0, 5.0, 13.0, 3.0, 3.0, 8.0)

EVENTS = (
    RiskEvent.from_history("port_delay_days", probability=44 / 1827, severities=_PORT_DISRUPTION_DAYS),
    RiskEvent.from_history("fuel_shock", probability=_FUEL.probability, severities=_FUEL.severities),
)

# Port delays and fuel shocks often share a cause (e.g. a supply disruption). Illustrative 0.5.
CORRELATION = np.array([[1.0, 0.5], [0.5, 1.0]])
