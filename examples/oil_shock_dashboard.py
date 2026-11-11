from __future__ import annotations

import json
from pathlib import Path

from regional_pass_through import MODEL as REG
from sea_freight_model import CORRELATION, EVENTS, MODEL as SEA
from nd_geometry.probability import monte_carlo, quantile_profile

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
TEMPLATE = HERE / "oil_shock_template.html"
GAMMA = -0.038  # weekly adjustment speed, since 2015 (docs/regional_pass_through.md)
WEEKS = 26

# (name, regional overrides, sea overrides). Brent moves both models.
# Shocks are changes from base (Brent $90, crack $30). Brent moves both models.
SCENARIOS = [
    ("Base", {}, {}),
    ("Brent +$15", {"brent_price": 105.0}, {"brent_price": 105.0}),
    ("Brent +$30 (Hormuz-type)", {"brent_price": 120.0}, {"brent_price": 120.0}),
    ("Crack to $60", {"diesel_crack": 60.0}, {}),
    ("10-day port strike", {}, {"port_delay_days": 10.0}),
    ("Combined: Brent $120, crack $60, 10-day strike",
     {"brent_price": 120.0, "diesel_crack": 60.0}, {"brent_price": 120.0, "port_delay_days": 10.0}),
]


def tornado(model, base_in, output):
    rows = []
    for inp in model.inputs:
        lo = model.evaluate({**base_in, inp.name: inp.low})[output]
        hi = model.evaluate({**base_in, inp.name: inp.high})[output]
        rows.append({"name": inp.name, "lo": lo, "hi": hi})
    rows.sort(key=lambda r: -abs(r["hi"] - r["lo"]))
    return rows


def build_payload() -> dict:
    reg_in, sea_in = REG.base_case(), SEA.base_case()
    reg_b, sea_b = REG.evaluate(reg_in), SEA.evaluate(sea_in)

    scen = []
    for name, ro, so in SCENARIOS:
        r = REG.evaluate({**reg_in, **ro})
        s = SEA.evaluate({**sea_in, **so})
        scen.append({
            "name": name, "eu": r["eu_pump_ex_tax"], "us": r["us_pump_ex_tax"],
            "deu": (r["eu_pump_ex_tax"] - reg_b["eu_pump_ex_tax"]) * 100,
            "dus": (r["us_pump_ex_tax"] - reg_b["us_pump_ex_tax"]) * 100,
            "teu": s["cost_per_teu"], "dteu": s["cost_per_teu"] - sea_b["cost_per_teu"],
            "days": s["transit_days"],
        })

    # Pump path after a $30 Brent shock: the gap to the long-run level closes at speed GAMMA each week.
    shock = REG.evaluate({**reg_in, "brent_price": 120.0})
    d_eu = shock["eu_pump_ex_tax"] - reg_b["eu_pump_ex_tax"]
    d_us = shock["us_pump_ex_tax"] - reg_b["us_pump_ex_tax"]
    path = [{"week": t, "eu": d_eu * (1 - (1 + GAMMA) ** t) * 100, "us": d_us * (1 - (1 + GAMMA) ** t) * 100}
            for t in range(WEEKS + 1)]

    prof = quantile_profile(SEA, EVENTS, "brent_price", "cost_per_teu", points=5, draws=3000, seed=1,
                            correlation=CORRELATION)
    res = monte_carlo(SEA, EVENTS, draws=5000, seed=1, correlation=CORRELATION)
    q = res.quantiles("cost_per_teu")
    freq = res.event_frequencies()

    tor_eu = [{"name": r["name"], "dlo": (r["lo"] - reg_b["eu_pump_ex_tax"]) * 100,
               "dhi": (r["hi"] - reg_b["eu_pump_ex_tax"]) * 100}
              for r in tornado(REG, reg_in, "eu_pump_ex_tax")]
    tor_sea = [{"name": r["name"], "dlo": r["lo"] - sea_b["cost_per_teu"], "dhi": r["hi"] - sea_b["cost_per_teu"]}
               for r in tornado(SEA, sea_in, "cost_per_teu")]

    return {
        "base": {"eu": reg_b["eu_pump_ex_tax"], "us": reg_b["us_pump_ex_tax"], "teu": sea_b["cost_per_teu"]},
        "scen": scen,
        "path": path,
        "prof": {k: [float(x) for x in prof[k]] for k in ("value", "p10", "p50", "p90")},
        "risk": {"p10": float(q[0.1]), "p50": float(q[0.5]), "p90": float(q[0.9])},
        "torEU": tor_eu,
        "torSea": tor_sea,
        "notes": [
            "Pump prices are steady state and ex-tax. The path uses one weekly adjustment speed (since 2015), "
            "not the full short-run dynamics, so early weeks are understated.",
            "Sea cost is fuel, time-charter hire, Suez toll and port fees per TEU carried. The toll and port fees are flagged placeholders. Terminal handling is excluded.",
            f"Port delay occurs in {freq['port_delay_days']:.1%} of risk draws; fuel shock in {freq['fuel_shock']:.1%}. "
            "The fuel-shock share comes from Brent history since 2015, including the 2020 and 2022 spikes.",
        ],
    }


def build_html() -> str:
    return TEMPLATE.read_text().replace("__PAYLOAD__", json.dumps(build_payload()))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "oil_shock_dashboard.html"
    path.write_text(build_html())
    print(f"Written {path}")
