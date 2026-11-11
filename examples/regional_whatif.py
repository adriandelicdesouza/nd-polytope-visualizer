from __future__ import annotations

from regional_pass_through import MODEL

# Steady-state pump impact of single-driver what-ifs against the base case.
# Speed of pass-through: the estimated half-life is about 18 weeks (docs/regional_pass_through.md).
SCENARIOS = {
    "Base": {},
    "Crude +$10": {"brent_price": 100.0},
    "Crude -$15": {"brent_price": 75.0},
    "Crack +$10": {"diesel_crack": 40.0},
    "Crack to $60": {"diesel_crack": 60.0},
    "EUR 1.20": {"usd_per_eur": 1.20},
    "EUR 1.00": {"usd_per_eur": 1.00},
    "Freight +2c/L": {"freight_per_litre": 0.04},
}

if __name__ == "__main__":
    inputs = MODEL.base_case()
    base = MODEL.evaluate(inputs)
    print(f"{'Scenario':<16}{'EU EUR/L':>10}{'EU change':>12}{'US USD/gal':>12}{'US change':>12}")
    for name, change in SCENARIOS.items():
        out = MODEL.evaluate({**inputs, **change})
        d_eu = (out["eu_pump_ex_tax"] - base["eu_pump_ex_tax"]) * 100
        d_us = (out["us_pump_ex_tax"] - base["us_pump_ex_tax"]) * 100
        print(f"{name:<16}{out['eu_pump_ex_tax']:>10.3f}{d_eu:>+9.1f} c/L{out['us_pump_ex_tax']:>12.3f}{d_us:>+9.1f} c/g")
    print("Pump prices are ex-tax, steady state. Changes are in cents.")
