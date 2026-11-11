from __future__ import annotations

from regional_pass_through import MODEL

# Steady-state pump impact of single-driver what-ifs against the base case.
# Speed of pass-through: the estimated half-life is about 18 weeks (docs/regional_pass_through.md).
# Values are changes from the base case (Brent $90, crack $30, EUR/USD 1.12, freight 2c/L).
SCENARIOS = {
    "Base": {},
    "Crude +$10": {"brent_price": +10.0},
    "Crude -$15": {"brent_price": -15.0},
    "Crack +$10": {"diesel_crack": +10.0},
    "Crack to $60": {"diesel_crack": +30.0},
    "EUR 1.20": {"usd_per_eur": 1.20},
    "EUR 1.00": {"usd_per_eur": 1.00},
    "Freight +2c/L": {"freight_per_litre": +0.02},
}

if __name__ == "__main__":
    inputs = MODEL.base_case()
    base = MODEL.evaluate(inputs)
    print(f"{'Scenario':<16}{'EU EUR/L':>10}{'EU change':>12}{'US USD/gal':>12}{'US change':>12}")
    for name, change in SCENARIOS.items():
        shifted = dict(inputs)
        for key, value in change.items():
            shifted[key] = value if key == "usd_per_eur" else inputs[key] + value  # EUR rate is absolute
        out = MODEL.evaluate(shifted)
        d_eu = (out["eu_pump_ex_tax"] - base["eu_pump_ex_tax"]) * 100
        d_us = (out["us_pump_ex_tax"] - base["us_pump_ex_tax"]) * 100
        print(f"{name:<16}{out['eu_pump_ex_tax']:>10.3f}{d_eu:>+9.1f} c/L{out['us_pump_ex_tax']:>12.3f}{d_us:>+9.1f} c/g")
    print("Pump prices are ex-tax, steady state. Changes are in cents.")
