# Freight model assumptions and sources

All money is USD. Values are the model defaults; ranges are the inputs the model sweeps.

| Assumption | Value used | Range | Source | Note |
|---|---|---|---|---|
| Non-fuel operating cost | 1.854 USD/mile (1.15 USD/km) | sweep not yet | ATRI 2026 Analysis of the Operational Costs of Trucking, 2025 data: https://www.truckersnews.com/news/article/15830055/truckings-operational-costs-reached-new-high-of-2336-per-mile-in-2025 | US industry average. Canadian costs will differ. |
| Driver pay and benefits | 1.028 USD/mile | within non-fuel cost | ATRI, as above | Used to derive delay cost. |
| Truck and trailer procurement | 0.40 USD/mile | within non-fuel cost | ATRI, as above | Used to derive delay cost. |
| Delay cost | 714 USD/day | fixed | Derived: (1.028 + 0.40) USD/mile x 500 miles/day | The 500 miles/day is my assumption. Idle-day cost is not sourced directly. |
| Fuel consumption | 35 L/100 km | 30 to 40 | ICCT: typical EU long-haul tractor-trailer 32.6 L/100 km, best-in-class 29.9 (https://theicct.org/publications/fuel-consumption-testing-tractor-trailers-EU-and-US); NRCan: Canadian fleet average 39.5 L/100 km in 1999 (https://publications.gc.ca/collections/Collection/M92-218-2001E.pdf) | Newer trucks use less. Fleets differ. |
| Driving hours per day | 13 h | fixed | Canadian HOS: maximum 13 hours driving per day south of 60N (https://matrackinc.com/hours-of-service-rules-in-canada). US limit is 11 h. | Legal maximum, so transit time is a lower bound. |
| Brent price | 80 USD/bbl | 50 to 120 | Recent reporting: Brent around 87 to 91 USD/bbl in 2026 (https://www.freightwaves.com/news/diesel-still-ripping-higher-than-the-rest-of-the-barrel-heres-why) | Range covers the recent level. |
| Diesel crack spread | 30 USD/bbl | 15 to 90 | 2025: NY Harbor and USGC ULSD cracks typically around 18 to 30 USD/bbl (https://www.kpler.com/blog/wos-refining-margins-tumble-further-asia-finds-support); 2026 peaks near 90 USD/bbl (https://www.cmegroup.com/newsletters/wti-insights-by-pvm/wti-insights-by-pvm-2026-04-05.html) | Upper bound set to the observed peak. |
| Litres per barrel | 158.987 | fixed | Unit conversion | |

## What the validation changed

- Non-fuel cost was 0.45 USD/km. ATRI implies about 1.15 USD/km, so the base-case cost per tonne roughly doubled.
- Driving hours went from 10 to 13 (legal limit), so transit time is now a minimum.
- Diesel crack range widened to cover the 2026 peak.
- Delay cost is now derived from ATRI, instead of a placeholder.

## Still unvalidated

- Probabilities of port delays and fuel shocks are placeholders. Source them from a risk assessment before using the output for decisions.
- Canadian operating costs and wages are not sourced here. ATRI is US data.
- Severity ranges for delays (1 to 10 days) and fuel shocks (5% to 30%) are not sourced.
