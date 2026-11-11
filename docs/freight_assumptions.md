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

## Event probabilities (sourced)

| Event | Probability | Severity | Source |
|---|---|---|---|
| Fuel shock (diesel up 5%+ over 4 weeks) | 13.3% of weeks since 2015 (615 weeks; 82 shock weeks) | Historical rises, median 7.5%, max 41% | US weekly retail diesel, FRED series GASDESW (EIA data), kept in `data/us_diesel_retail_weekly_fred_GASDESW.csv`; computed by `nd_geometry.history.price_shock_history` |
| Port delay | 2.4% (44 disrupted days out of 1,827, 2020 to 2024) | 12, 5, 13, 3, 3 and 8 days, resampled | Montreal 12-day job action (Aug 2020) and 5-day strike (Apr 2021); 13-day BC strike (Jul 2023); Montreal 3-day strike (Sep 2024) (https://www.cbc.ca/1.7337888); US ILA 3-day strike (Oct 2024) and BC lockout of about 8 days (Nov 2024) (https://gofreight.com/blog/port-strikes-freight) |

Caveats: the port sample is six events, so the estimate is rough. Congestion and weather delays are not in it. Fuel shocks are US pump prices, not a pure wholesale measure.

## Still unvalidated

- The correlation of 0.5 between port delays and fuel shocks is illustrative. No source yet.
- Canadian operating costs and wages are not sourced here. ATRI is US data.
