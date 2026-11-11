# Sea freight model assumptions and sources

Model: examples/sea_freight_model.py. One 8,000 TEU container ship, Shanghai to Rotterdam. Output is cost per TEU carried (fuel plus time-charter hire) and transit days. Money is USD.

Included: fuel, time-charter hire, Suez canal toll, and port fees. Not included: terminal handling, cargo handling, overhead, and fuel burned while idle in port.

## Assumptions

| Assumption | Value used | Range | Source | Note |
|---|---|---|---|---|
| Ship size | 8,000 TEU | fixed | Assumption | Matches the size in the fuel-burn source below. |
| Fuel burn at reference | 225 t/day at 24 kn | fixed | Notteboom, via transportgeography.org: "a containership of around 8,000 TEU would consume about 225 tons of bunker fuel per day at 24 knots" (https://transportgeography.org/contents/chapter4/transportation-and-energy/fuel-consumption-containerships/) | Other sources give 250 to 350 t/day for large fast ships (maritimepage.com), so this is a lower-end figure. |
| Burn vs speed | cubic law (exponent 3.0) | fixed | Same source: 150 t/day at 21 kn, which implies an exponent of about 3.0 | Derived from two points. |
| Service speed | 20 kn | 16 to 24 | Judgment. 16 kn is the speed in a routing study (porteconomicsmanagement.org); 24 kn is the reference speed | |
| Route distance | 10,600 nm (Suez) | 10,600 to 13,800 | Port Economics Management Program, Shanghai to Rotterdam via Suez 10,600 nm; via the Cape 13,800 nm (https://porteconomicsmanagement.org/routing-options-shanghai-rotterdam/) | The top of the range is the Cape route, so the input also covers the Red Sea diversion. |
| VLSFO price | Brent x 7.53 x 1.10 | premium 0.95 to 1.25 | Ship and Bunker G20 VLSFO Q2 2026 average 856.50 $/mt (https://shipandbunker.com/prices/port/AV-G20); Brent Q2 2026 average 102.93 $/bbl (https://shipandbunker.com/news/world/352050-eia-slashes-brent-forecast-as-supply-disruptions-ease-ship-bunker-cuts-bunker-price-outlook-sharply). 856.50 / 7.53 / 102.93 = 1.105 | Premium is derived from one quarter. Ship and Bunker publishes quarterly averages on the same page. |
| Barrels per tonne | 7.53 | fixed | Ship and Bunker conversion (https://shipandbunker.com/prices/port/AV-G04) | |
| Brent price | 90 $/bbl | 60 to 120 | EIA September STEO, Q4 2026 Brent forecast 90.66 $/bbl, reported by Ship and Bunker (https://shipandbunker.com/news/world/341945-ship-bunker-raises-q4-bunker-price-outlook-by-112mt-as-eia-sees-brent-holding-near-90-through-year-end) | |
| Time-charter hire | 50,000 USD/day | 40,000 to 60,000 | New ConTex: 5,700 and 6,500 TEU at about 58,000 to 59,000 USD/day, 2026 (https://container-news.com/containership-charter-rates-strengthen-as-mid-sized-tonnage-leads-gains/); 9,000 TEU at 45,000 USD/day (https://container-news.com/maersk-cma-cgm-and-hapag-lloyd-drive-charter-activity/) | 50,000 is my interpolation for 8,000 TEU. Not a quoted 8,000 TEU rate. |
| Load factor | 85% | 60 to 95% | Assumption | No source yet. |
| Port delay | 0 days | 0 to 15 | Event, see below | |

## Events

| Event | Probability | Severity | Source |
|---|---|---|---|
| Fuel shock (VLSFO up with Brent) | 32.7% of weeks since 2015 (Brent up 5% or more over 4 weeks) | Median 10%, maximum 138% | FRED DCOILBRENTEU weekly means, kept in `data/brent_weekly_fred_DCOILBRENTEU.csv`; computed by `nd_geometry.history.price_shock_history`. The share is high because the period includes the 2020 rebound and the 2022 spike. |
| Port delay | 2.4% (44 disrupted days out of 1,827, 2020 to 2024) | 12, 5, 13, 3, 3 and 8 days, resampled | Montreal 12-day job action (Aug 2020) and 5-day strike (Apr 2021); 13-day BC strike (Jul 2023); Montreal 3-day strike (Sep 2024) (https://www.cbc.ca/1.7337888); US ILA 3-day strike (Oct 2024) and BC lockout of about 8 days (Nov 2024) (https://gofreight.com/blog/port-strikes-freight). These are gateway strikes, not Shanghai or Rotterdam. |

## Caveats

- The port sample is six events. Congestion and weather delays are not in it.
- The fuel-shock frequency comes from the Brent history, so it reflects volatility over 2015 to 2026. Check it against a shorter window before using it.
- The correlation of 0.5 between port delays and fuel shocks is illustrative. No source yet.
- Charter hire, the VLSFO premium and the load factor are judgments drawn from the sources above, not quoted figures for this exact ship.
- The Suez toll and port fees are flagged placeholders. Replace them with the vessel's SCA toll and the port tariffs before use.

## What the rebuild changed

- The earlier model was for road freight: trucks, kilometres, driving hours and ATRI truck costs. It was replaced with a sea-freight model.
- Fuel is now VLSFO priced from Brent, not diesel. The fuel-shock history follows Brent, not US retail diesel.
- Delay cost is now time-charter hire on the ship, not truck and driver cost.

## Port and canal fees (added)

| Assumption | Value used | Range | Source | Flag |
|---|---|---|---|---|
| Suez canal toll per transit | 500,000 USD | 0 to 800,000 | Sea-routes calculator: Suez fee of 400,000 to 600,000 USD for large container ships (https://everycalculators.com/sea-routes-calculator.html) | Placeholder. Low-quality source. The toll depends on the vessel's SCNT, and no reliable headline figure was found (https://carraglobe.com/suez-vs-cape-of-good-hope-transit-times/). Set to 0 for the Cape route. |
| Suez surcharge | 12% on transit dues | not modelled | Suez Canal Authority surcharge (https://www.iss-shipping.com/sca-introduces-revised-toll-surcharges/) | Effective date conflicts across sources. Not modelled separately; the toll input is the all-in placeholder. |
| 15% container rebate | not modelled | n/a | Sources conflict: one reports it withdrawn from 7 April 2026 (https://dredgewire.com/suez-canal-scraps-15-container-rebate-as-hormuz-crisis-drains-traffic-and-reshapes-global-shipping-routes); another reports it in force in February 2026 (https://businessmonthlyeg.com/15-toll-cut-lures-mega-ships-back-to-the-suez-canal/) | Unresolved. |
| Port fixed fee per call | 40,000 USD | 20,000 to 80,000 | Dockage 5,000 to 20,000, pilotage 2,000 to 10,000, tug 1,000 to 5,000, admin 1,000 to 3,000 (https://everycalculators.com/shipping-routes-calculator.html), summed to about 38,000 at the top of ranges | Estimate. The calculator ranges are low-quality. Rotterdam and Shanghai tariffs were not retrieved. |
| Port dues per GT | 0.50 USD/GT per call | 0.25 to 1.00 | Port dues of 0.50 to 10 USD per GT (Mammadov 2023, quoted at https://medium.com/shipping-intel/voyage-cost-port-charges-55a0a5db3eed) | Range is wide. The base is the low end. |
| Gross tonnage | 90,000 GT | fixed | Assumption | Not sourced. Scales the port dues. |
| Port calls per voyage | 2 | fixed | Origin and destination | |
