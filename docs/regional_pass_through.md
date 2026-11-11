# Regional diesel pass-through

Model: examples/regional_pass_through.py. Outputs are long-run (steady-state) pump prices ex-tax.

## Estimates (weekly, lags 0 to 4, Newey-West SE with 8 lags)

Source data: EU Weekly Oil Bulletin price history (pre-tax diesel, EU aggregate, 2005 to 2026); FRED DCOILBRENTEU, DDFUELNYH (NY Harbor ULSD), GASDESW (US retail diesel), DEXUSEU.

Error-correction form: change in pump price = gamma x (pump minus ULSD minus margin) at t-1, plus short-run crude and crack terms. Long-run ULSD pass-through is fixed at 1.

| Region | Sample | n | Adjustment speed gamma (SE) | Half-life (weeks) | Short-run ULSD pass-through, wk 0-4 | Margin |
|---|---|---|---|---|---|---|
| EU ex-tax (EUR/L) | 2005 | 850 | -0.023 (0.011) | 30.4 | 0.44 | 0.150 |
| EU ex-tax (EUR/L) | 2015 | 519 | -0.038 (0.018) | 17.8 | 0.41 | 0.172 |
| US retail (USD/gal) | 1994 | 1,047 | -0.013 (0.005) | 53.7 | 0.57 | 1.047 |
| US retail (USD/gal) | 2015 | 607 | -0.038 (0.010) | 17.7 | 0.51 | 1.197 |

The model uses the 2015 margins. Fitted gamma values are small, so the adjustment is slow and the timing is uncertain.

## Assumptions not estimated

- Freight passes to the pump in full (FREIGHT_PASS_THROUGH = 1.0).
- Margins are static means over each sample, so they carry some look-ahead.
- No standard errors on the margins.
