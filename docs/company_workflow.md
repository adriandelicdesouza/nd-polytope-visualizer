# Company workflow: filings, reverse DCF, revenue build, correlation, backtest

One command takes a ticker to a belief-weighted DCF and a check on how well past beliefs held up:

```bash
export ND_SEC_USER_AGENT="Your Name you@example.com"   # the SEC requires a contact
python -m nd_geometry.company NFLX --build examples/specs/nflx_revenue_build.json \
    --set wacc=0.095,0.115,0.135 --out outputs/nflx_company.html
```

Options: `--correlation none|prior|historical`, `--market-price`, `--facts-file` (offline), `--no-backtest`, `--years`.
Tested on Netflix against SEC company facts (filed through the Q3 2026 10-Q) and Yahoo's price for 2026-11-10.

## 1. Inputs from filings (`filings.py`, `prices.py`)

- XBRL company facts give base revenue, share count (cover page), net debt (debt less cash and short-term investments at
  the latest balance sheet; operating leases excluded) and a low / base / high range for every driver.
- A range spans the latest five fiscal years; the base is the mean of the latest three (`base_years`). Every driver is
  labelled `filings`, `default` or `override`. WACC and terminal growth are not in filings: they are defaults unless
  overridden (the Netflix run uses your 9.5% / 11.5% / 13.5% WACC).
- `other_reinvest_pct` is the cash-conversion gap, (NOPAT + D&A - operating cash flow) / revenue net of the working-capital
  drag. It carries reinvestment that is not capex, such as Netflix content spend, so such companies are not overvalued.
- `as_of` hides anything filed later, which is what makes the backtest honest.
- The price reference is Yahoo's last close, with split history so past share counts match the price series.

## 2. Reverse DCF (`reverse.py`)

`reverse_dcf(result, market_price)` reports the belief mass at or above the market price, the percentile it sits at, and,
among draws priced near the market (a kernel that widens until 200 draws count), each driver's implied mean and p10 to
p90 as a shift in belief standard deviations. `solve_driver` moves one driver alone until the price matches, or says the
price is out of that driver's range.

## 3. Revenue build (`revenue_build.py`)

`SegmentBuild` grows each stream by the product of named factors. A factor is a year-by-year path plus a shift ("add") or
multiplier ("scale"); that shift or multiplier is the model input, so beliefs sit on the build's own assumptions.
`examples/extract_nflx_build.py` converts `NFLX_Revenue_Build_Model.xlsx` to `examples/specs/nflx_revenue_build.json`.
The Python build matches the workbook's base, bull and bear consolidated revenue to one cent for FY2027 to FY2036
(tests/test_revenue_build.py). Ranges are the workbook's own scenario spreads; FX is a 0 to 1 scale on the workbook's
bear FX path, so base and bull sit at 0.

## 4. Correlated drivers (`correlation.py`)

A Gaussian copula over each driver's marginal belief. Marginals are unchanged; only dependence is added. Matrices are
validated (symmetric, unit diagonal, positive semi-definite, with a clear error). `default_dcf_correlation` gives stated
priors (rates, capex with D&A, build drivers with each other and with margin). `historical_correlation` shrinks them
toward the sample correlation of the filed history (30% weight; annual history is short). The priors are assumptions.

## 5. Backtest (`backtest.py`)

At every past fiscal year-end the inputs are rebuilt from filings available that day, beliefs are pushed through the model,
and revenue, EBIT and NOPAT + D&A - capex one to three years ahead are compared with what was later reported. Output:
central 50% and 80% band coverage, mean PIT, KS test, sharpness, a belief-width sweep (`calibrate_width`), a pooled
basket (`backtest_basket`) and `price_check` (where the market price sat in the as-of belief distribution).

## What the Netflix test showed

- Beliefs built from filed history were over-confident: the central 80% band held 42% of outcomes (105 forecasts, 13
  origins). Widening them 3x lifted that only to about 54%.
- Errors are one-sided, not just narrow. Revenue +1y came in under the forecast (mean PIT 0.38); EBIT and the cash
  flow proxy came in over it (mean PIT 0.71 to 0.85), because Netflix's margin kept expanding past its trailing range.
  Changing which years set the base (1, 2, 3 or 5) did not change this.
- At $63.26 the market sits above all belief mass (0.0% at or above), with filings-based margin and 11.5% WACC. The
  market-implied drivers are a higher margin (27.5% vs 24.3%), a lower WACC (10.5% vs 11.5%) and less reinvestment.
  The margin range is history, not Netflix's current guidance, so supplying one with `--set ebit_margin=...` is the first
  thing to try.
- Triangular beliefs on skewed ranges have a mean above the base case (`other_reinvest_pct`), which is why the belief
  median ($30.67) is below the base-case price ($41.66).

## Limits

- Filings-based history lags companies whose economics are changing; the backtest quantifies it but nothing here
  forecasts a regime change.
- One company is a weak calibration test. `backtest_basket` pools names but needs `ND_SEC_USER_AGENT`.
- `price_check` is not a calibration of a DCF as a price forecaster. For companies with persistently negative free cash
  flow in the as-of history the belief distribution sits below any market price.
- Net debt excludes operating leases; segment data are not in company facts, so builds come from a spec file.