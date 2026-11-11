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
## Phase 29: any number of inputs (support-point polytope)

The corner method needs 2**n model runs, which is fine to about 12 inputs. `belief_density` now switches
automatically above `corner_limit` (default 12) to `nd_geometry.support.support_polytope`: for a few hundred
directions in output space it maximises the direction's score over the input box by cached coordinate search
(cost grows with n times directions, not 2**n). It is exact for models monotone in each input (the linear and
DCF cases tested) and an inner approximation otherwise, so the polytope never claims outputs the model cannot
make. `SupportResult.vertex_inputs` and `input_roles` say which input values build each vertex and whether an
input is one-sided, a trade-off or interior. Tested to 60 inputs.

Backtest on 22 companies (1,875 forecasts, filings-only beliefs): the 80% band holds 42% and the 50% band 26%.
Widening the ranges about 3x brings the 80% band to 67%. Beliefs from filing history are over-confident.
## Phase 30: big DCFs end to end

- `--wacc-build` replaces the single WACC with rf, equity beta, equity risk premium, cost of debt and debt weight.
- `--per-year` gives margin, tax, capex, D&A, working capital and reinvestment one input per forecast year
  (NFLX, 5 years, both flags: 39 inputs, whole report in about 30 s).
- The report now includes the full output polytope (`hull_report`: vertices, facets, volume, bounds, which inputs
  build each extreme, one-sided / trade-off / interior inputs, model runs and infeasible combinations).
- `story.cheapest_story`: the smallest joint move of every input (in range half-widths) that reproduces the market
  price, or a statement that the box cannot reach it. Single-driver solves are usually unreachable with dozens of inputs.
- `--range-width F` widens every filing-based range. The basket backtest says F of about 3 gets the 80% band from
  42% to 67%; it does not reach 75-80%, so a per-output fitted width is still open.
## Phase 31: option-implied density combined with the DCF

`nd_geometry.option_combine` takes a single-expiry chain from the options project (Yahoo via its cached adapter),
keeps usable quotes (positive bid and ask, spread at most the mid, strike within 60% to 160% of spot), recovers
the risk-neutral density of the price at expiry (Breeden-Litzenberger on a robust quadratic smile in log-moneyness),
and combines it with the DCF belief density of fair value per share:

- option density mapped to today by the forward factor e^{-(r-q)T} (an assumption; stated in the output)
- combined = normalised product of the two densities; mixture (50/50) reported beside it
- overlap reported before normalising: near zero means the DCF and the market disagree

Run (needs the options project at ~/options-pricing-model; pydantic and yfinance in this venv):

    PYTHONPATH=src .venv/bin/python -m nd_geometry.option_combine NFLX --expiry 2027-01-15 \
      --facts-file /tmp/nflx_facts.json --wacc-build --per-year --years 5 --json /tmp/nflx_combo.json
## Phase 32: fixes from the Netflix check

- `--market-price` is checked against the live close; a difference over 15% prints a warning. The NFLX runs had used a
  placeholder $1,100 against a $63.26 close, which made the reverse DCF meaningless.
- Reverse DCF no longer divides by zero when no belief draw is near a price far outside the belief mass.
- Base case vs belief median: for NFLX the base-case price ($47.87) sits well above the belief median ($33.31)
  because the other-reinvestment range is 0% to 15% around a 0.9% base, so its belief mean is about 5.3%. That comes
  from the inputs, not a bug.