# Roadmap

## Done

- Polytopes, projections, slicing and rotations.
- Parameter spaces, evaluation and sensitivity.
- Response geometry and response level sets.
- 4D level-set topology, with staircase triangulation of simplex sections (latest work).

## Done (recent)

- (Superseded) Road freight assumptions checked against sources (`docs/freight_assumptions.md`). This roughly doubled the base-case cost per tonne, because non-fuel cost was about 2.5 times too low, and set driving hours to the legal 13 h.
- Correlated risk events (`monte_carlo(..., correlation=...)`): Gaussian copula, marginal probabilities preserved, validated against the closed form for two events.
- Probability dashboard and oil-linked freight model (see earlier entries).

## Next

1. Source the event probabilities and severity ranges (port delay days, fuel shock size) before the pitch. They are still placeholders.
2. Validate Canadian operating costs and wages; the sourced figures are US (ATRI).
3. Oil shipping to regional prices (Europe and North America): link tanker or freight costs, regional crack spreads and pump-price pass-through, so a crude move can be priced before it reaches the pump. This needs regional data sources and a pass-through model.
4. Analytic check of Monte Carlo quantiles for a simple case.
5. Input constraints (e.g. portfolio weights) when a portfolio model is added.
6. Extend the 4D facet-cell work to 5D+ only if a real example needs it.
- Done: regional diesel pass-through model (error-correction, sourced; docs/regional_pass_through.md).
- Done: DCF design review across projects (docs/dcf_design_review.md); single-company dcf_v2 model; DCF fixes applied to dcf-core and bloomberg-terminal.
- Done: freight model rebuilt for sea shipping (8,000 TEU container ship, Shanghai to Rotterdam; examples/sea_freight_model.py; docs/freight_assumptions.md).

## Company workflow (phases 23 to 28)

- Done: company inputs from SEC filings with source labels, price reference line, generic company DCF (phase 23).
- Done: reverse DCF, market-implied drivers and their belief probability (phase 24).
- Done: segment revenue build carrying the beliefs; Netflix build reproduces the workbook's base, bull and bear (phase 25).
- Done: correlated drivers through a Gaussian copula, priors and historical shrinkage (phase 26).
- Done: calibration backtest against later filings, width sweep, basket and price check (phase 27).
- Done: `python -m nd_geometry.company` and an HTML report; docs/company_workflow.md (phase 28).
- Next: run `backtest_basket` on a 20 to 30 name basket (needs ND_SEC_USER_AGENT) and decide whether the default belief
  construction needs a trend term; on Netflix it is over-confident and biased low on margin.
- Next: a CAPM WACC from filings plus a market beta, so WACC is not a default.
- Next: segment-level history for builds (company facts carry no segments); today builds come from a spec file.

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