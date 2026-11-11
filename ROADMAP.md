# Roadmap

## Done

- Polytopes, projections, slicing and rotations.
- Parameter spaces, evaluation and sensitivity.
- Response geometry and response level sets.
- 4D level-set topology, with staircase triangulation of simplex sections (latest work).

## Done (recent)

- Freight assumptions checked against sources (`docs/freight_assumptions.md`). This roughly doubled the base-case cost per tonne, because non-fuel cost was about 2.5 times too low, and set driving hours to the legal 13 h.
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
