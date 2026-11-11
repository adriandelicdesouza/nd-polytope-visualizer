# Roadmap

## Done

- Polytopes, projections, slicing and rotations.
- Parameter spaces, evaluation and sensitivity.
- Response geometry and response level sets.
- 4D level-set topology, with staircase triangulation of simplex sections (latest work).

## Done (recent)

- Model template, analysis runner, boundary refinement, explanations, views, regression baseline (see earlier entries).
- Probability layer (`probability.py`): risk events with probabilities and severity ranges, Monte Carlo draws with a fixed seed, joint outcome distributions, quantiles, and quantile profiles across an input.
- Oil-linked road freight model (`examples/freight_oil_model.py`): Brent price plus diesel crack, vessel capacity, load, distance, speed, port delays and fuel shocks.
- Interactive dashboard (`dashboard.py`): self-contained HTML in the terminal's palette; choose any two variables for a joint density, filter inputs, filter by event, and trace quantiles across an input.

## Next

1. Validate the cost assumptions (delay cost per day, driving hours, fuel use) against sourced data before the pitch; the probabilities are placeholders.
2. Correlated risks: events are currently independent. Add a dependence structure (e.g. a shared shock) when the risks are linked in practice.
3. Analytic check: compare Monte Carlo quantiles with a closed-form result for a simple case.
4. Slice navigation and sharing: fixed views for the pitch, and a way to rerun the dashboard from a saved model.
5. Input constraints (e.g. portfolio weights) when a portfolio model is added.
6. Extend the 4D facet-cell work to 5D+ only if a real example needs it.
