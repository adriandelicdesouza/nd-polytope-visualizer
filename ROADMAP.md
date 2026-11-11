# Roadmap

## Done

- Polytopes, projections, slicing and rotations.
- Parameter spaces, evaluation and sensitivity.
- Response geometry and response level sets.
- 4D level-set topology, with staircase triangulation of simplex sections (latest work).

## Done (recent)

- Model template (`model.py`) and analysis runner (`analysis.py`), grid or seeded random with box corners.
- Boundary refinement (`refine.py`): jitters samples behind the envelope and stops when its volume settles.
- Explanations (`explain.py`): plain-language summary and ranked one-at-a-time input drivers.
- Views (`views.py`): input surfaces over any two inputs with the rest fixed; envelope projections; standalone HTML via plotly.
- Regression baseline (`baseline.py`): compares a saved summary and flags output or input changes.
- Ten-input DCF and end-to-end report (`examples/dcf_full_report.py`); outputs go to `outputs/` (git-ignored).

## Next

1. Slice navigation: choose which axes to show and fix the rest from the command line or a small UI.
2. Freight/oil model with vessel size, load, fuel, route and disruption inputs, to test the design on a second domain.
3. Non-monotone models: check that refinement finds interior extremes on real models, not only the test case.
4. Deferred until a real model needs them: input constraints (e.g. portfolio weights) and stochastic models (fixed seeds or confidence bands).
5. Extend the 4D facet-cell work to 5D+ only if a real example needs it.
