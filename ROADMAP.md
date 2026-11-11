# Roadmap

## Done

- Polytopes, projections, slicing and rotations.
- Parameter spaces, evaluation and sensitivity.
- Response geometry and response level sets.
- 4D level-set topology, with staircase triangulation of simplex sections (latest work).

## Done (recent)

- Merged all phase branches into `master` (fast-forward from `phase-22-nd-level-set-topology`).
- Added `plotly` as a runtime dependency; a clean install previously failed on import.
- Recorded expected output for the examples in `examples/EXPECTED_OUTPUT.md`.
- Vector-output evaluation (`evaluate_vector_model`) and output polytopes (`output_polytope`, using scipy's convex hull), with a DCF surface example (`examples/dcf_output_polytope.py`).

## Done (recent)

- Model template (`model.py`): named inputs with units, ranges and base values; named outputs; `Model.evaluate` validates inputs and outputs.
- Analysis runner (`analysis.py`): `analyse(model)` samples the grid, builds the output envelope, reports output ranges and whether the base case sits inside, and saves a JSON summary.
- Two models run through the same runner: DCF and road freight (`examples/dcf_model.py`, `examples/freight_model.py`).

## Next

1. Explanation layer: plain-language summary, and a ranking of which inputs drive each output.
2. Multi-dimensional views: 2D and 3D slices and projections of the envelope, with chosen axes.
3. Boundary refinement, so the envelope is accurate for curved models before results are used for decisions.
4. Regression baseline: compare a saved summary against a later run to flag envelope changes.
5. Deferred until a real model needs them: input constraints (e.g. portfolio weights) and stochastic models.
6. Extend the 4D facet-cell work to 5D+ only if a real example needs it.
