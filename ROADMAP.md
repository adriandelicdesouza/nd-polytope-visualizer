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

## Next

- Extend the 4D facet-cell work to 5D or higher, only if a real example needs it. Deferred.
- Output polytopes are the convex hull of the grid outputs. For curved models this is an inner approximation, so consider a denser grid or a refinement step near the boundary.
- Input constraints (e.g. weights summing to 1) and stochastic models are deferred until they are needed for integration.
