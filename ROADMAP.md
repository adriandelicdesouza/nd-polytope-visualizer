# Roadmap

## Done

- Polytopes, projections, slicing and rotations.
- Parameter spaces, evaluation and sensitivity.
- Response geometry and response level sets.
- 4D level-set topology, with staircase triangulation of simplex sections (latest work).

## Done (recent)

- Merged all phase branches into `master` (fast-forward from `phase-22-nd-level-set-topology`). The phase branches are now history.
- Added `plotly` as a runtime dependency; a clean install previously failed on import.
- Recorded expected output for the four examples in `examples/EXPECTED_OUTPUT.md`.

## Next

- Extend the 4D facet-cell work to 5D or higher, only if a real example needs it. The current examples run in 6D without it, so this is deferred.
