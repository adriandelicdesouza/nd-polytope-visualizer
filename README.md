# nd-polytope-visualizer

Geometry tools for studying how model outputs change across an n-dimensional parameter space. The package works with polytopes, projections, slices, rotations and level sets, and it renders them.

## What it covers

- `polytopes.py`: polytope construction.
- `parameter_space.py`: the parameter space that is sampled or sliced.
- `projection.py`, `slicing.py`, `rotations.py`: reducing dimension for display.
- `evaluation.py`, `response_geometry.py`: evaluating a model over the space and the geometry of its response.
- `sensitivity.py`: sensitivity cells and their boundaries.
- `topology.py`: level-set topology, including 4D facet cells.
- `rendering.py`: output.

## Layout

```
src/nd_geometry/   the package
examples/          scripts, including visualize_6d.py and sensitivity_analysis.py
tests/             one test module per area
```

## Setup

```bash
pip install -e .
```

Run the examples from the repo root, for example `python examples/visualize_6d.py`. Tests live in `tests/`.

## Branches

Work is on `phase-*` branches, one per development phase. The newest is `phase-22-nd-level-set-topology`.

## Roadmap

See [ROADMAP.md](ROADMAP.md).
