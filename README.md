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
- `filings.py`, `prices.py`, `dcf.py`: a company's DCF inputs from SEC filings, with the current price (see below).
- `revenue_build.py`, `correlation.py`, `reverse.py`, `backtest.py`, `company.py`: revenue builds, correlated beliefs,
  reverse DCF, calibration backtest, and the one-command workflow.

## Layout

```
src/nd_geometry/   the package
examples/          scripts, including visualize_6d.py and sensitivity_analysis.py
tests/             one test module per area
```

## Company workflow

```bash
export ND_SEC_USER_AGENT="Your Name you@example.com"
python -m nd_geometry.company NFLX --build examples/specs/nflx_revenue_build.json --out outputs/nflx_company.html
```

Details, method and the Netflix test results are in [docs/company_workflow.md](docs/company_workflow.md).

## Setup

```bash
pip install -e .
```

Run the examples from the repo root, for example `python examples/visualize_6d.py`. Tests live in `tests/`.

## Branches

Work is on `phase-*` branches, one per development phase. The newest is `phase-28-company-workflow` (phases 23 to 28 are stacked on one another).

## Roadmap

See [ROADMAP.md](ROADMAP.md).
