# DCF design review across projects

Reviewed: dcf-core, bloomberg-terminal (DCF endpoint), equity-research-platform (engine/dcf and docs/methodology/dcf.md), and this project's DCF examples. n-dimm-sens has no commits and was not reviewed.

Multidimensional sampling stays in this project. The other repos received only single-company DCF fixes.

## Design elements taken

- From equity-research-platform: cumulative per-period discounting, mid-period timing, tax only on positive EBIT, and refusing invalid inputs with a clear error rather than returning a number.
- From dcf-core: the driver-by-driver forecast structure (revenue, EBIT, D&A, capex, NWC, UFCF), kept as the simple reference.
- From bloomberg-terminal: NWC change on the revenue increment, the sensitivity grid over WACC and terminal growth, and the warnings for thin WACC-minus-growth spreads and high terminal-value shares.

## Built here: dcf_v2 (examples/dcf_v2_model.py)

- Mid-year discounting by default, with an end-of-year switch.
- Terminal value discounted with the same timing as the final-year flow.
- Refuses WACC at or below terminal growth.
- No tax credit on losses.
- Returns implied price, enterprise value and terminal-value share.
- Five tests in tests/test_dcf_v2.py.

Scope: a single company with a fixed base revenue, share count and net debt. It is not yet tied to real company data.

## Changes made in source repos

- dcf-core (branch v2, commit 808de5b): opening NWC is set on the same revenue share as year 1, so year 1 no longer pays the full balance as an outflow. NOPAT gives no tax credit on losses. One test that encoded the old year-1 value was corrected to 110.75. ROADMAP updated. 24 tests pass.
- bloomberg-terminal (main, commit f2f49c1): NOPAT taxes only positive EBIT. WACC at or below terminal growth now returns HTTP 400 instead of a zero terminal value. The DCF test file has 15 passing and 1 failing test; the failure is pre-existing and unchanged by this commit.

## Still open, by repo

- dcf-core: cli.py is empty even though the README describes a CLI. Its ROADMAP item 3 (share WACC logic with the terminal) is not done.
- bloomberg-terminal: the WACC is a CAPM rate with the cost of debt set to risk-free plus a spread, not the actual cost. Its DCF has no mid-year option and no stub periods. One test (test_older_5_year_revenue_growth_uses_years_6_to_10) fails before and after this change.
- equity-research-platform: no code changed. Its specification is the most complete reference here. Its explicit provenance on every WACC input and its validation report are the designs to copy into the other repos.
- n-dimm-sens: empty repository, no commits. Not reviewed.
