# Expected example output

Recorded from a clean install (`pip install -e ".[dev]"` in a fresh venv) by running each script from the repo root. Compare against these if an example's output changes.

## evaluate_model.py

```text
[0.02 0.1  0.06] -> 1.7000000000000002
[0.02 0.1  0.09] -> 1.1333333333333335
[0.02 0.1  0.12] -> 0.8500000000000001
[0.02  0.175 0.06 ] -> 2.975
[0.02  0.175 0.09 ] -> 1.9833333333333334
[0.02  0.175 0.12 ] -> 1.4875
[0.02 0.25 0.06] -> 4.25
[0.02 0.25 0.09] -> 2.8333333333333335
[0.02 0.25 0.12] -> 2.125
[0.06 0.1  0.06] -> 1.7666666666666668
[0.06 0.1  0.09] -> 1.177777777777778
[0.06 0.1  0.12] -> 0.8833333333333334
[0.06  0.175 0.06 ] -> 3.091666666666667
[0.06  0.175 0.09 ] -> 2.0611111111111113
[0.06  0.175 0.12 ] -> 1.5458333333333334
[0.06 0.25 0.06] -> 4.416666666666667
[0.06 0.25 0.09] -> 2.9444444444444446
[0.06 0.25 0.12] -> 2.2083333333333335
[0.1  0.1  0.06] -> 1.8333333333333337
[0.1  0.1  0.09] -> 1.2222222222222223
[0.1  0.1  0.12] -> 0.9166666666666669
[0.1   0.175 0.06 ] -> 3.2083333333333335
[0.1   0.175 0.09 ] -> 2.138888888888889
[0.1   0.175 0.12 ] -> 1.6041666666666667
[0.1  0.25 0.06] -> 4.583333333333334
[0.1  0.25 0.09] -> 3.055555555555556
[0.1  0.25 0.12] -> 2.291666666666667
```

## sensitivity_analysis.py

```text
Dimensions: 3
Shape: (5, 5, 5)
Minimum: 0.8500000000000001
Maximum: 4.583333333333334
```

## visualize_6d_linear.py

```text
Vertices: 32
Edges:    80
```

## visualize_6d.py

```text
Vertices: 8
Edges:    12
```

## dcf_output_polytope.py

```text
Parameters: ('revenue_growth', 'ebit_margin', 'wacc', 'terminal_growth')
Grid samples: 625
Output bounds (min, max):
  enterprise_value: 71.7159 to 522.4520
  terminal_share: 0.6016 to 0.8103
Polytope vertices: 16
Polytope area: 50.8689
Base case inside: True
```

## dcf_model.py

```text
enterprise_value: 71.72 to 522.45
terminal_share: 0.60 to 0.81
Base case inside envelope: True
```

## freight_oil_model.py

```text
```

## dcf_full_report.py

```text
Envelope volume by refinement round: [402662.1, 403192.3, 403275.4]
Base case inside envelope: True
DCF (ten inputs): 4560 samples (random+refined) across 10 inputs and 3 outputs.
enterprise_value ranges from -39.59 $m to 1,010.59 $m; the base case of 272.23 $m sits in the middle of that range.
  gross_margin moves enterprise_value most, accounting for 27% of the one-at-a-time swing.
pv_terminal_value ranges from -40.61 $m to 823.25 $m; the base case of 191.32 $m sits in the middle of that range.
  gross_margin moves pv_terminal_value most, accounting for 24% of the one-at-a-time swing.
year5_free_cash_flow ranges from -2.90 $m to 58.72 $m; the base case of 24.17 $m sits in the middle of that range.
  gross_margin moves year5_free_cash_flow most, accounting for 34% of the one-at-a-time swing.
The base case lies inside the output envelope.
Influence is measured one input at a time, so interactions between inputs are not shown.
Baseline check against itself: {'ok': True, 'changes': []}
```

## freight_oil_report.py

```text
Event frequencies: {'port_delay_days': 0.253, 'fuel_shock': 0.15}
cost_per_tonne: P10 15.85, P50 42.03, P90 297.18
transit_days: P10 0.55, P50 1.34, P90 7.58
cost_per_tonne by Brent price ($/bbl):
    50.0: P10 14.08, P50 37.33, P90 294.08
    67.5: P10 14.97, P50 39.66, P90 297.26
    85.0: P10 15.83, P50 41.90, P90 298.55
   102.5: P10 16.71, P50 44.22, P90 300.55
   120.0: P10 17.59, P50 46.53, P90 302.60
Dashboard written.
```
