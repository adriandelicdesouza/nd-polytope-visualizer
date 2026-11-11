from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShockHistory:
    """Share of time a price shock is active, and the sizes it took when active."""

    probability: float
    severities: tuple[float, ...]
    observations: int


def price_shock_history(
    path: str | Path,
    value_column: str,
    since: str = "2015-01-01",
    lag: int = 4,
    threshold: float = 0.05,
) -> ShockHistory:
    """Estimate how often a price rises by `threshold` or more over `lag` periods.

    A period is in shock when its price is at least `threshold` above the price
    `lag` periods earlier. Probability is the share of periods in shock since `since`;
    severities are the rises in those periods. Rows must be dated and sorted ascending.
    """
    rows = [
        (row["observation_date"], float(row[value_column]))
        for row in csv.DictReader(Path(path).open())
        if row[value_column] not in ("", ".")
    ]
    rows = [row for row in rows if row[0] >= since]

    rises = []
    observations = 0
    for index in range(lag, len(rows)):
        observations += 1
        change = rows[index][1] / rows[index - lag][1] - 1.0
        if change >= threshold:
            rises.append(change)

    if observations == 0:
        raise ValueError("not enough observations after `since` for this lag")

    return ShockHistory(
        probability=len(rises) / observations,
        severities=tuple(float(value) for value in np.round(rises, 4)),
        observations=observations,
    )
