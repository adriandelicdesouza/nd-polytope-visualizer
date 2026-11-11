from __future__ import annotations

import json
from pathlib import Path

from .analysis import AnalysisResult


def compare_to_baseline(
    result: AnalysisResult,
    baseline_path: str | Path,
    rel_tolerance: float = 0.01,
) -> dict:
    """Compare an analysis with a saved baseline summary (written by `result.save`).

    A change is flagged when an output's low, high or base value, or an input's
    bounds or base value, moves by more than `rel_tolerance` of the baseline value.
    Returns {"ok": bool, "changes": [...]}.
    """
    baseline = json.loads(Path(baseline_path).read_text())
    current = result.to_dict()
    changes: list[dict] = []

    if baseline["model"] != current["model"]:
        changes.append({"item": "model", "baseline": baseline["model"], "current": current["model"]})

    def check(kind: str, name: str, ref_entry: dict, cur_entry: dict, fields: tuple[str, ...]) -> None:
        for field in fields:
            reference = float(ref_entry[field])
            value = float(cur_entry[field])
            scale = max(abs(reference), 1e-12)
            if abs(value - reference) > rel_tolerance * scale:
                changes.append({
                    "item": f"{kind} {name}.{field}",
                    "baseline": reference,
                    "current": value,
                    "relative_change": (value - reference) / scale,
                })

    base_inputs = {entry["name"]: entry for entry in baseline["inputs"]}
    cur_inputs = {entry["name"]: entry for entry in current["inputs"]}
    if set(base_inputs) != set(cur_inputs):
        changes.append({"item": "input names", "baseline": sorted(base_inputs), "current": sorted(cur_inputs)})
    else:
        for name in cur_inputs:
            check("input", name, base_inputs[name], cur_inputs[name], ("low", "high", "base"))

    base_outputs = {entry["name"]: entry for entry in baseline["outputs"]}
    cur_outputs = {entry["name"]: entry for entry in current["outputs"]}
    if set(base_outputs) != set(cur_outputs):
        changes.append({"item": "output names", "baseline": sorted(base_outputs), "current": sorted(cur_outputs)})
    else:
        for name in cur_outputs:
            check("output", name, base_outputs[name], cur_outputs[name], ("low", "high", "base"))

    return {"ok": not changes, "changes": changes}
