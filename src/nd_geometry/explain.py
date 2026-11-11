from __future__ import annotations

from .analysis import AnalysisResult


def input_influence(result: AnalysisResult) -> dict[str, list[tuple[str, float, float]]]:
    """Rank inputs by how much each output swings when that input moves low to high.

    Each input is varied alone, with the others held at their base values, so this
    captures direct effects only, not interactions. Returns, per output, a list of
    (input name, swing, share of total swing) sorted from most to least influential.
    """
    model = result.model
    base = model.base_case()
    swings: dict[str, dict[str, float]] = {name: {} for name in model.output_names}

    for item in model.inputs:
        low_case = model.evaluate({**base, item.name: item.low})
        high_case = model.evaluate({**base, item.name: item.high})
        for name in model.output_names:
            swings[name][item.name] = abs(high_case[name] - low_case[name])

    ranking: dict[str, list[tuple[str, float, float]]] = {}
    for name, per_input in swings.items():
        total = sum(per_input.values())
        ordered = sorted(per_input.items(), key=lambda pair: pair[1], reverse=True)
        ranking[name] = [
            (input_name, swing, swing / total if total > 0 else 0.0)
            for input_name, swing in ordered
        ]
    return ranking


def _format(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value * 100:,.1f}%"
    return f"{value:,.2f} {unit}"


def _position_words(position: float) -> str:
    if position < 0.25:
        return "near the low end"
    if position > 0.75:
        return "near the high end"
    return "in the middle"


def explain(result: AnalysisResult) -> list[str]:
    """Return a plain-language summary of an analysis, one sentence per line."""
    model = result.model
    bounds = result.output_bounds
    positions = result.base_position()
    influence = input_influence(result)
    units = {item.name: item.unit for item in model.outputs}

    lines = [
        f"{model.name}: {len(result.outputs)} samples ({result.method}) across "
        f"{len(model.inputs)} inputs and {len(model.outputs)} outputs."
    ]

    for item in model.outputs:
        low, high = bounds[item.name]
        base_value = result.base_outputs[item.name]
        unit = units[item.name]
        lines.append(
            f"{item.name} ranges from {_format(low, unit)} to {_format(high, unit)}; "
            f"the base case of {_format(base_value, unit)} "
            f"sits {_position_words(positions[item.name])} of that range."
        )
        top_name, _, top_share = influence[item.name][0]
        lines.append(
            f"  {top_name} moves {item.name} most, accounting for "
            f"{top_share:.0%} of the one-at-a-time swing."
        )

    if result.base_inside:
        lines.append("The base case lies inside the output envelope.")
    else:
        lines.append("Warning: the base case lies outside the output envelope.")

    lines.append(
        "Influence is measured one input at a time, so interactions between inputs are not shown."
    )
    return lines
