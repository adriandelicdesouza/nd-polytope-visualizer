from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from .analysis import AnalysisResult, evaluate_rows
from .output_polytope import output_polytope


def refine(
    result: AnalysisResult,
    rounds: int = 4,
    candidates_per_point: int = 8,
    tolerance: float = 1e-3,
    seed: int = 0,
) -> tuple[AnalysisResult, list[float]]:
    """Add samples near the envelope's boundary until its volume stops changing.

    Each round jitters the inputs behind the current hull vertices, evaluates the
    model there, and rebuilds the hull. The jitter shrinks each round. Returns the
    refined result and the volume after each round (starting with the original).
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    model = result.model
    lows = np.array([item.low for item in model.inputs])
    highs = np.array([item.high for item in model.inputs])
    span = highs - lows

    rng = np.random.default_rng(seed)
    inputs = result.inputs.copy()
    outputs = result.outputs.copy()
    polytope = result.polytope
    volumes = [polytope.volume]
    step = 0.1

    for _ in range(rounds):
        hull = ConvexHull(outputs)
        boundary = inputs[hull.vertices]
        base = np.repeat(boundary, candidates_per_point, axis=0)
        offsets = rng.uniform(-1.0, 1.0, size=base.shape) * span * step
        new_inputs = np.clip(base + offsets, lows, highs)

        inputs = np.vstack([inputs, new_inputs])
        outputs = np.vstack([outputs, evaluate_rows(model, new_inputs)])
        polytope = output_polytope(outputs, output_names=model.output_names)
        volumes.append(polytope.volume)

        step *= 0.5
        if abs(volumes[-1] - volumes[-2]) <= tolerance * volumes[-2]:
            break

    base_outputs = model.evaluate(model.base_case())
    base_point = np.array([base_outputs[name] for name in model.output_names])

    refined = AnalysisResult(
        model=model,
        method=f"{result.method}+refined",
        inputs=inputs,
        outputs=outputs,
        polytope=polytope,
        base_outputs=base_outputs,
        base_inside=polytope.contains(base_point),
        points_per_dimension=result.points_per_dimension,
    )
    return refined, volumes
