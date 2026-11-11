"""Load a model from a JSON spec, so assumptions can be changed without editing Python."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from nd_geometry.model import Input, Model, Output


def load_model(spec_path: str | Path) -> Model:
    """Build a Model from a spec file.

    Spec keys: name, function ("module:attribute"), sys_path (relative to the spec file),
    inputs (name, unit, low, high, base) and outputs (name, unit).
    """
    path = Path(spec_path).resolve()
    spec = json.loads(path.read_text())
    for entry in spec.get("sys_path", []):
        location = str((path.parent / entry).resolve())
        if location not in sys.path:
            sys.path.insert(0, location)
    module_name, _, attribute = spec["function"].partition(":")
    if not attribute:
        raise ValueError("function must look like 'module:attribute'")
    function = getattr(importlib.import_module(module_name), attribute)
    inputs = tuple(
        Input(item["name"], item.get("unit", ""), float(item["low"]), float(item["high"]), float(item["base"]))
        for item in spec["inputs"]
    )
    outputs = tuple(Output(item["name"], item.get("unit", "")) for item in spec["outputs"])
    return Model(name=spec["name"], inputs=inputs, outputs=outputs, function=function)
