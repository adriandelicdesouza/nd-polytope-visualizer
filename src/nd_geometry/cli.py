"""Command line: analyse a model spec, run a scenario, or slice an output surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nd_geometry.analysis import analyse
from nd_geometry.explain import explain
from nd_geometry.spec import load_model
from nd_geometry.views import input_surface


def _parse_sets(items: list[str]) -> dict[str, float]:
    values = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"bad value {item!r}; use name=value")
        values[name] = float(value)
    return values


def _check_names(model, names: dict[str, float]) -> None:
    unknown = sorted(set(names) - set(model.input_names))
    if unknown:
        raise SystemExit(f"unknown inputs {unknown}; model inputs are {list(model.input_names)}")


def cmd_analyse(args: argparse.Namespace) -> int:
    model = load_model(args.spec)
    result = analyse(model, method=args.method, samples=args.samples, seed=args.seed,
                     points_per_dimension=args.points)
    lines = explain(result)
    print("\n".join(lines))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.save(out / "analysis.json")
        (out / "explain.txt").write_text("\n".join(lines) + "\n")
        print(f"Saved to {out}")
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    model = load_model(args.spec)
    overrides = _parse_sets(args.set)
    _check_names(model, overrides)
    base_inputs = model.base_case()
    base = model.evaluate(base_inputs)
    scenario = model.evaluate({**base_inputs, **overrides})
    print(f"{'output':<18}{'base':>14}{'scenario':>14}{'change':>14}")
    for name in model.output_names:
        change = scenario[name] - base[name]
        print(f"{name:<18}{base[name]:>14.4f}{scenario[name]:>14.4f}{change:>+14.4f}")
    return 0


def cmd_surface(args: argparse.Namespace) -> int:
    model = load_model(args.spec)
    fixed = _parse_sets(args.fix)
    _check_names(model, {**fixed, args.x: 0.0, args.y: 0.0})
    result = analyse(model, method="random", samples=args.samples, seed=args.seed)
    surface = input_surface(result, args.x, args.y, args.output, fixed=fixed, points=args.points)
    z = surface["z"]
    print(f"{args.output} over {args.x} (columns) and {args.y} (rows): min {min(map(min, z)):.4f}, "
          f"max {max(map(max, z)):.4f}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        payload = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in surface.items()}
        (out / f"surface_{args.output}.json").write_text(json.dumps(payload, indent=2))
        print(f"Saved to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nd_geometry", description="N-dimensional model analysis from a spec.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("analyse", help="output polytope, envelope and driver influence")
    run.add_argument("spec")
    run.add_argument("--method", default="random", choices=["random", "grid"])
    run.add_argument("--samples", type=int, default=2000)
    run.add_argument("--points", type=int, default=5, help="grid points per input (grid method)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", help="directory for analysis.json and explain.txt")
    run.set_defaults(func=cmd_analyse)

    scenario = sub.add_parser("scenario", help="outputs for overridden inputs, against base")
    scenario.add_argument("spec")
    scenario.add_argument("--set", action="append", default=[], help="name=value, repeatable")
    scenario.set_defaults(func=cmd_scenario)

    surface = sub.add_parser("surface", help="one output over two inputs, others fixed")
    surface.add_argument("spec")
    surface.add_argument("--x", required=True)
    surface.add_argument("--y", required=True)
    surface.add_argument("--output", required=True)
    surface.add_argument("--fix", action="append", default=[], help="name=value for other inputs, repeatable")
    surface.add_argument("--points", type=int, default=15)
    surface.add_argument("--samples", type=int, default=300)
    surface.add_argument("--seed", type=int, default=0)
    surface.add_argument("--out", help="directory for surface JSON")
    surface.set_defaults(func=cmd_surface)

    args = parser.parse_args(argv)
    return args.func(args)
