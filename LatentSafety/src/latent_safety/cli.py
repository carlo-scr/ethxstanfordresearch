"""Command-line entry points for reproducible protocol smoke tests."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any

from latent_safety.manifest import base_manifest, write_json_atomic
from latent_safety.metrics.defect import empirical_defect_curve
from latent_safety.synthetic import make_aliasing_fixture


def _pilot_smoke(config_path: Path, output_path: Path) -> int:
    with config_path.open("rb") as stream:
        config: dict[str, Any] = tomllib.load(stream)

    fixture = make_aliasing_fixture(
        n_pairs=int(config["fixture"]["n_pairs"]),
        max_margin=float(config["fixture"]["max_margin"]),
        seed=int(config["run"]["seed"]),
    )
    deltas = tuple(float(value) for value in config["metric"]["deltas"])
    quantile = float(config["metric"]["quantile"])
    faithful = empirical_defect_curve(
        fixture.faithful_latents,
        fixture.margins,
        deltas=deltas,
        quantile=quantile,
    )
    collapsed = empirical_defect_curve(
        fixture.collapsed_latents,
        fixture.margins,
        deltas=deltas,
        quantile=quantile,
    )

    expected_margin = float(config["fixture"]["max_margin"])
    checks = {
        "faithful_exact_defect_is_zero": faithful[0].witness_margin == 0.0,
        "collapsed_exact_defect_is_known_margin": abs(
            collapsed[0].witness_margin - expected_margin
        )
        < 1e-12,
        "curve_is_monotone": all(
            left.witness_margin <= right.witness_margin
            for curve in (faithful, collapsed)
            for left, right in zip(curve, curve[1:])
        ),
    }
    repo_root = Path(__file__).resolve().parents[2]
    payload = base_manifest(repo_root=repo_root, config_path=config_path)
    payload.update(
        {
            "experiment": config["experiment"],
            "checks": checks,
            "metrics": {
                "faithful": [dataclasses.asdict(item) for item in faithful],
                "collapsed": [dataclasses.asdict(item) for item in collapsed],
            },
        }
    )
    write_json_atomic(output_path, payload)
    print(json.dumps({"output": str(output_path), "checks": checks}, sort_keys=True))
    return 0 if all(checks.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latent-safety")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("pilot-smoke", help="run deterministic estimator controls")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command == "pilot-smoke":
        return _pilot_smoke(arguments.config.resolve(), arguments.output.resolve())
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())

