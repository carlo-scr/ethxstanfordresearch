#!/usr/bin/env python3
"""Run the exhaustive registered finite-tree oracle on both controlled domains."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.benchmarks.dynamic_oracle import (  # noqa: E402
    audit_controlled_dynamic_oracle,
)
from latent_safety.learning import load_config  # noqa: E402

CONFIG_PATHS = (
    ROOT / "configs/e1_world_models/torch_pilot.toml",
    ROOT / "configs/e1_world_models/torch_pendulum_pilot.toml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(3,),
        default=3,
        help="Registered finite horizon (currently fixed at 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON path. Existing files are never overwritten.",
    )
    args = parser.parse_args()

    reports = [
        audit_controlled_dynamic_oracle(
            load_config(config_path).data, horizon=args.horizon
        )
        for config_path in CONFIG_PATHS
    ]
    payload = {
        "schema_version": 1,
        "status": "finite_nominal_theorem_oracle_not_empirical_evidence",
        "reports": [dataclasses.asdict(report) for report in reports],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
        return

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
