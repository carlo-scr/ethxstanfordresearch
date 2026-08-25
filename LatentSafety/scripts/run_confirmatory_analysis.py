#!/usr/bin/env python3
"""Run the frozen 36-bound E2 inference contract on a complete JSON observation file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.confirmatory import (  # noqa: E402
    ConfirmatoryObservation,
    run_confirmatory_inference,
)
from latent_safety.manifest import write_json_atomic  # noqa: E402


def _load_observations(path: Path) -> tuple[ConfirmatoryObservation, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read confirmatory input {path}: {error}") from error
    rows: Any = payload.get("observations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON list or an object with an observations list")
    observations: list[ConfirmatoryObservation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"observations[{index}] must be an object")
        try:
            observations.append(ConfirmatoryObservation(**row))
        except TypeError as error:
            raise ValueError(f"invalid observations[{index}] fields: {error}") from error
    return tuple(observations)


def analyze(input_path: Path) -> dict[str, Any]:
    raw = input_path.read_bytes()
    observations = _load_observations(input_path)
    result = run_confirmatory_inference(observations)
    return {
        "schema_version": 1,
        "analysis": "e2_confirmatory_36_bound_v1",
        "input": {
            "path": input_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "observation_count": len(observations),
        },
        "result": result.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    try:
        payload = analyze(input_path)
    except ValueError as error:
        parser.error(str(error))
    write_json_atomic(output_path, payload)
    result = payload["result"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "bounds": len(result["bounds"]),
                "passed_bounds": result["passed_bound_count"],
                "confirmatory_success": result["confirmatory_success"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
