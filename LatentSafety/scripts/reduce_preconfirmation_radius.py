#!/usr/bin/env python3
"""Recompute one diagnostic E2 matched-radius result from paired validation JSONL records.

Production selection does not trust this standalone artifact. The canonical E2 aggregator reopens
checksum-linked task manifests and recomputes both radius curves from pinned post-fit records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "e2_frontier" / "intervention.toml"
IMPLEMENTATION_PATH = (
    ROOT / "src" / "latent_safety" / "analysis" / "matched_radius.py"
)
RECORDS_PATH = ROOT / "src" / "latent_safety" / "records.py"
DEFECT_METRICS_PATH = ROOT / "src" / "latent_safety" / "metrics" / "defect.py"
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.matched_radius import (  # noqa: E402
    CONTROL_REFERENCE_RELATIVE_RADIUS,
    MATCHED_RADIUS_PROTOCOL,
    MATCHED_RADIUS_SCHEMA_VERSION,
    MATCHED_RADIUS_SEMANTICS_VERSION,
    MAX_SCALE_POINTS,
    MAX_RELATIVE_MASS_MISMATCH,
    RELATIVE_RADIUS_GRID,
    TAIL_QUANTILE,
    build_validation_radius_audit,
    match_validation_radius_audits,
    validate_paired_records,
)
from latent_safety.manifest import write_json_atomic  # noqa: E402
from latent_safety.records import read_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_config() -> None:
    try:
        with CONFIG_PATH.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read frozen matched-radius config: {error}") from error
    table = config.get("matched_radius_reduction")
    if not isinstance(table, dict):
        raise ValueError("config is missing [matched_radius_reduction]")
    expected = {
        "entrypoint": "scripts/reduce_preconfirmation_radius.py",
        "input_split": "validation_only",
        "relative_radius_grid": list(RELATIVE_RADIUS_GRID),
        "control_reference_relative_radius": CONTROL_REFERENCE_RELATIVE_RADIUS,
        "max_scale_points": MAX_SCALE_POINTS,
        "trajectory_tail_quantile": TAIL_QUANTILE,
        "radius_selection_uses_safety": False,
        "max_relative_mass_mismatch": MAX_RELATIVE_MASS_MISMATCH,
        "calibration_access": False,
        "final_test_access": False,
    }
    drift = {
        field: (table.get(field), expected_value)
        for field, expected_value in expected.items()
        if table.get(field) != expected_value
    }
    if drift:
        raise ValueError(f"frozen matched-radius config does not match implementation: {drift!r}")


def _validate_cli_audit(audit) -> None:
    radii = tuple(point.relative_radius for point in audit.curve)
    if radii != RELATIVE_RADIUS_GRID:
        raise ValueError(
            "validation radius audit does not use the prospectively frozen radius grid"
        )
    if audit.scale_used_points != min(audit.record_count, MAX_SCALE_POINTS):
        raise ValueError(
            "validation radius audit does not use the frozen deterministic scale subset size"
        )


def reduce_inputs(
    *,
    control_records_path: Path | None = None,
    learned_records_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute from one paired record set and return a diagnostic checksum envelope."""

    _validate_frozen_config()
    if control_records_path is None or learned_records_path is None:
        raise ValueError("record mode requires both control and learned record files")
    control_path = control_records_path.resolve()
    learned_path = learned_records_path.resolve()
    control_records, learned_records = validate_paired_records(
        read_jsonl(control_path), read_jsonl(learned_path)
    )
    control_audit = build_validation_radius_audit(
        control_records, relative_radii=RELATIVE_RADIUS_GRID
    )
    learned_audit = build_validation_radius_audit(
        learned_records, relative_radii=RELATIVE_RADIUS_GRID
    )

    _validate_cli_audit(control_audit)
    _validate_cli_audit(learned_audit)

    result = match_validation_radius_audits(
        control_audit,
        learned_audit,
        control_reference_relative_radius=CONTROL_REFERENCE_RELATIVE_RADIUS,
        max_relative_mass_mismatch=MAX_RELATIVE_MASS_MISMATCH,
    )
    implementation_paths = (
        Path(__file__).resolve(),
        IMPLEMENTATION_PATH,
        RECORDS_PATH,
        DEFECT_METRICS_PATH,
    )
    return {
        "schema_version": MATCHED_RADIUS_SCHEMA_VERSION,
        "analysis": MATCHED_RADIUS_PROTOCOL,
        "semantics_version": MATCHED_RADIUS_SEMANTICS_VERSION,
        "evidence_eligible": False,
        "evidence_boundary": (
            "diagnostic replay only; production selection must use the canonical aggregator, "
            "which rechecks task-manifest integrity and recomputes these curves"
        ),
        "protocol_config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(CONFIG_PATH),
            "relative_radius_grid": list(RELATIVE_RADIUS_GRID),
            "control_reference_relative_radius": CONTROL_REFERENCE_RELATIVE_RADIUS,
            "max_scale_points": MAX_SCALE_POINTS,
            "trajectory_tail_quantile": TAIL_QUANTILE,
            "maximum_relative_neighborhood_mass_mismatch": (
                MAX_RELATIVE_MASS_MISMATCH
            ),
        },
        "implementation": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for path in implementation_paths
        ],
        "input": {
            "mode": "paired_validation_record_files",
            "split": "validation_only",
            "calibration_access": False,
            "final_test_access": False,
            "control": {
                "path": control_path.name,
                "sha256": _sha256(control_path),
                "radius_audit_sha256": control_audit.audit_sha256,
            },
            "learned": {
                "path": learned_path.name,
                "sha256": _sha256(learned_path),
                "radius_audit_sha256": learned_audit.audit_sha256,
            },
        },
        "radius_audits": {
            "control": control_audit.to_dict(),
            "learned": learned_audit.to_dict(),
        },
        "result": result.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-records", type=Path)
    parser.add_argument("--learned-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    try:
        payload = reduce_inputs(
            control_records_path=args.control_records,
            learned_records_path=args.learned_records,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    write_json_atomic(output_path, payload)
    result = payload["result"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "selector_metrics_ready": result["selector_metrics_ready"],
                "mass_match_passed": result["mass_match_passed"],
                "strict_safety_improvement": result["strict_safety_improvement"],
                "all_matched_radius_gates_passed": result[
                    "all_matched_radius_gates_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
