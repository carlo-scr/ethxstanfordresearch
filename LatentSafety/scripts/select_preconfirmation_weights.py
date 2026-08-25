#!/usr/bin/env python3
"""Freeze validation-selected E2 learned-arm weights from the complete pilot factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGGREGATION_ROOT = ROOT
CONFIG_PATH = ROOT / "configs" / "e2_frontier" / "intervention.toml"
IMPLEMENTATION_PATH = ROOT / "src" / "latent_safety" / "analysis" / "preconfirmation.py"
MATCHED_RADIUS_IMPLEMENTATION_PATH = (
    ROOT / "src" / "latent_safety" / "analysis" / "matched_radius.py"
)
ARTIFACT_IMPLEMENTATION_PATH = (
    ROOT / "src" / "latent_safety" / "analysis" / "preconfirmation_artifacts.py"
)
AGGREGATOR_IMPLEMENTATION_PATH = ROOT / "scripts" / "aggregate_e2_preconfirmation.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_e2_preconfirmation as AGGREGATOR_MODULE  # noqa: E402

from latent_safety.analysis.preconfirmation import (  # noqa: E402
    CONTROL_ARM,
    EXPECTED_OBSERVATIONS,
    PILOT_SEEDS,
    PreconfirmationObservation,
    run_preconfirmation_selection,
    validate_frozen_e2_intervention_protocol,
)
from latent_safety.analysis.matched_radius import (  # noqa: E402
    CONTROL_REFERENCE_RELATIVE_RADIUS,
    MAX_RELATIVE_MASS_MISMATCH,
    RELATIVE_RADIUS_GRID,
)
from latent_safety.analysis.preconfirmation_artifacts import (  # noqa: E402
    canonical_sha256,
)
from latent_safety.manifest import write_json_atomic  # noqa: E402


CANONICAL_AGGREGATE = "e2_preconfirmation_validation_observations_v1"


def _observations_from_payload(payload: object) -> tuple[PreconfirmationObservation, ...]:
    rows: Any = payload.get("observations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON list or an object with an observations list")
    observations: list[PreconfirmationObservation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"observations[{index}] must be an object")
        try:
            observations.append(PreconfirmationObservation(**row))
        except TypeError as error:
            raise ValueError(f"invalid observations[{index}] fields: {error}") from error
    return tuple(observations)


def _load_observations(path: Path) -> tuple[PreconfirmationObservation, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read preconfirmation input {path}: {error}") from error
    return _observations_from_payload(payload)


def _lower_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identity(row: dict[str, Any]) -> tuple[str, str, str, int, float | None]:
    return (
        str(row["domain"]),
        str(row["model_family"]),
        str(row["arm"]),
        int(row["seed"]),
        None if row["weight"] is None else float(row["weight"]),
    )


def _validate_radius_sidecars(
    payload: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    declared = _lower_sha256(payload.get("aggregate_sha256"), field="aggregate_sha256")
    unsigned = dict(payload)
    unsigned.pop("aggregate_sha256", None)
    if canonical_sha256(unsigned) != declared:
        raise ValueError("canonical preconfirmation aggregate self-hash mismatch")
    if payload.get("schema_version") != 1:
        raise ValueError("canonical aggregate schema_version must be exactly 1")
    scope = payload.get("scope")
    if not isinstance(scope, dict) or (
        scope.get("input_split") != "validation_only"
        or scope.get("calibration_access") is not False
        or scope.get("final_test_access") is not False
        or scope.get("radius_selection_used_safety") is not False
    ):
        raise ValueError("canonical aggregate violates validation-only selection scope")
    radius_protocol = payload.get("radius_protocol")
    if not isinstance(radius_protocol, dict) or radius_protocol != {
        "relative_radius_grid": list(RELATIVE_RADIUS_GRID),
        "control_reference_relative_radius": CONTROL_REFERENCE_RELATIVE_RADIUS,
        "maximum_relative_neighborhood_mass_mismatch": MAX_RELATIVE_MASS_MISMATCH,
    }:
        raise ValueError("canonical aggregate radius protocol drifted")
    source_plan = payload.get("source_plan")
    if not isinstance(source_plan, dict):
        raise ValueError("canonical aggregate source plan is missing")
    _lower_sha256(source_plan.get("sha256"), field="source_plan.sha256")
    _lower_sha256(
        source_plan.get("plan_sha256"), field="source_plan.plan_sha256"
    )
    sidecars = payload.get("row_provenance")
    if (
        not isinstance(sidecars, list)
        or len(rows) != EXPECTED_OBSERVATIONS
        or len(sidecars) != len(rows)
        or payload.get("row_provenance_count") != len(rows)
        or payload.get("observation_count") != len(rows)
    ):
        raise ValueError("canonical aggregate requires one radius sidecar per observation")
    row_lookup = {_identity(row): row for row in rows}
    if len(row_lookup) != len(rows):
        raise ValueError("canonical aggregate contains duplicate observation identities")
    sidecar_lookup: dict[
        tuple[str, str, str, int, float | None], dict[str, Any]
    ] = {}
    for index, (row, sidecar_value) in enumerate(zip(rows, sidecars, strict=True)):
        if not isinstance(sidecar_value, dict):
            raise ValueError(f"row_provenance[{index}] must be an object")
        sidecar = sidecar_value
        declared_freeze = _lower_sha256(
            sidecar.get("radius_freeze_sha256"),
            field=f"row_provenance[{index}].radius_freeze_sha256",
        )
        unsigned_freeze = dict(sidecar)
        unsigned_freeze.pop("radius_freeze_sha256", None)
        if canonical_sha256(unsigned_freeze) != declared_freeze:
            raise ValueError(f"row_provenance[{index}] self-hash mismatch")
        if sidecar.get("observation_sha256") != canonical_sha256(row):
            raise ValueError(f"row_provenance[{index}] does not bind its observation")
        key = _identity(row)
        sidecar_key = (
            sidecar.get("domain"),
            sidecar.get("model_family"),
            sidecar.get("arm"),
            sidecar.get("seed"),
            sidecar.get("weight"),
        )
        if sidecar_key != key:
            raise ValueError(f"row_provenance[{index}] identity mismatch")
        if sidecar.get("task_id") != index:
            raise ValueError("canonical aggregate task/row order must remain 0..287")
        if sidecar.get("radius_selection_used_safety") is not False:
            raise ValueError("radius selection must explicitly exclude safety values")
        if not math.isclose(
            float(sidecar.get("control_reference_relative_radius", -1.0)),
            CONTROL_REFERENCE_RELATIVE_RADIUS,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("radius sidecar control reference drifted")
        if sidecar.get("maximum_relative_neighborhood_mass_mismatch") != (
            MAX_RELATIVE_MASS_MISMATCH
        ):
            raise ValueError("radius sidecar mass-mismatch threshold drifted")
        _lower_sha256(sidecar.get("task_sha256"), field="sidecar.task_sha256")
        task_manifest = sidecar.get("task_manifest")
        if not isinstance(task_manifest, dict):
            raise ValueError("radius sidecar task manifest provenance is missing")
        _lower_sha256(task_manifest.get("sha256"), field="sidecar.task_manifest.sha256")
        _lower_sha256(
            task_manifest.get("manifest_sha256"),
            field="sidecar.task_manifest.manifest_sha256",
        )
        if not isinstance(task_manifest.get("path"), str) or not task_manifest["path"]:
            raise ValueError("radius sidecar task manifest path is missing")
        _lower_sha256(
            sidecar.get("control_audit_sha256"), field="sidecar.control_audit_sha256"
        )
        _lower_sha256(sidecar.get("pairing_sha256"), field="sidecar.pairing_sha256")
        if row["arm"] == CONTROL_ARM:
            if (
                sidecar.get("kind") != "none_control_reference"
                or sidecar.get("control_task_id") != index
                or sidecar.get("learned_relative_radius") is not None
                or sidecar.get("learned_audit_sha256") is not None
                or sidecar.get("mass_match_passed") is not None
                or sidecar.get("strict_safety_improvement") is not None
            ):
                raise ValueError("none-control radius sidecar schema mismatch")
        else:
            if sidecar.get("kind") != "learned_mass_matched":
                raise ValueError("learned row lacks a mass-matched radius sidecar")
            _lower_sha256(
                sidecar.get("learned_audit_sha256"), field="sidecar.learned_audit_sha256"
            )
            relative_radius = sidecar.get("learned_relative_radius")
            if relative_radius not in RELATIVE_RADIUS_GRID:
                raise ValueError("learned radius is absent from the frozen radius grid")
            control_key = (
                row["domain"],
                row["model_family"],
                CONTROL_ARM,
                row["seed"],
                None,
            )
            control = row_lookup.get(control_key)
            if control is None:
                raise ValueError("learned radius sidecar has no paired none observation")
            expected_control_sidecar = sidecar_lookup.get(control_key)
            if expected_control_sidecar is None or sidecar.get("control_task_id") != (
                expected_control_sidecar.get("task_id")
            ):
                raise ValueError("learned radius sidecar names the wrong none control")
            mismatch = (
                abs(float(row["neighborhood_mass"]) - float(control["neighborhood_mass"]))
                / float(control["neighborhood_mass"])
                if float(control["neighborhood_mass"]) > 0.0
                else None
            )
            declared_mismatch = sidecar.get("relative_neighborhood_mass_mismatch")
            if mismatch is None or not isinstance(declared_mismatch, (int, float)) or not math.isclose(
                float(declared_mismatch), mismatch, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("radius sidecar neighborhood-mass mismatch is inconsistent")
            if sidecar.get("mass_match_passed") is not (
                mismatch <= MAX_RELATIVE_MASS_MISMATCH
            ):
                raise ValueError("radius sidecar mass-match decision is inconsistent")
            strict = float(row["safety"]) < float(control["safety"])
            if sidecar.get("strict_safety_improvement") is not strict:
                raise ValueError("radius sidecar strict safety sign is inconsistent")
            control_sidecar = next(
                (
                    candidate
                    for candidate in sidecars
                    if isinstance(candidate, dict)
                    and candidate.get("task_id") == sidecar.get("control_task_id")
                ),
                None,
            )
            if not isinstance(control_sidecar, dict) or (
                control_sidecar.get("control_audit_sha256")
                != sidecar.get("control_audit_sha256")
                or control_sidecar.get("pairing_sha256") != sidecar.get("pairing_sha256")
                or control_sidecar.get("control_absolute_radius")
                != sidecar.get("control_absolute_radius")
            ):
                raise ValueError("learned radius sidecar disagrees with its none-control audit")
        if key in sidecar_lookup:
            raise ValueError("duplicate radius sidecar identity")
        sidecar_lookup[key] = sidecar
    return sidecars


def _selected_radius_freezes(
    result: Any,
    sidecars: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["domain"],
            row["model_family"],
            row["arm"],
            row["seed"],
            row["weight"],
        ): row
        for row in sidecars
    }
    freezes: list[dict[str, Any]] = []
    for selection in result.selections:
        if selection.selected_weight is None:
            continue
        per_seed = [
            lookup[
                (
                    selection.domain,
                    selection.model_family,
                    selection.arm,
                    seed,
                    selection.selected_weight,
                )
            ]
            for seed in PILOT_SEEDS
        ]
        freezes.append(
            {
                "domain": selection.domain,
                "model_family": selection.model_family,
                "arm": selection.arm,
                "selected_weight": selection.selected_weight,
                "seed_radius_freezes": per_seed,
            }
        )
    return freezes


def _reauthenticate_canonical_aggregate(payload: dict[str, Any]) -> None:
    """Rebuild from the pinned plan and require exact object equality before selection."""

    source_plan = payload.get("source_plan")
    if not isinstance(source_plan, dict):
        raise ValueError("canonical aggregate source plan is missing")
    declared = source_plan.get("path")
    if not isinstance(declared, str) or not declared or Path(declared).is_absolute():
        raise ValueError("canonical aggregate source plan path must be repository-relative")
    plan_path = (AGGREGATION_ROOT / declared).resolve()
    try:
        plan_path.relative_to(AGGREGATION_ROOT.resolve())
    except ValueError as error:
        raise ValueError("canonical aggregate source plan path escapes the repository") from error
    try:
        rebuilt = AGGREGATOR_MODULE.aggregate(plan_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"canonical aggregate source reauthentication failed: {error}"
        ) from error
    if rebuilt != payload:
        raise ValueError(
            "canonical aggregate does not exactly match reconstruction from its source "
            "plan and 288 authenticated task manifests"
        )


def select(input_path: Path) -> dict[str, Any]:
    raw = input_path.read_bytes()
    config_raw = CONFIG_PATH.read_bytes()
    try:
        protocol_config = tomllib.loads(config_raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read frozen E2 protocol config: {error}") from error
    try:
        validate_frozen_e2_intervention_protocol(protocol_config)
    except ValueError as error:
        raise ValueError(f"frozen E2 protocol config drifted: {error}") from error
    try:
        input_payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read preconfirmation input {input_path}: {error}") from error
    if (
        not isinstance(input_payload, dict)
        or input_payload.get("analysis") != CANONICAL_AGGREGATE
    ):
        raise ValueError(
            "file-backed weight selection requires the authenticated canonical 288-row "
            "preconfirmation aggregate with radius sidecars"
        )
    _reauthenticate_canonical_aggregate(input_payload)
    rows = input_payload.get("observations")
    if not isinstance(rows, list):
        raise ValueError("canonical aggregate observations must be a list")
    sidecars = _validate_radius_sidecars(input_payload, rows)
    observations = _observations_from_payload(input_payload)
    result = run_preconfirmation_selection(observations)
    selected_radius_freezes = _selected_radius_freezes(result, sidecars)
    if result.ready_for_confirmation and len(selected_radius_freezes) != 18:
        raise ValueError(
            "confirmation-ready selection must preserve all 18 selected radius freezes"
        )
    result_payload = result.to_dict()
    result_payload["selected_radius_freezes"] = selected_radius_freezes
    result_payload["selected_radius_freeze_count"] = len(selected_radius_freezes)
    return {
        "schema_version": 1,
        "analysis": "e2_preconfirmation_weight_freeze_v1",
        "protocol_config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(config_raw).hexdigest(),
        },
        "implementation": [
            {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            {
                "path": IMPLEMENTATION_PATH.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(IMPLEMENTATION_PATH.read_bytes()).hexdigest(),
            },
            {
                "path": MATCHED_RADIUS_IMPLEMENTATION_PATH.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    MATCHED_RADIUS_IMPLEMENTATION_PATH.read_bytes()
                ).hexdigest(),
            },
            {
                "path": ARTIFACT_IMPLEMENTATION_PATH.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    ARTIFACT_IMPLEMENTATION_PATH.read_bytes()
                ).hexdigest(),
            },
            {
                "path": AGGREGATOR_IMPLEMENTATION_PATH.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    AGGREGATOR_IMPLEMENTATION_PATH.read_bytes()
                ).hexdigest(),
            },
        ],
        "input": {
            "path": input_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "observation_count": len(observations),
            "split": "validation_only",
            "calibration_or_final_test_used": False,
        },
        "result": result_payload,
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
        payload = select(input_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    write_json_atomic(output_path, payload)
    result = payload["result"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "required_selections": result["required_selection_count"],
                "passed_selections": result["passed_selection_count"],
                "ready_for_confirmation": result["ready_for_confirmation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
