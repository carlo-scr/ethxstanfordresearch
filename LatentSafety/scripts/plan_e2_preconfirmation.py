#!/usr/bin/env python3
"""Build the canonical 288-task validation-only E2 preconfirmation plan."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.preconfirmation import (  # noqa: E402
    CONTROL_ARM,
    DOMAINS,
    EXPECTED_OBSERVATIONS,
    LEARNED_ARMS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    POSITIVE_WEIGHTS,
    PROFILE_ARM,
    validate_frozen_e2_intervention_protocol,
)
from latent_safety.learning.config import load_config, validate_config  # noqa: E402
from latent_safety.manifest import config_sha256, write_json_atomic  # noqa: E402


INTERVENTION_CONFIG = Path("configs/e2_frontier/intervention.toml")
OUTPUT_ROOT = Path("runs/e2_frontier/preconfirmation_288")
PROFILE_TEACHER_PLAN = Path("runs/e2_frontier/profile_teacher_plan.json")
HISTORY_MODE = "stack_h4"
HISTORY_ENCODER = "stack"
HISTORY_LENGTH = 4
LATENT_DIM = 8
FCSRL_HEAD_HIDDEN_DIM = 64
READY_PLAN_STATUS = "indexed_validation_only_execution_ready"

BASE_CONFIGS: dict[str, str] = {
    "controlled_cart_video": "configs/e1_world_models/torch_pilot.toml",
    "controlled_pendulum_video": (
        "configs/e1_world_models/torch_pendulum_pilot.toml"
    ),
    "controlled_dubins_navigation_pixels": (
        "configs/e1_world_models/torch_dubins_pilot.toml"
    ),
}


def canonical_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_sha256(task: dict[str, Any]) -> str:
    unsigned = dict(task)
    unsigned.pop("task_sha256", None)
    return canonical_sha256(unsigned)


def _repo_relative(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside the repository: {resolved}") from error
    if relative == Path("."):
        raise ValueError(f"{label} must name a repository file")
    return relative


def _git_code_state() -> dict[str, str | bool | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _table(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"intervention [{name}] must be a TOML table")
    return value


def _exact_list(
    table: dict[str, Any],
    name: str,
    expected: tuple[object, ...],
    *,
    label: str,
) -> None:
    value = table.get(name)
    if not isinstance(value, list) or tuple(value) != expected:
        raise ValueError(f"{label}.{name} must be exactly {list(expected)!r}")


def _validate_intervention(payload: dict[str, Any]) -> None:
    """Fail closed if the declared 288-row protocol drifts from the selector."""

    if payload.get("schema_version") != 1:
        raise ValueError("intervention schema_version must be exactly 1")
    if payload.get("experiment") != "e2_safety_utility_frontier":
        raise ValueError("unexpected E2 intervention experiment identity")
    if payload.get("status") != "planned_preconfirmation_tuning_only":
        raise ValueError("intervention status must remain preconfirmation tuning only")

    validate_frozen_e2_intervention_protocol(payload)

    _exact_list(_table(payload, "run"), "seeds", PILOT_SEEDS, label="run")
    _exact_list(
        _table(payload, "controls"),
        "names",
        (CONTROL_ARM,),
        label="controls",
    )
    _exact_list(
        _table(payload, "required_core_learned_arms"),
        "names",
        LEARNED_ARMS,
        label="required_core_learned_arms",
    )
    _exact_list(
        _table(payload, "regularization"),
        "positive_weights",
        POSITIVE_WEIGHTS,
        label="regularization",
    )

    selection = _table(payload, "selection")
    if selection.get("rule") != "validation_pareto_frontier_only":
        raise ValueError("selection.rule must be validation_pareto_frontier_only")
    artifact = _table(payload, "selection_artifact")
    if artifact.get("input_split") != "validation_only":
        raise ValueError("selection_artifact.input_split must be validation_only")
    if artifact.get("final_test_access") is not False:
        raise ValueError("selection_artifact.final_test_access must be false")
    if artifact.get("expected_observations") != EXPECTED_OBSERVATIONS:
        raise ValueError(
            "selection_artifact.expected_observations must be exactly "
            f"{EXPECTED_OBSERVATIONS}"
        )
    if artifact.get("required_weight_freezes") != 18:
        raise ValueError("selection_artifact.required_weight_freezes must be exactly 18")
    reuse = artifact.get("profile_label_manifest_reuse")
    if not isinstance(reuse, str) or "all five weights" not in reuse:
        raise ValueError(
            "selection_artifact.profile_label_manifest_reuse must freeze five-weight reuse"
        )


def _weight_slug(weight: float) -> str:
    text = format(weight, ".12g")
    return text.replace(".", "p")


def _profile_manifest_path(domain: str, family: str, seed: int) -> str:
    return (
        "runs/e2_frontier/profile_labels/"
        f"{domain}/{family}/data_seed_{seed}/crossfit_label_manifest.json"
    )


def _profile_coverage_path(domain: str, family: str, seed: int) -> str:
    return (
        "runs/e2_frontier/profile_coverage/"
        f"{domain}/{family}/data_seed_{seed}/profile_coverage_gate.json"
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _lower_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _profile_input_cell(domain: str, family: str, seed: int) -> dict[str, Any]:
    """Freeze both raw and self-authenticating hashes when a profile cell exists."""

    assembly_relative = _profile_manifest_path(domain, family, seed)
    coverage_relative = _profile_coverage_path(domain, family, seed)
    assembly_path = ROOT / assembly_relative
    coverage_path = ROOT / coverage_relative
    if not assembly_path.exists() and not coverage_path.exists():
        return {
            "domain": domain,
            "model_family": family,
            "seed": seed,
            "ready_for_evidence": False,
            "blocker": "assembled labels and coverage gate are missing",
            "teacher_plan_sha256": None,
            "assembly": {
                "path": assembly_relative,
                "file_sha256": None,
                "manifest_sha256": None,
            },
            "coverage": {
                "path": coverage_relative,
                "file_sha256": None,
                "manifest_sha256": None,
            },
        }
    if not assembly_path.is_file() or not coverage_path.is_file():
        missing = assembly_relative if not assembly_path.is_file() else coverage_relative
        raise ValueError(
            "profile input cell is partially materialized; preserve it and complete or "
            f"prospectively replace the cell before planning: {missing}"
        )

    assembly = _load_json_object(assembly_path, label="profile label manifest")
    assembly_declared = _lower_sha256(
        assembly.get("manifest_sha256"), label="profile label manifest_sha256"
    )
    assembly_unsigned = dict(assembly)
    assembly_unsigned.pop("manifest_sha256", None)
    if canonical_sha256(assembly_unsigned) != assembly_declared:
        raise ValueError(f"profile label manifest self-hash mismatch: {assembly_path}")
    teacher_plan_sha = _lower_sha256(
        assembly.get("plan_sha256"), label="profile teacher plan_sha256"
    )
    expected_assembly = {
        "schema_version": 1,
        "status": "complete_crossfit_label_manifest",
        "semantic_arm": PROFILE_ARM,
        "domain": domain,
        "model_family": family,
        "data_seed": seed,
        "engineering_smoke": False,
        "evidence_eligible": True,
    }
    mismatches = {
        field: {"expected": expected, "found": assembly.get(field)}
        for field, expected in expected_assembly.items()
        if assembly.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            "profile label manifest is not evidence-ready: "
            + json.dumps(mismatches, sort_keys=True)
        )

    coverage = _load_json_object(coverage_path, label="profile coverage manifest")
    coverage_declared = _lower_sha256(
        coverage.get("manifest_sha256"), label="profile coverage manifest_sha256"
    )
    coverage_unsigned = dict(coverage)
    coverage_unsigned.pop("manifest_sha256", None)
    if canonical_sha256(coverage_unsigned) != coverage_declared:
        raise ValueError(f"profile coverage manifest self-hash mismatch: {coverage_path}")
    expected_coverage = {
        "schema_version": 1,
        "status": "gate_passed",
        "scientific_gate_passed": True,
        "evidence_eligible": True,
        "plan_sha256": teacher_plan_sha,
        "domain": domain,
        "model_family": family,
        "data_seed": seed,
        "engineering_smoke": False,
    }
    coverage_mismatches = {
        field: {"expected": expected, "found": coverage.get(field)}
        for field, expected in expected_coverage.items()
        if coverage.get(field) != expected
    }
    assembly_link = coverage.get("assembly")
    if (
        not isinstance(assembly_link, dict)
        or assembly_link.get("manifest_sha256") != assembly_declared
        or assembly_link.get("file_sha256") != file_sha256(assembly_path)
    ):
        coverage_mismatches["assembly"] = {
            "expected_manifest_sha256": assembly_declared,
            "found": assembly_link,
        }
    if coverage_mismatches:
        raise ValueError(
            "profile coverage manifest is not evidence-ready: "
            + json.dumps(coverage_mismatches, sort_keys=True)
        )
    return {
        "domain": domain,
        "model_family": family,
        "seed": seed,
        "ready_for_evidence": True,
        "blocker": None,
        "teacher_plan_sha256": teacher_plan_sha,
        "assembly": {
            "path": assembly_relative,
            "file_sha256": file_sha256(assembly_path),
            "manifest_sha256": assembly_declared,
        },
        "coverage": {
            "path": coverage_relative,
            "file_sha256": file_sha256(coverage_path),
            "manifest_sha256": coverage_declared,
        },
    }


def _profile_teacher_plan_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = _repo_relative(resolved, label="profile teacher plan")
    managed = Path("runs/e2_frontier")
    try:
        relative.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            "profile teacher plan must lie below runs/e2_frontier"
        ) from error
    if not resolved.exists():
        return {
            "path": relative.as_posix(),
            "file_sha256": None,
            "plan_sha256": None,
            "code": None,
            "ready_for_evidence": False,
            "blocker": "profile teacher plan is missing",
        }
    if not resolved.is_file():
        raise ValueError(f"profile teacher plan must be a file: {resolved}")
    payload = _load_json_object(resolved, label="profile teacher plan")
    declared = _lower_sha256(
        payload.get("plan_sha256"), label="profile teacher plan_sha256"
    )
    unsigned = dict(payload)
    unsigned.pop("plan_sha256", None)
    if canonical_sha256(unsigned) != declared:
        raise ValueError(f"profile teacher plan self-hash mismatch: {resolved}")
    if payload.get("schema_version") != 2 or payload.get("task_count") != 330:
        raise ValueError("profile teacher plan must be the canonical 330-task schema-v2 plan")
    code = payload.get("code")
    code_ready = (
        isinstance(code, dict)
        and isinstance(code.get("git_revision"), str)
        and bool(code["git_revision"])
        and code.get("git_dirty") is False
    )
    return {
        "path": relative.as_posix(),
        "file_sha256": file_sha256(resolved),
        "plan_sha256": declared,
        "code": code,
        "ready_for_evidence": code_ready,
        "blocker": None if code_ready else "profile teacher plan is not from a clean revision",
    }


def _output_dir(
    domain: str,
    family: str,
    arm: str,
    seed: int,
    weight: float | None,
) -> str:
    prefix = OUTPUT_ROOT / domain / family / arm
    if weight is not None:
        prefix /= f"weight_{_weight_slug(weight)}"
    return (prefix / f"seed_{seed}").as_posix()


def build_plan(
    config_path: Path = ROOT / INTERVENTION_CONFIG,
    profile_teacher_plan_path: Path = ROOT / PROFILE_TEACHER_PLAN,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config_relative = _repo_relative(config_path, label="intervention config")
    if config_relative != INTERVENTION_CONFIG:
        raise ValueError(
            "the canonical E2 planner accepts only "
            f"{INTERVENTION_CONFIG.as_posix()}"
        )
    try:
        with config_path.open("rb") as stream:
            intervention = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read intervention config {config_path}: {error}") from error
    _validate_intervention(intervention)

    base_entries: dict[str, dict[str, str]] = {}
    base_configs: dict[str, Any] = {}
    for domain in DOMAINS:
        relative = BASE_CONFIGS.get(domain)
        if relative is None:
            raise ValueError(f"no base config is registered for {domain!r}")
        path = ROOT / relative
        config = load_config(path)
        if config.data.task != domain:
            raise ValueError(
                f"{relative} declares {config.data.task!r}, expected {domain!r}"
            )
        if config.data.history_length != HISTORY_LENGTH:
            raise ValueError(f"{relative} must declare four-frame histories")
        if not config.evaluation.emit_audit_records:
            raise ValueError(f"{relative} must emit validation audit records")
        if config.objective.fcsrl_head_hidden_dim != FCSRL_HEAD_HIDDEN_DIM:
            raise ValueError(
                f"{relative} must freeze FCSRL head width {FCSRL_HEAD_HIDDEN_DIM}"
            )
        base_entries[domain] = {
            "path": relative,
            "sha256": config_sha256(path),
        }
        base_configs[domain] = config

    task_specs: list[tuple[str, str, str, int, float | None]] = [
        (domain, family, CONTROL_ARM, seed, None)
        for domain, family, seed in itertools.product(
            DOMAINS, MODEL_FAMILIES, PILOT_SEEDS
        )
    ]
    task_specs.extend(
        (domain, family, arm, seed, float(weight))
        for domain, family, arm, weight, seed in itertools.product(
            DOMAINS,
            MODEL_FAMILIES,
            LEARNED_ARMS,
            POSITIVE_WEIGHTS,
            PILOT_SEEDS,
        )
    )

    profile_cells = {
        (domain, family, seed): _profile_input_cell(domain, family, seed)
        for domain, family, seed in itertools.product(
            DOMAINS, MODEL_FAMILIES, PILOT_SEEDS
        )
    }
    teacher_plan_entry = _profile_teacher_plan_entry(profile_teacher_plan_path)

    tasks: list[dict[str, Any]] = []
    for task_id, (domain, family, arm, seed, weight) in enumerate(task_specs):
        base = base_configs[domain]
        safety_weight = 0.0 if arm == CONTROL_ARM else float(weight)
        kl_weight = 0.0 if family == "ae" else float(base.objective.kl_weight)
        resolved = dataclasses.replace(
            base,
            run=dataclasses.replace(base.run, seed=seed, device="cuda"),
            data=dataclasses.replace(base.data, history_length=HISTORY_LENGTH),
            model=dataclasses.replace(
                base.model,
                family=family,
                history_encoder=HISTORY_ENCODER,
                latent_dim=LATENT_DIM,
            ),
            objective=dataclasses.replace(
                base.objective,
                safety_arm=arm,
                safety_weight=safety_weight,
                kl_weight=kl_weight,
                fcsrl_head_hidden_dim=FCSRL_HEAD_HIDDEN_DIM,
            ),
        )
        validate_config(resolved)
        profile_manifest = (
            _profile_manifest_path(domain, family, seed)
            if arm == PROFILE_ARM
            else None
        )
        profile_coverage = (
            _profile_coverage_path(domain, family, seed)
            if arm == PROFILE_ARM
            else None
        )
        profile_cell = profile_cells[(domain, family, seed)] if arm == PROFILE_ARM else None
        task: dict[str, Any] = {
            "task_id": task_id,
            "domain": domain,
            "model_family": family,
            "history_mode": HISTORY_MODE,
            "history_encoder": HISTORY_ENCODER,
            "history_length": HISTORY_LENGTH,
            "latent_dim": LATENT_DIM,
            "safety_arm": arm,
            "seed": seed,
            "regularization_weight": weight,
            "safety_weight": safety_weight,
            "kl_weight": kl_weight,
            "fcsrl_head_hidden_dim": FCSRL_HEAD_HIDDEN_DIM,
            "device": "cuda",
            "selection_split": "validation_only",
            "calibration_or_final_test_selection_access": False,
            "base_config": base_entries[domain]["path"],
            "base_config_sha256": base_entries[domain]["sha256"],
            "profile_label_manifest": profile_manifest,
            "profile_coverage_manifest": profile_coverage,
            "profile_teacher_plan": (
                teacher_plan_entry["path"] if arm == PROFILE_ARM else None
            ),
            "profile_teacher_plan_file_sha256": (
                teacher_plan_entry["file_sha256"] if arm == PROFILE_ARM else None
            ),
            "profile_teacher_plan_sha256": (
                profile_cell["teacher_plan_sha256"] if profile_cell is not None else None
            ),
            "profile_label_manifest_sha256": (
                profile_cell["assembly"]["manifest_sha256"]
                if profile_cell is not None
                else None
            ),
            "profile_label_manifest_file_sha256": (
                profile_cell["assembly"]["file_sha256"]
                if profile_cell is not None
                else None
            ),
            "profile_coverage_manifest_sha256": (
                profile_cell["coverage"]["manifest_sha256"]
                if profile_cell is not None
                else None
            ),
            "profile_coverage_manifest_file_sha256": (
                profile_cell["coverage"]["file_sha256"]
                if profile_cell is not None
                else None
            ),
            "output_dir": _output_dir(domain, family, arm, seed, weight),
        }
        task["task_sha256"] = task_sha256(task)
        tasks.append(task)

    control_count = sum(task["safety_arm"] == CONTROL_ARM for task in tasks)
    learned_count = len(tasks) - control_count
    if (control_count, learned_count, len(tasks)) != (18, 270, EXPECTED_OBSERVATIONS):
        raise AssertionError("E2 preconfirmation factorial count drifted")
    if len({task["output_dir"] for task in tasks}) != len(tasks):
        raise AssertionError("E2 preconfirmation outputs are not unique")

    profile_groups: dict[tuple[str, str, int], set[str]] = {}
    for task in tasks:
        if task["safety_arm"] != PROFILE_ARM:
            continue
        key = (task["domain"], task["model_family"], task["seed"])
        profile_groups.setdefault(key, set()).add(task["profile_label_manifest"])
    if len(profile_groups) != 18 or any(len(paths) != 1 for paths in profile_groups.values()):
        raise AssertionError("predicted-profile five-weight manifest reuse drifted")

    ready_cell_teacher_hashes = {
        cell["teacher_plan_sha256"]
        for cell in profile_cells.values()
        if cell["ready_for_evidence"]
    }
    if (
        teacher_plan_entry["plan_sha256"] is not None
        and ready_cell_teacher_hashes
        and ready_cell_teacher_hashes != {teacher_plan_entry["plan_sha256"]}
    ):
        raise ValueError(
            "profile label/coverage cells do not all authenticate the pinned teacher plan"
        )
    code_state = _git_code_state()
    profile_inputs_ready = (
        bool(teacher_plan_entry["ready_for_evidence"])
        and teacher_plan_entry["code"] == code_state
        and all(bool(cell["ready_for_evidence"]) for cell in profile_cells.values())
        and ready_cell_teacher_hashes == {teacher_plan_entry["plan_sha256"]}
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "e2_preconfirmation_validation_weight_freeze_288",
        "status": (
            READY_PLAN_STATUS
            if profile_inputs_ready
            else "blocked_on_production_profile_inputs"
        ),
        "code": code_state,
        "intervention_config": {
            "path": config_relative.as_posix(),
            "sha256": config_sha256(config_path),
        },
        "base_configs": base_entries,
        "scope": {
            "selection_split": "validation_only",
            "calibration_access": False,
            "final_test_access": False,
            "selection_entrypoint": artifact_entrypoint(intervention),
        },
        "axes": {
            "domains": list(DOMAINS),
            "model_families": list(MODEL_FAMILIES),
            "history_modes": [HISTORY_MODE],
            "control_arms": [CONTROL_ARM],
            "learned_arms": list(LEARNED_ARMS),
            "positive_weights": list(POSITIVE_WEIGHTS),
            "seeds": list(PILOT_SEEDS),
        },
        "fixed": {
            "history_encoder": HISTORY_ENCODER,
            "history_length": HISTORY_LENGTH,
            "latent_dim": LATENT_DIM,
            "fcsrl_head_hidden_dim": FCSRL_HEAD_HIDDEN_DIM,
        },
        "counts": {
            "controls": control_count,
            "learned": learned_count,
            "total": len(tasks),
        },
        "task_formula": (
            "18 none controls plus 3 learned arms x 5 weights x 3 domains x "
            "2 families x 3 pilot seeds"
        ),
        "profile_handoff": {
            "required_arm": PROFILE_ARM,
            "cell_count": 18,
            "weights_per_cell": len(POSITIVE_WEIGHTS),
            "manifest_filename": "crossfit_label_manifest.json",
            "coverage_manifest_filename": "profile_coverage_gate.json",
            "reuse_rule": "one checksum-identical manifest per domain-family-seed cell",
            "all_inputs_ready": profile_inputs_ready,
            "teacher_plan": teacher_plan_entry,
            "cells": list(profile_cells.values()),
        },
        "runner": {
            "path": "scripts/run_e2_preconfirmation_task.py",
            "task_manifest": "preconfirmation_task_manifest.json",
            "evidence_requires_clean_matching_revision": True,
            "evidence_requires_validation_only_trainer": True,
            "validation_postfit_audit_implemented": True,
            "authenticated_task_handoff_implemented": True,
            "evidence_requires_matched_radius_selector_rows": True,
            "execution_enabled": profile_inputs_ready,
            "generic_execution_enabled": profile_inputs_ready,
            "profile_execution_enabled": profile_inputs_ready,
            "execution_blocker": (
                None
                if profile_inputs_ready
                else "profile tasks require their checksum-pinned label and coverage inputs"
            ),
            "downstream_status": (
                "downstream matched-radius reduction and canonical validation-only "
                "aggregation remain separate authenticated post-run steps"
            ),
        },
        "task_count": len(tasks),
        "tasks": tasks,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def artifact_entrypoint(intervention: dict[str, Any]) -> str:
    artifact = _table(intervention, "selection_artifact")
    value = artifact.get("entrypoint")
    if not isinstance(value, str) or not value:
        raise ValueError("selection_artifact.entrypoint must be a non-empty string")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / INTERVENTION_CONFIG,
        help="frozen intervention config; alternate paths are rejected",
    )
    parser.add_argument(
        "--profile-teacher-plan",
        type=Path,
        default=ROOT / PROFILE_TEACHER_PLAN,
        help="canonical teacher plan to checksum-pin when profile artifacts exist",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = build_plan(args.config, args.profile_teacher_plan)
    except ValueError as error:
        parser.error(str(error))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.exists():
        parser.error(
            f"refusing to overwrite existing preconfirmation plan: {output}; "
            "choose a prospectively new path"
        )
    write_json_atomic(output, plan)
    print(
        f"wrote {plan['task_count']} validation-only preconfirmation tasks "
        f"to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
