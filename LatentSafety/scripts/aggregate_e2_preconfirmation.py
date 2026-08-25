#!/usr/bin/env python3
"""Authenticate and aggregate the exact 288-task validation-only E2 pilot grid."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMPLEMENTATION_PATHS = (
    ("scripts/aggregate_e2_preconfirmation.py", Path(__file__).resolve()),
    (
        "src/latent_safety/analysis/matched_radius.py",
        ROOT / "src/latent_safety/analysis/matched_radius.py",
    ),
    (
        "src/latent_safety/analysis/preconfirmation.py",
        ROOT / "src/latent_safety/analysis/preconfirmation.py",
    ),
    (
        "src/latent_safety/analysis/preconfirmation_artifacts.py",
        ROOT / "src/latent_safety/analysis/preconfirmation_artifacts.py",
    ),
    (
        "src/latent_safety/metrics/defect.py",
        ROOT / "src/latent_safety/metrics/defect.py",
    ),
    (
        "src/latent_safety/learning/profile_protocol.py",
        ROOT / "src/latent_safety/learning/profile_protocol.py",
    ),
    (
        "src/latent_safety/learning/profile_artifacts.py",
        ROOT / "src/latent_safety/learning/profile_artifacts.py",
    ),
    ("src/latent_safety/manifest.py", ROOT / "src/latent_safety/manifest.py"),
    ("src/latent_safety/records.py", ROOT / "src/latent_safety/records.py"),
)

from latent_safety.analysis.matched_radius import (  # noqa: E402
    CONTROL_REFERENCE_RELATIVE_RADIUS,
    MAX_RELATIVE_MASS_MISMATCH,
    RELATIVE_RADIUS_GRID,
    build_validation_radius_audit,
    match_validation_radius_audits,
    validate_paired_records,
)
from latent_safety.analysis.preconfirmation import (  # noqa: E402
    CONTROL_ARM,
    DOMAINS,
    EXPECTED_OBSERVATIONS,
    LEARNED_ARMS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    POSITIVE_WEIGHTS,
    PROFILE_ARM,
    PreconfirmationObservation,
    validate_frozen_e2_intervention_protocol,
)
from latent_safety.analysis.preconfirmation_artifacts import (  # noqa: E402
    PreconfirmationArtifactError,
    authenticate_preconfirmation_task_manifest,
    canonical_sha256,
    file_sha256,
)
from latent_safety.manifest import write_json_atomic  # noqa: E402
from latent_safety.records import read_jsonl  # noqa: E402


AGGREGATE_PROTOCOL = "e2_preconfirmation_validation_observations_v1"
OUTPUT_ROOT = Path("runs/e2_frontier/preconfirmation_288")
HISTORY_MODE = "stack_h4"
HISTORY_ENCODER = "stack"
HISTORY_LENGTH = 4
LATENT_DIM = 8
FCSRL_HEAD_HIDDEN_DIM = 64


class PreconfirmationAggregationError(ValueError):
    """The plan or a completed task violates the frozen aggregation contract."""


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


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreconfirmationAggregationError(f"cannot read {field} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PreconfirmationAggregationError(f"{field} must contain a JSON object")
    return payload


def _sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreconfirmationAggregationError(f"{field} must be a lowercase SHA-256")
    return value


def _repo_file(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise PreconfirmationAggregationError(f"{field} must be repository-relative")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PreconfirmationAggregationError(f"{field} escapes the repository") from error
    if not path.is_file():
        raise PreconfirmationAggregationError(f"{field} is missing: {path}")
    return path


def _task_hash(task: dict[str, Any]) -> str:
    unsigned = dict(task)
    unsigned.pop("task_sha256", None)
    return canonical_sha256(unsigned)


def _expected_specs() -> list[tuple[str, str, str, int, float | None]]:
    specs = [
        (domain, family, CONTROL_ARM, seed, None)
        for domain, family, seed in itertools.product(
            DOMAINS, MODEL_FAMILIES, PILOT_SEEDS
        )
    ]
    specs.extend(
        (domain, family, arm, seed, float(weight))
        for domain, family, arm, weight, seed in itertools.product(
            DOMAINS,
            MODEL_FAMILIES,
            LEARNED_ARMS,
            POSITIVE_WEIGHTS,
            PILOT_SEEDS,
        )
    )
    return specs


def _weight_slug(weight: float) -> str:
    return format(weight, ".12g").replace(".", "p")


def _expected_output_dir(
    domain: str,
    family: str,
    arm: str,
    seed: int,
    weight: float | None,
) -> str:
    path = OUTPUT_ROOT / domain / family / arm
    if weight is not None:
        path /= f"weight_{_weight_slug(weight)}"
    return (path / f"seed_{seed}").as_posix()


def validate_plan(plan_path: Path) -> dict[str, Any]:
    """Authenticate the plan and require the exact frozen 18+270 expansion."""

    plan = _load_object(plan_path, field="E2 preconfirmation plan")
    if (
        plan.get("schema_version") != 1
        or plan.get("experiment") != "e2_preconfirmation_validation_weight_freeze_288"
    ):
        raise PreconfirmationAggregationError("unsupported E2 preconfirmation plan identity")
    declared = _sha(plan.get("plan_sha256"), field="plan.plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    actual = canonical_sha256(unsigned)
    if not hmac.compare_digest(declared, actual):
        raise PreconfirmationAggregationError(
            f"plan self-hash mismatch: declared {declared}, recomputed {actual}"
        )
    if plan.get("status") != "indexed_validation_only_execution_ready":
        raise PreconfirmationAggregationError(
            "aggregation requires the fully profile-ready indexed execution plan"
        )
    code = plan.get("code")
    if (
        not isinstance(code, dict)
        or not isinstance(code.get("git_revision"), str)
        or not code.get("git_revision")
        or code.get("git_dirty") is not False
    ):
        raise PreconfirmationAggregationError(
            "aggregation requires a plan frozen from a clean committed revision"
        )
    current_code = _git_code_state()
    if current_code != code:
        raise PreconfirmationAggregationError(
            "aggregation code state must exactly match the plan's clean committed revision"
        )
    scope = plan.get("scope")
    if not isinstance(scope, dict) or (
        scope.get("selection_split") != "validation_only"
        or scope.get("calibration_access") is not False
        or scope.get("final_test_access") is not False
        or scope.get("selection_entrypoint")
        != "scripts/select_preconfirmation_weights.py"
    ):
        raise PreconfirmationAggregationError("plan scope is not validation-only")
    axes = plan.get("axes")
    expected_axes = {
        "domains": list(DOMAINS),
        "model_families": list(MODEL_FAMILIES),
        "history_modes": [HISTORY_MODE],
        "control_arms": [CONTROL_ARM],
        "learned_arms": list(LEARNED_ARMS),
        "positive_weights": list(POSITIVE_WEIGHTS),
        "seeds": list(PILOT_SEEDS),
    }
    if axes != expected_axes:
        raise PreconfirmationAggregationError("plan axes do not equal the frozen factorial")
    if plan.get("fixed") != {
        "history_encoder": HISTORY_ENCODER,
        "history_length": HISTORY_LENGTH,
        "latent_dim": LATENT_DIM,
        "fcsrl_head_hidden_dim": FCSRL_HEAD_HIDDEN_DIM,
    }:
        raise PreconfirmationAggregationError("plan fixed architecture fields drifted")
    if plan.get("counts") != {"controls": 18, "learned": 270, "total": 288}:
        raise PreconfirmationAggregationError("plan factorial counts drifted")

    intervention = plan.get("intervention_config")
    if not isinstance(intervention, dict):
        raise PreconfirmationAggregationError("plan.intervention_config must be an object")
    intervention_path = _repo_file(
        intervention.get("path"), field="plan.intervention_config.path"
    )
    if intervention.get("sha256") != file_sha256(intervention_path):
        raise PreconfirmationAggregationError("intervention config SHA mismatch")
    try:
        with intervention_path.open("rb") as stream:
            intervention_payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PreconfirmationAggregationError(
            f"cannot read intervention protocol config: {error}"
        ) from error
    try:
        validate_frozen_e2_intervention_protocol(intervention_payload)
        selection_artifact = intervention_payload["selection_artifact"]
    except (KeyError, TypeError, ValueError) as error:
        raise PreconfirmationAggregationError(
            f"intervention protocol contract drifted: {error}"
        ) from error
    required_selection_contract = {
        "entrypoint": "scripts/select_preconfirmation_weights.py",
        "aggregator_entrypoint": "scripts/aggregate_e2_preconfirmation.py",
        "aggregator_protocol": AGGREGATE_PROTOCOL,
        "task_handoff_protocol": "e2_preconfirmation_task_handoff_v1",
        "input_split": "validation_only",
        "expected_observations": EXPECTED_OBSERVATIONS,
        "required_radius_sidecars": EXPECTED_OBSERVATIONS,
        "required_weight_freezes": 18,
        "selected_radius_freezes_required": True,
        "final_test_access": False,
    }
    if any(
        selection_artifact.get(field) != expected
        for field, expected in required_selection_contract.items()
    ):
        raise PreconfirmationAggregationError(
            "intervention selection-artifact/aggregator contract drifted"
        )
    base_configs = plan.get("base_configs")
    if not isinstance(base_configs, dict) or set(base_configs) != set(DOMAINS):
        raise PreconfirmationAggregationError("plan must pin one base config per domain")
    base_hashes: dict[str, str] = {}
    base_kl_weights: dict[str, float] = {}
    for domain in DOMAINS:
        entry = base_configs.get(domain)
        if not isinstance(entry, dict):
            raise PreconfirmationAggregationError(f"base config entry missing for {domain}")
        path = _repo_file(entry.get("path"), field=f"base_configs.{domain}.path")
        digest = file_sha256(path)
        if entry.get("sha256") != digest:
            raise PreconfirmationAggregationError(f"base config SHA mismatch for {domain}")
        base_hashes[domain] = digest
        try:
            with path.open("rb") as stream:
                base_payload = tomllib.load(stream)
            objective = base_payload["objective"]
            base_kl = objective["kl_weight"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
            raise PreconfirmationAggregationError(
                f"cannot read base objective KL weight for {domain}: {error}"
            ) from error
        if (
            isinstance(base_kl, bool)
            or not isinstance(base_kl, (int, float))
            or not math.isfinite(float(base_kl))
            or float(base_kl) <= 0.0
        ):
            raise PreconfirmationAggregationError(
                f"base objective KL weight for {domain} must be finite and positive"
            )
        base_kl_weights[domain] = float(base_kl)

    tasks = plan.get("tasks")
    expected_specs = _expected_specs()
    if (
        not isinstance(tasks, list)
        or len(tasks) != EXPECTED_OBSERVATIONS
        or plan.get("task_count") != EXPECTED_OBSERVATIONS
    ):
        raise PreconfirmationAggregationError("plan must contain exactly 288 tasks")
    outputs: set[str] = set()
    task_hashes: set[str] = set()
    for task_id, (task_value, spec) in enumerate(zip(tasks, expected_specs, strict=True)):
        if not isinstance(task_value, dict):
            raise PreconfirmationAggregationError(f"plan.tasks[{task_id}] must be an object")
        task = task_value
        domain, family, arm, seed, weight = spec
        expected_identity = {
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
            "safety_weight": 0.0 if arm == CONTROL_ARM else weight,
            "kl_weight": 0.0 if family == "ae" else base_kl_weights[domain],
            "fcsrl_head_hidden_dim": FCSRL_HEAD_HIDDEN_DIM,
            "device": "cuda",
            "selection_split": "validation_only",
            "calibration_or_final_test_selection_access": False,
            "base_config_sha256": base_hashes[domain],
        }
        mismatches = {
            field: {"found": task.get(field), "expected": expected}
            for field, expected in expected_identity.items()
            if task.get(field) != expected
        }
        if mismatches:
            raise PreconfirmationAggregationError(
                f"task {task_id} identity mismatch: " + json.dumps(mismatches, sort_keys=True)
            )
        base_entry = base_configs[domain]
        if task.get("base_config") != base_entry.get("path"):
            raise PreconfirmationAggregationError(f"task {task_id} base-config path mismatch")
        task_declared = _sha(task.get("task_sha256"), field=f"tasks[{task_id}].task_sha256")
        if not hmac.compare_digest(task_declared, _task_hash(task)):
            raise PreconfirmationAggregationError(f"task {task_id} self-hash mismatch")
        if task_declared in task_hashes:
            raise PreconfirmationAggregationError("duplicate task SHA-256")
        task_hashes.add(task_declared)
        output = task.get("output_dir")
        if not isinstance(output, str) or not output:
            raise PreconfirmationAggregationError(f"task {task_id} output_dir is invalid")
        output_path = Path(output)
        expected_output = _expected_output_dir(domain, family, arm, seed, weight)
        if output != expected_output:
            raise PreconfirmationAggregationError(
                f"task {task_id} output_dir mismatch: found {output!r}, "
                f"expected {expected_output!r}"
            )
        try:
            relative = output_path.relative_to(OUTPUT_ROOT)
        except ValueError as error:
            raise PreconfirmationAggregationError(
                f"task {task_id} output lies outside {OUTPUT_ROOT.as_posix()}"
            ) from error
        if relative == Path(".") or output in outputs:
            raise PreconfirmationAggregationError("duplicate or root task output directory")
        outputs.add(output)
        profile_fields = (
            "profile_label_manifest",
            "profile_coverage_manifest",
            "profile_teacher_plan",
            "profile_teacher_plan_file_sha256",
            "profile_teacher_plan_sha256",
            "profile_label_manifest_sha256",
            "profile_label_manifest_file_sha256",
            "profile_coverage_manifest_sha256",
            "profile_coverage_manifest_file_sha256",
        )
        if arm == PROFILE_ARM:
            for field in profile_fields:
                value = task.get(field)
                if value is None:
                    raise PreconfirmationAggregationError(
                        f"profile task {task_id} has no pinned {field}"
                    )
                if field.endswith("sha256"):
                    _sha(value, field=f"tasks[{task_id}].{field}")
        elif any(task.get(field) is not None for field in profile_fields):
            raise PreconfirmationAggregationError(
                f"non-profile task {task_id} contains profile handoff fields"
            )

    handoff = plan.get("profile_handoff")
    if not isinstance(handoff, dict) or (
        handoff.get("all_inputs_ready") is not True
        or handoff.get("cell_count") != 18
        or handoff.get("weights_per_cell") != 5
    ):
        raise PreconfirmationAggregationError("plan profile handoff is not complete")
    runner = plan.get("runner")
    if not isinstance(runner, dict) or any(
        (
            runner.get("path") != "scripts/run_e2_preconfirmation_task.py",
            runner.get("task_manifest") != "preconfirmation_task_manifest.json",
            runner.get("evidence_requires_clean_matching_revision") is not True,
            runner.get("evidence_requires_validation_only_trainer") is not True,
            runner.get("validation_postfit_audit_implemented") is not True,
            runner.get("authenticated_task_handoff_implemented") is not True,
            runner.get("execution_enabled") is not True,
            runner.get("generic_execution_enabled") is not True,
            runner.get("profile_execution_enabled") is not True,
            runner.get("execution_blocker") is not None,
        )
    ):
        raise PreconfirmationAggregationError(
            "plan runner does not freeze the executable authenticated handoff contract"
        )
    return plan


def _point_at_reference(audit: Any) -> Any:
    matches = [
        point
        for point in audit.curve
        if math.isclose(
            point.relative_radius,
            CONTROL_REFERENCE_RELATIVE_RADIUS,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ]
    if len(matches) != 1:
        raise PreconfirmationAggregationError(
            "control audit does not contain the frozen reference radius exactly once"
        )
    return matches[0]


def _selector_fields(point: Any, *, task_id: int) -> dict[str, Any]:
    if (
        point.trajectory_balanced_p95_required_violation is None
        or point.median_nonself_neighborhood_mass is None
        or point.eligible_given_viable is None
    ):
        raise PreconfirmationAggregationError(
            f"task {task_id} has no complete selector metrics at its frozen radius"
        )
    empty_trajectory_count = int(point.empty_trajectory_count)
    return {
        "safety": float(point.trajectory_balanced_p95_required_violation),
        "eligible_center_coverage": float(point.eligible_given_viable),
        "neighborhood_mass": float(point.median_nonself_neighborhood_mass),
        "empty_trajectory_count": empty_trajectory_count,
        "coverage_complete": empty_trajectory_count == 0,
    }


def _signed_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["radius_freeze_sha256"] = canonical_sha256(signed)
    return signed


def aggregate(plan_path: Path) -> dict[str, Any]:
    """Return a deterministic, selector-ready 288-row artifact or fail without output."""

    root = ROOT.resolve()
    plan_path = plan_path.resolve()
    plan = validate_plan(plan_path)
    plan_sha = str(plan["plan_sha256"])
    tasks: list[dict[str, Any]] = plan["tasks"]
    handoffs: dict[int, dict[str, Any]] = {}
    task_manifest_paths: dict[int, Path] = {}
    audits: dict[int, Any] = {}
    records: dict[int, Any] = {}
    checkpoint_hashes: set[str] = set()
    for task in tasks:
        task_id = int(task["task_id"])
        try:
            handoff, manifest_path = authenticate_preconfirmation_task_manifest(
                repo_root=root,
                plan_path=plan_path,
                plan_sha256=plan_sha,
                task=task,
            )
        except (OSError, ValueError, PreconfirmationArtifactError) as error:
            raise PreconfirmationAggregationError(
                f"task {task_id} artifact authentication failed: {error}"
            ) from error
        selected_checkpoint = str(handoff["provenance"]["selected_checkpoint_sha256"])
        if selected_checkpoint in checkpoint_hashes:
            raise PreconfirmationAggregationError(
                f"task {task_id} reuses another task's selected checkpoint"
            )
        checkpoint_hashes.add(selected_checkpoint)
        audit_path = (root / task["output_dir"] / "audit_validation.jsonl").resolve()
        try:
            task_records = read_jsonl(audit_path)
            radius_audit = build_validation_radius_audit(
                task_records, relative_radii=RELATIVE_RADIUS_GRID
            )
        except (OSError, ValueError) as error:
            raise PreconfirmationAggregationError(
                f"task {task_id} validation radius audit failed: {error}"
            ) from error
        handoffs[task_id] = handoff
        task_manifest_paths[task_id] = manifest_path
        audits[task_id] = radius_audit
        records[task_id] = task_records

    controls: dict[tuple[str, str, int], dict[str, Any]] = {}
    for task in tasks:
        if task["safety_arm"] != CONTROL_ARM:
            continue
        key = (task["domain"], task["model_family"], task["seed"])
        if key in controls:
            raise PreconfirmationAggregationError(f"duplicate none control for {key!r}")
        controls[key] = task
    expected_control_keys = set(itertools.product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS))
    if set(controls) != expected_control_keys:
        raise PreconfirmationAggregationError("none-control pairing map is incomplete")

    observations: list[dict[str, Any]] = []
    row_provenance: list[dict[str, Any]] = []
    profile_reuse: dict[tuple[str, str, int], set[tuple[str, ...]]] = {}
    for task in tasks:
        task_id = int(task["task_id"])
        handoff = handoffs[task_id]
        utility = handoff["utility_summary"]
        identity = {
            "domain": task["domain"],
            "model_family": task["model_family"],
            "arm": task["safety_arm"],
            "seed": task["seed"],
            "weight": task["regularization_weight"],
        }
        manifest_path = task_manifest_paths[task_id]
        common_provenance = {
            "task_id": task_id,
            **identity,
            "task_sha256": task["task_sha256"],
            "task_manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": file_sha256(manifest_path),
                "manifest_sha256": handoff["manifest_sha256"],
            },
        }
        if task["safety_arm"] == CONTROL_ARM:
            audit = audits[task_id]
            point = _point_at_reference(audit)
            metric_fields = _selector_fields(point, task_id=task_id)
            radius_freeze = _signed_freeze(
                {
                    **common_provenance,
                    "kind": "none_control_reference",
                    "control_task_id": task_id,
                    "control_reference_relative_radius": point.relative_radius,
                    "control_absolute_radius": point.absolute_radius,
                    "learned_relative_radius": None,
                    "learned_absolute_radius": None,
                    "control_audit_sha256": audit.audit_sha256,
                    "learned_audit_sha256": None,
                    "pairing_sha256": audit.pairing_sha256,
                    "relative_neighborhood_mass_mismatch": None,
                    "maximum_relative_neighborhood_mass_mismatch": (
                        MAX_RELATIVE_MASS_MISMATCH
                    ),
                    "mass_match_passed": None,
                    "strict_safety_improvement": None,
                    "radius_selection_used_safety": False,
                }
            )
        else:
            control = controls[(task["domain"], task["model_family"], task["seed"])]
            control_id = int(control["task_id"])
            try:
                validate_paired_records(records[control_id], records[task_id])
                matched = match_validation_radius_audits(
                    audits[control_id],
                    audits[task_id],
                    control_reference_relative_radius=CONTROL_REFERENCE_RELATIVE_RADIUS,
                    max_relative_mass_mismatch=MAX_RELATIVE_MASS_MISMATCH,
                )
            except ValueError as error:
                raise PreconfirmationAggregationError(
                    f"task {task_id} is not physically paired to none task {control_id}: {error}"
                ) from error
            if not matched.selector_metrics_ready or matched.learned_point is None:
                raise PreconfirmationAggregationError(
                    f"task {task_id} matched-radius selector metrics are incomplete"
                )
            metric_fields = _selector_fields(matched.learned_point, task_id=task_id)
            radius_freeze = _signed_freeze(
                {
                    **common_provenance,
                    "kind": "learned_mass_matched",
                    "control_task_id": control_id,
                    "control_reference_relative_radius": (
                        matched.control_reference_relative_radius
                    ),
                    "control_absolute_radius": matched.control_point.absolute_radius,
                    "learned_relative_radius": matched.learned_relative_radius,
                    "learned_absolute_radius": matched.learned_point.absolute_radius,
                    "control_audit_sha256": matched.control_audit_sha256,
                    "learned_audit_sha256": matched.learned_audit_sha256,
                    "pairing_sha256": matched.pairing_sha256,
                    "relative_neighborhood_mass_mismatch": (
                        matched.relative_neighborhood_mass_mismatch
                    ),
                    "maximum_relative_neighborhood_mass_mismatch": (
                        matched.maximum_relative_neighborhood_mass_mismatch
                    ),
                    "mass_match_passed": matched.mass_match_passed,
                    "strict_safety_improvement": matched.strict_safety_improvement,
                    "radius_selection_used_safety": matched.radius_selection_used_safety,
                }
            )
        observation: dict[str, Any] = {
            **identity,
            **metric_fields,
            "reconstruction": float(utility["reconstruction"]),
            "rollout": float(utility["rollout"]),
            "run_complete": True,
        }
        if task["safety_arm"] == PROFILE_ARM:
            profile = handoff["profile_handoff"]
            if not isinstance(profile, dict):
                raise PreconfirmationAggregationError(
                    f"profile task {task_id} has no authenticated profile handoff"
                )
            observation.update(
                {
                    "profile_normalized_p95_error": profile["coverage_gate"][
                        "normalized_p95_error"
                    ],
                    "profile_teacher_count": profile["teacher_count"],
                    "profile_teacher_failure_count": profile["teacher_failure_count"],
                    "profile_label_manifest_sha256": profile["label_assembly"][
                        "manifest_sha256"
                    ],
                }
            )
            reuse_key = (task["domain"], task["model_family"], task["seed"])
            profile_reuse.setdefault(reuse_key, set()).add(
                (
                    profile["teacher_plan"]["plan_sha256"],
                    profile["teacher_plan"]["sha256"],
                    profile["label_assembly"]["manifest_sha256"],
                    profile["label_assembly"]["sha256"],
                    profile["coverage_gate"]["manifest_sha256"],
                    profile["coverage_gate"]["sha256"],
                )
            )
        try:
            parsed = PreconfirmationObservation(**observation)
            parsed.validate()
        except (TypeError, ValueError) as error:
            raise PreconfirmationAggregationError(
                f"task {task_id} did not produce a valid selector observation: {error}"
            ) from error
        radius_freeze["observation_sha256"] = canonical_sha256(observation)
        # Re-sign after binding the exact numerical observation.
        radius_freeze.pop("radius_freeze_sha256", None)
        radius_freeze["radius_freeze_sha256"] = canonical_sha256(radius_freeze)
        observations.append(observation)
        row_provenance.append(radius_freeze)

    if len(profile_reuse) != 18 or any(len(values) != 1 for values in profile_reuse.values()):
        raise PreconfirmationAggregationError(
            "profile five-weight runs do not reuse one checksum-identical teacher/assembly/coverage handoff"
        )
    if len(observations) != EXPECTED_OBSERVATIONS or len(row_provenance) != EXPECTED_OBSERVATIONS:
        raise AssertionError("E2 aggregate row-count invariant drifted")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": AGGREGATE_PROTOCOL,
        "source_plan": {
            "path": plan_path.relative_to(root).as_posix(),
            "sha256": file_sha256(plan_path),
            "plan_sha256": plan_sha,
        },
        "implementation": [
            {
                "path": label,
                "sha256": file_sha256(path),
            }
            for label, path in IMPLEMENTATION_PATHS
        ],
        "scope": {
            "input_split": "validation_only",
            "calibration_access": False,
            "final_test_access": False,
            "paired_none_control": "same domain, model family, and pilot seed",
            "radius_selection_used_safety": False,
        },
        "radius_protocol": {
            "relative_radius_grid": list(RELATIVE_RADIUS_GRID),
            "control_reference_relative_radius": CONTROL_REFERENCE_RELATIVE_RADIUS,
            "maximum_relative_neighborhood_mass_mismatch": MAX_RELATIVE_MASS_MISMATCH,
        },
        "observation_count": len(observations),
        "row_provenance_count": len(row_provenance),
        "observations": observations,
        "row_provenance": row_provenance,
    }
    payload["aggregate_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    try:
        payload = aggregate(plan_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    write_json_atomic(output_path.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "observation_count": payload["observation_count"],
                "row_provenance_count": payload["row_provenance_count"],
                "aggregate_sha256": payload["aggregate_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
