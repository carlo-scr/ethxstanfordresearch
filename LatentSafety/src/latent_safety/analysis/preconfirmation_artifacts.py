"""Authenticated handoff artifacts for one completed E2 preconfirmation task.

The indexed runner writes ``preconfirmation_task_manifest.json`` only after a run has completed.
The manifest is a deterministic checksum envelope around the heterogeneous ordinary-trainer and
predicted-profile outputs.  Consumers do not trust the envelope alone: rebuilding it reopens and
revalidates every referenced file, the validation-only access boundary, the resolved task cell,
and (for the profile arm) the complete teacher/assembly/coverage handoff.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latent_safety.analysis.preconfirmation import (
    DOMAINS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    PROFILE_ARM,
)
from latent_safety.learning.profile_protocol import (
    COVERAGE_BUNDLES_PER_DOMAIN_SEED,
    FOLD_COUNT,
    FOLD_SEED,
    PROFILE_PROTOCOL_VERSION,
    REQUIRED_NORMALIZED_P95,
    assign_trajectory_folds,
    evaluate_profile_gate,
    teacher_seed,
)
from latent_safety.learning.profile_artifacts import ids_sha256
from latent_safety.learning.config import load_config
from latent_safety.learning.data import trajectory_split_assignment
from latent_safety.manifest import write_json_atomic
from latent_safety.records import read_jsonl


TASK_HANDOFF_PROTOCOL = "e2_preconfirmation_task_handoff_v1"
TASK_MANIFEST_FILENAME = "preconfirmation_task_manifest.json"


class PreconfirmationArtifactError(ValueError):
    """A completed task artifact violates the E2 handoff contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreconfirmationArtifactError(f"{field} must be a JSON object")
    return value


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreconfirmationArtifactError(f"cannot read {field} {path}: {error}") from error
    return _object(payload, field=field)


def _sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreconfirmationArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreconfirmationArtifactError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or (nonnegative and numeric < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise PreconfirmationArtifactError(f"{field} must be {qualifier}")
    return numeric


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreconfirmationArtifactError(f"{field} must be an integer >= {minimum}")
    return value


def _self_hash(payload: dict[str, Any], *, hash_field: str, field: str) -> str:
    declared = _sha(payload.get(hash_field), field=f"{field}.{hash_field}")
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    actual = canonical_sha256(unsigned)
    if not hmac.compare_digest(declared, actual):
        raise PreconfirmationArtifactError(
            f"{field} self-hash mismatch: declared {declared}, recomputed {actual}"
        )
    return declared


def _repo_path(repo_root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreconfirmationArtifactError(f"{field} must be a repository-relative path")
    declared = Path(value)
    if declared.is_absolute():
        raise PreconfirmationArtifactError(f"{field} must be repository-relative")
    resolved = (repo_root / declared).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise PreconfirmationArtifactError(f"{field} escapes the repository") from error
    return resolved


def _run_reference(run_dir: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreconfirmationArtifactError(f"{field} must be a non-empty file path")
    declared = Path(value)
    resolved = declared.resolve() if declared.is_absolute() else (run_dir / declared).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as error:
        raise PreconfirmationArtifactError(f"{field} points outside the task output") from error
    return resolved


def _base_config_reference(
    repo_root: Path,
    manifest: dict[str, Any],
    task: Mapping[str, object],
    *,
    field: str,
) -> None:
    config = _object(manifest.get("config"), field=f"{field}.config")
    declared_path = config.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise PreconfirmationArtifactError(f"{field}.config.path is missing")
    path = Path(declared_path)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    expected = _repo_path(repo_root, task.get("base_config"), field="task.base_config")
    if resolved != expected or config.get("sha256") != task.get("base_config_sha256"):
        raise PreconfirmationArtifactError(f"{field} base-config provenance mismatch")
    if file_sha256(expected) != task.get("base_config_sha256"):
        raise PreconfirmationArtifactError("planned base config file SHA changed")


def _artifact_entry(path: Path, run_dir: Path) -> dict[str, str]:
    if not path.is_file():
        raise PreconfirmationArtifactError(f"required task artifact is missing: {path}")
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": file_sha256(path),
    }


def _require_exact_scope(scope: object, *, field: str) -> dict[str, Any]:
    normalized = _object(scope, field=field)
    expected = {
        "name": "train_validation_only",
        "materialized_splits": ["train", "validation"],
        "evaluated_splits": ["train", "validation"],
        "rollout_splits": ["validation"],
        "oracle_action_profiles_materialized": False,
        "calibration_access": False,
        "final_test_access": False,
    }
    if normalized != expected:
        raise PreconfirmationArtifactError(
            f"{field} must equal the frozen train/validation-only scope"
        )
    return normalized


def _check_task_config(
    repo_root: Path,
    values: dict[str, Any],
    task: Mapping[str, object],
) -> None:
    base_path = _repo_path(
        repo_root, task.get("base_config"), field="task.base_config"
    )
    expected_base_sha = _sha(
        task.get("base_config_sha256"), field="task.base_config_sha256"
    )
    if file_sha256(base_path) != expected_base_sha:
        raise PreconfirmationArtifactError("planned base config file SHA changed")
    try:
        with base_path.open("rb") as stream:
            base = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PreconfirmationArtifactError(
            f"cannot read planned base config {base_path}: {error}"
        ) from error

    data = _object(values.get("data"), field="resolved_config.values.data")
    model = _object(values.get("model"), field="resolved_config.values.model")
    objective = _object(values.get("objective"), field="resolved_config.values.objective")
    run = _object(values.get("run"), field="resolved_config.values.run")
    evaluation = _object(
        values.get("evaluation"), field="resolved_config.values.evaluation"
    )
    base_data = _object(base.get("data"), field="base_config.data")
    base_model = _object(base.get("model"), field="base_config.model")
    base_objective = _object(base.get("objective"), field="base_config.objective")
    base_run = _object(base.get("run"), field="base_config.run")
    base_evaluation = _object(base.get("evaluation"), field="base_config.evaluation")

    expected_data = dict(base_data)
    expected_data["task"] = task.get("domain")
    expected_data["history_length"] = task.get("history_length")
    expected_model = dict(base_model)
    expected_model.update(
        {
            "family": task.get("model_family"),
            "history_encoder": task.get("history_encoder"),
            "latent_dim": task.get("latent_dim"),
        }
    )
    expected_objective = dict(base_objective)
    expected_objective.update(
        {
            "safety_arm": task.get("safety_arm"),
            "safety_weight": task.get("safety_weight"),
            "kl_weight": task.get("kl_weight"),
            "fcsrl_head_hidden_dim": task.get("fcsrl_head_hidden_dim"),
        }
    )
    expected_run = dict(base_run)
    expected_run["seed"] = task.get("seed")
    expected_run["device"] = task.get("device")
    expected_evaluation = dict(base_evaluation)
    if task.get("safety_arm") == PROFILE_ARM:
        # The profile trainer deliberately disables the ordinary trainer's audit writer;
        # its separately authenticated post-fit validation audit remains mandatory below.
        expected_evaluation["emit_audit_records"] = False

    found_run = dict(run)
    # The actual output is CLI-bound and authenticated by task.output_dir.  Generic runs retain
    # the base value here, while the profile resolver records its semantic-arm path.
    found_run.pop("output_dir", None)
    expected_run.pop("output_dir", None)
    exact_sections = {
        "data": (data, expected_data),
        "model": (model, expected_model),
        "objective": (objective, expected_objective),
        "run_without_output_dir": (found_run, expected_run),
        "evaluation": (evaluation, expected_evaluation),
    }
    mismatches = {
        field: {"found": found, "expected": expected}
        for field, (found, expected) in exact_sections.items()
        if found != expected
    }
    if mismatches:
        raise PreconfirmationArtifactError(
            "resolved config disagrees with planned task: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _postfit_manifest(
    path: Path,
    *,
    task: Mapping[str, object],
    checkpoint_sha256: str,
    resolved_values: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = _load_object(path, field="validation post-fit manifest")
    manifest_sha = _self_hash(
        payload,
        hash_field="manifest_sha256",
        field="validation post-fit manifest",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "validation_postfit_audit_success"
        or payload.get("semantic_arm") != task.get("safety_arm")
        or payload.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise PreconfirmationArtifactError(
            "validation post-fit manifest identity/checkpoint mismatch"
        )
    split_access = _object(payload.get("split_access"), field="postfit.split_access")
    if (
        split_access.get("materialized") != ["validation"]
        or split_access.get("not_materialized") != ["calibration", "test"]
        or split_access.get("checkpoint_selection_complete_before_profile_targets")
        is not True
    ):
        raise PreconfirmationArtifactError(
            "validation post-fit manifest violates split isolation"
        )
    physical = _object(
        payload.get("physical_profile_target"), field="postfit.physical_profile_target"
    )
    resolved_data = _object(
        resolved_values.get("data"), field="resolved_config.values.data"
    )
    if (
        physical.get("source") != "explicit_observed_physical_rollouts"
        or physical.get("constant_action_grid") != resolved_data.get("actions")
        or physical.get("profile_rollout_horizon")
        != resolved_data.get("action_profile_horizon")
        or physical.get("action_safety_profile_helper_called") is not False
        or physical.get("used_for_fitting_or_checkpoint_selection") is not False
    ):
        raise PreconfirmationArtifactError(
            "post-fit physical profiles are not isolated evaluation-only observations"
        )
    return payload, manifest_sha


def _utility_summary(
    postfit: dict[str, Any], resolved_values: dict[str, Any]
) -> dict[str, float | int]:
    utility = _object(postfit.get("e2_utility_summary"), field="postfit.e2_utility_summary")
    reconstruction = _finite(
        utility.get("reconstruction_mse"),
        field="postfit.e2_utility_summary.reconstruction_mse",
        nonnegative=True,
    )
    rollout = _finite(
        utility.get("maximum_horizon_rollout_pixel_mse"),
        field="postfit.e2_utility_summary.maximum_horizon_rollout_pixel_mse",
        nonnegative=True,
    )
    maximum_horizon = _integer(
        utility.get("maximum_horizon"),
        field="postfit.e2_utility_summary.maximum_horizon",
        minimum=1,
    )
    ordinary = _object(
        postfit.get("ordinary_validation_metrics"),
        field="postfit.ordinary_validation_metrics",
    )
    ordinary_reconstruction = _finite(
        ordinary.get("reconstruction"),
        field="postfit.ordinary_validation_metrics.reconstruction",
        nonnegative=True,
    )
    evaluation = _object(
        resolved_values.get("evaluation"), field="resolved_config.values.evaluation"
    )
    horizons = evaluation.get("rollout_horizons")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in horizons
        )
    ):
        raise PreconfirmationArtifactError(
            "resolved evaluation.rollout_horizons must be positive integers"
        )
    expected_horizon = max(horizons)
    rollout_metrics = _object(
        postfit.get("rollout_metrics"), field="postfit.rollout_metrics"
    )
    maximum_metrics = _object(
        rollout_metrics.get(str(expected_horizon)),
        field=f"postfit.rollout_metrics.{expected_horizon}",
    )
    reported_rollout = _finite(
        maximum_metrics.get("pixel_mse"),
        field=f"postfit.rollout_metrics.{expected_horizon}.pixel_mse",
        nonnegative=True,
    )
    if (
        maximum_horizon != expected_horizon
        or reconstruction != ordinary_reconstruction
        or rollout != reported_rollout
    ):
        raise PreconfirmationArtifactError(
            "E2 utility summary disagrees with authenticated validation metrics"
        )
    return {
        "reconstruction": reconstruction,
        "rollout": rollout,
        "maximum_horizon": maximum_horizon,
    }


def _profile_teacher_task_hash(task: dict[str, Any]) -> str:
    unsigned = dict(task)
    unsigned.pop("task_sha256", None)
    return canonical_sha256(unsigned)


def _validate_profile_teacher_plan(
    repo_root: Path,
    teacher: dict[str, Any],
) -> dict[tuple[str, str, int, int], str]:
    data_seeds = (*PILOT_SEEDS, *range(100, 108))
    expected_specs = list(
        itertools.product(DOMAINS, MODEL_FAMILIES, data_seeds, range(FOLD_COUNT))
    )
    if (
        teacher.get("experiment") != "predicted_profile_crossfit_teacher_preflight"
        or teacher.get("status")
        != "pipeline_components_ready_production_cells_and_gates_unexecuted"
        or teacher.get("task_formula")
        != "3 domains x 2 families x 11 data seeds x 5 folds"
        or teacher.get("task_count") != len(expected_specs)
        or teacher.get("domains") != list(DOMAINS)
        or teacher.get("model_families") != list(MODEL_FAMILIES)
        or teacher.get("data_seeds") != list(data_seeds)
        or teacher.get("folds") != list(range(FOLD_COUNT))
    ):
        raise PreconfirmationArtifactError(
            "profile teacher plan axes/counts drifted from the frozen protocol"
        )
    base_configs = _object(
        teacher.get("base_configs"), field="profile teacher plan.base_configs"
    )
    if set(base_configs) != set(DOMAINS):
        raise PreconfirmationArtifactError(
            "profile teacher plan must pin one base config per domain"
        )
    fold_memberships: dict[tuple[str, int, int], tuple[str, ...]] = {}
    all_training_ids: dict[tuple[str, int], tuple[str, ...]] = {}
    for domain in DOMAINS:
        base_entry = _object(
            base_configs.get(domain),
            field=f"profile teacher plan.base_configs.{domain}",
        )
        base_path = _repo_path(
            repo_root,
            base_entry.get("path"),
            field=f"profile teacher plan.base_configs.{domain}.path",
        )
        base_sha = _sha(
            base_entry.get("sha256"),
            field=f"profile teacher plan.base_configs.{domain}.sha256",
        )
        if not base_path.is_file() or file_sha256(base_path) != base_sha:
            raise PreconfirmationArtifactError(
                f"profile teacher plan base config changed for {domain}"
            )
        try:
            base_config = load_config(base_path)
        except (OSError, ValueError) as error:
            raise PreconfirmationArtifactError(
                f"cannot load profile teacher base config for {domain}: {error}"
            ) from error
        if base_config.data.task != domain:
            raise PreconfirmationArtifactError(
                f"profile teacher base config task mismatch for {domain}"
            )
        for data_seed in data_seeds:
            split_assignment = trajectory_split_assignment(
                base_config.data, seed=data_seed
            )
            training_ids = tuple(
                sorted(
                    trajectory_id
                    for trajectory_id, split in split_assignment.items()
                    if split == "train"
                )
            )
            assignments = assign_trajectory_folds(
                training_ids,
                fold_count=FOLD_COUNT,
                seed=FOLD_SEED,
            )
            all_training_ids[(domain, data_seed)] = training_ids
            for fold in range(FOLD_COUNT):
                membership = tuple(
                    sorted(
                        trajectory_id
                        for trajectory_id, assigned_fold in assignments.items()
                        if assigned_fold == fold
                    )
                )
                if not membership:
                    raise PreconfirmationArtifactError(
                        "frozen profile fold assignment produced an empty fold"
                    )
                fold_memberships[(domain, data_seed, fold)] = membership
    tasks = teacher.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(expected_specs):
        raise PreconfirmationArtifactError(
            "profile teacher plan task inventory is incomplete"
        )
    hashes: dict[tuple[str, str, int, int], str] = {}
    for index, (row_value, spec) in enumerate(zip(tasks, expected_specs, strict=True)):
        row = _object(row_value, field=f"profile teacher plan.tasks[{index}]")
        domain, family, data_seed, fold = spec
        base_entry = _object(
            base_configs.get(domain),
            field=f"profile teacher plan.base_configs.{domain}",
        )
        expected = {
            "task_id": index,
            "domain": domain,
            "model_family": family,
            "data_seed": data_seed,
            "fold_index": fold,
            "teacher_seed": teacher_seed(data_seed, fold),
            "base_config": base_entry.get("path"),
            "base_config_sha256": base_entry.get("sha256"),
            "history_mode": "stack_h4",
            "latent_dim": 8,
            "hidden_dim": 128,
            "transition_hidden_dim": 192,
            "objective": "h_prediction",
            "safety_weight": 0.5,
            "epochs": 30,
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise PreconfirmationArtifactError(
                f"profile teacher task {index} numeric/identity contract drifted"
            )
        held_out_ids = fold_memberships[(domain, data_seed, fold)]
        held_out_set = set(held_out_ids)
        fitting_ids = tuple(
            trajectory_id
            for trajectory_id in all_training_ids[(domain, data_seed)]
            if trajectory_id not in held_out_set
        )
        split_contract = {
            "training_trajectory_count": len(fitting_ids),
            "training_trajectory_ids_sha256": ids_sha256(fitting_ids),
            "held_out_trajectory_count": len(held_out_ids),
            "held_out_trajectory_ids_sha256": ids_sha256(held_out_ids),
        }
        if any(row.get(field) != value for field, value in split_contract.items()):
            raise PreconfirmationArtifactError(
                f"profile teacher task {index} fold membership disagrees with "
                f"FOLD_SEED={FOLD_SEED} and FOLD_COUNT={FOLD_COUNT}"
            )
        declared = _sha(
            row.get("task_sha256"),
            field=f"profile teacher plan.tasks[{index}].task_sha256",
        )
        if not hmac.compare_digest(declared, _profile_teacher_task_hash(row)):
            raise PreconfirmationArtifactError(
                f"profile teacher task {index} self-hash mismatch"
            )
        hashes[(domain, family, data_seed, fold)] = declared
    return hashes


def _profile_handoff(
    repo_root: Path,
    task: Mapping[str, object],
    run: dict[str, Any],
    checkpoint: dict[str, Any],
    resolved_values: dict[str, Any],
) -> dict[str, Any]:
    teacher_path = _repo_path(
        repo_root, task.get("profile_teacher_plan"), field="task.profile_teacher_plan"
    )
    labels_path = _repo_path(
        repo_root, task.get("profile_label_manifest"), field="task.profile_label_manifest"
    )
    coverage_path = _repo_path(
        repo_root,
        task.get("profile_coverage_manifest"),
        field="task.profile_coverage_manifest",
    )
    teacher = _load_object(teacher_path, field="profile teacher plan")
    labels = _load_object(labels_path, field="profile label assembly")
    coverage = _load_object(coverage_path, field="profile coverage gate")
    teacher_sha = _self_hash(teacher, hash_field="plan_sha256", field="profile teacher plan")
    label_sha = _self_hash(labels, hash_field="manifest_sha256", field="profile label assembly")
    coverage_sha = _self_hash(
        coverage, hash_field="manifest_sha256", field="profile coverage gate"
    )
    code = teacher.get("code")
    if (
        teacher.get("schema_version") != 2
        or teacher.get("task_count") != 330
        or not isinstance(code, dict)
        or not isinstance(code.get("git_revision"), str)
        or not code.get("git_revision")
        or code.get("git_dirty") is not False
    ):
        raise PreconfirmationArtifactError(
            "profile teacher plan is not the clean canonical 330-task plan"
        )
    planned_teacher_hashes = _validate_profile_teacher_plan(repo_root, teacher)
    teacher_base_configs = _object(
        teacher.get("base_configs"), field="profile teacher plan.base_configs"
    )
    teacher_base = _object(
        teacher_base_configs.get(str(task.get("domain"))),
        field="profile teacher plan current-domain base config",
    )
    if teacher_base != {
        "path": task.get("base_config"),
        "sha256": task.get("base_config_sha256"),
    }:
        raise PreconfirmationArtifactError(
            "profile teacher plan base config disagrees with the E2 task"
        )
    pinned = {
        "teacher_plan_sha256": (teacher_sha, task.get("profile_teacher_plan_sha256")),
        "teacher_plan_file_sha256": (
            file_sha256(teacher_path),
            task.get("profile_teacher_plan_file_sha256"),
        ),
        "label_manifest_sha256": (label_sha, task.get("profile_label_manifest_sha256")),
        "label_manifest_file_sha256": (
            file_sha256(labels_path),
            task.get("profile_label_manifest_file_sha256"),
        ),
        "coverage_manifest_sha256": (
            coverage_sha,
            task.get("profile_coverage_manifest_sha256"),
        ),
        "coverage_manifest_file_sha256": (
            file_sha256(coverage_path),
            task.get("profile_coverage_manifest_file_sha256"),
        ),
    }
    for field, (actual, expected) in pinned.items():
        if not isinstance(expected, str) or not hmac.compare_digest(actual, expected):
            raise PreconfirmationArtifactError(f"profile {field} disagrees with the plan")
    identity = {
        "domain": task.get("domain"),
        "model_family": task.get("model_family"),
        "data_seed": task.get("seed"),
    }
    for name, payload in (("assembly", labels), ("coverage", coverage)):
        if any(payload.get(field) != expected for field, expected in identity.items()):
            raise PreconfirmationArtifactError(f"profile {name} identity mismatch")
        if payload.get("plan_sha256") != teacher_sha:
            raise PreconfirmationArtifactError(f"profile {name} teacher-plan mismatch")
    if (
        labels.get("schema_version") != 1
        or labels.get("protocol_version") != PROFILE_PROTOCOL_VERSION
        or labels.get("status") != "complete_crossfit_label_manifest"
        or labels.get("semantic_arm") != PROFILE_ARM
        or labels.get("engineering_smoke") is not False
        or labels.get("evidence_eligible") is not True
        or labels.get("base_config_sha256") != task.get("base_config_sha256")
    ):
        raise PreconfirmationArtifactError("profile label assembly is not evidence-ready")
    teachers = labels.get("teachers")
    if (
        not isinstance(teachers, list)
        or len(teachers) != FOLD_COUNT
        or {item.get("fold_index") for item in teachers if isinstance(item, dict)}
        != set(range(FOLD_COUNT))
    ):
        raise PreconfirmationArtifactError("profile assembly must authenticate five fold teachers")
    teacher_hashes: dict[int, str] = {}
    for item in teachers:
        assert isinstance(item, dict)
        fold = int(item["fold_index"])
        teacher_hashes[fold] = _sha(
            item.get("task_sha256"), field=f"profile assembly teacher {fold} task_sha256"
        )
        expected_teacher_hash = planned_teacher_hashes.get(
            (
                str(task.get("domain")),
                str(task.get("model_family")),
                int(task.get("seed")),
                fold,
            )
        )
        if teacher_hashes[fold] != expected_teacher_hash:
            raise PreconfirmationArtifactError(
                "profile assembly teacher hash disagrees with the teacher plan"
            )
    data_values = _object(resolved_values.get("data"), field="resolved_config.values.data")
    objective_values = _object(
        resolved_values.get("objective"), field="resolved_config.values.objective"
    )
    action_grid = data_values.get("actions")
    if not isinstance(action_grid, list) or not action_grid:
        raise PreconfirmationArtifactError("resolved profile action grid is missing")
    normalized_actions = [
        _finite(value, field="resolved profile action", nonnegative=False)
        for value in action_grid
    ]
    profile_horizon = _integer(
        data_values.get("action_profile_horizon"),
        field="resolved profile action_profile_horizon",
        minimum=1,
    )
    margin_scale = _finite(
        objective_values.get("margin_scale"),
        field="resolved profile margin_scale",
        nonnegative=True,
    )
    if margin_scale <= 0.0:
        raise PreconfirmationArtifactError("resolved profile margin_scale must be positive")
    coverage_identity = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "engineering_smoke": False,
        "action_grid": normalized_actions,
        "profile_rollout_horizon": profile_horizon,
        "margin_scale": margin_scale,
        "required_normalized_p95": REQUIRED_NORMALIZED_P95,
    }
    if any(coverage.get(field) != expected for field, expected in coverage_identity.items()):
        raise PreconfirmationArtifactError("profile coverage protocol identity mismatch")
    gate = _object(coverage.get("gate"), field="profile coverage.gate")
    bundle_count = _integer(
        gate.get("bundle_count"), field="profile coverage.gate.bundle_count", minimum=1
    )
    if (
        bundle_count != COVERAGE_BUNDLES_PER_DOMAIN_SEED
        or coverage.get("bundle_count") != bundle_count
    ):
        raise PreconfirmationArtifactError(
            f"profile coverage must contain exactly {COVERAGE_BUNDLES_PER_DOMAIN_SEED} bundles"
        )
    bundles = coverage.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != bundle_count:
        raise PreconfirmationArtifactError("profile coverage bundle inventory is incomplete")
    predictions: dict[str, tuple[float, ...]] = {}
    targets: dict[str, tuple[float, ...]] = {}
    fold_counts = {fold: 0 for fold in range(FOLD_COUNT)}
    for index, bundle_value in enumerate(bundles):
        bundle = _object(bundle_value, field=f"profile coverage.bundles[{index}]")
        bundle_id = bundle.get("bundle_id")
        fold = bundle.get("teacher_fold")
        prediction = bundle.get("prediction")
        target = bundle.get("observed_target")
        if (
            not isinstance(bundle_id, str)
            or not bundle_id.startswith(f"coverage-{task.get('domain')}-seed-")
            or bundle_id in predictions
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in fold_counts
            or bundle.get("teacher_task_sha256") != teacher_hashes[fold]
        ):
            raise PreconfirmationArtifactError("profile coverage bundle identity is invalid")
        converted: list[tuple[float, ...]] = []
        for vector, name in ((prediction, "prediction"), (target, "target")):
            if not isinstance(vector, list) or len(vector) != len(normalized_actions):
                raise PreconfirmationArtifactError(
                    f"profile coverage bundle {name} shape mismatch"
                )
            converted.append(
                tuple(
                    _finite(value, field=f"profile coverage bundle {name}")
                    for value in vector
                )
            )
        predictions[bundle_id], targets[bundle_id] = converted
        expected_error = max(
            abs(left - right)
            for left, right in zip(predictions[bundle_id], targets[bundle_id], strict=True)
        ) / margin_scale
        reported_error = _finite(
            bundle.get("normalized_max_error"),
            field="profile coverage bundle normalized_max_error",
            nonnegative=True,
        )
        if not math.isclose(reported_error, expected_error, rel_tol=0.0, abs_tol=1e-12):
            raise PreconfirmationArtifactError("profile coverage bundle error mismatch")
        fold_counts[fold] += 1
    if (
        coverage.get("bundle_ids_sha256") != ids_sha256(sorted(predictions))
        or set(fold_counts.values()) != {bundle_count // FOLD_COUNT}
    ):
        raise PreconfirmationArtifactError(
            "profile coverage bundle checksum or teacher balance mismatch"
        )
    recomputed_gate = evaluate_profile_gate(
        predictions,
        targets,
        margin_scale=margin_scale,
        required_max=REQUIRED_NORMALIZED_P95,
    )
    if gate != json.loads(json.dumps(recomputed_gate.to_dict())):
        raise PreconfirmationArtifactError("profile coverage aggregate error metrics mismatch")
    normalized_p95 = recomputed_gate.normalized_p95_error
    if (
        coverage.get("status") != "gate_passed"
        or coverage.get("scientific_gate_passed") is not True
        or coverage.get("evidence_eligible") is not True
        or gate.get("passed") is not True
    ):
        raise PreconfirmationArtifactError("profile coverage gate is not evidence-ready")
    assembly_link = _object(coverage.get("assembly"), field="profile coverage.assembly")
    if (
        assembly_link.get("manifest_sha256") != label_sha
        or assembly_link.get("file_sha256") != file_sha256(labels_path)
    ):
        raise PreconfirmationArtifactError("profile coverage does not bind the label assembly")
    if (
        run.get("teacher_plan_sha256") != teacher_sha
        or _object(run.get("label_manifest"), field="profile run.label_manifest").get(
            "manifest_sha256"
        )
        != label_sha
        or _object(run.get("coverage_gate"), field="profile run.coverage_gate").get(
            "manifest_sha256"
        )
        != coverage_sha
        or run.get("evidence_eligible") is not True
    ):
        raise PreconfirmationArtifactError("profile run handoff/evidence identity mismatch")
    for field, expected in (
        ("teacher_plan_sha256", teacher_sha),
        ("label_manifest_sha256", label_sha),
        ("coverage_manifest_sha256", coverage_sha),
    ):
        if checkpoint.get(field) != expected:
            raise PreconfirmationArtifactError(f"profile checkpoint {field} mismatch")
    return {
        "teacher_plan": {
            "path": teacher_path.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(teacher_path),
            "plan_sha256": teacher_sha,
        },
        "label_assembly": {
            "path": labels_path.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(labels_path),
            "manifest_sha256": label_sha,
        },
        "coverage_gate": {
            "path": coverage_path.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(coverage_path),
            "manifest_sha256": coverage_sha,
            "bundle_count": bundle_count,
            "normalized_p95_error": normalized_p95,
        },
        "teacher_count": FOLD_COUNT,
        "teacher_failure_count": 0,
    }


def build_preconfirmation_task_manifest(
    *,
    repo_root: Path,
    plan_path: Path,
    plan_sha256: str,
    task: Mapping[str, object],
) -> dict[str, Any]:
    """Reopen a completed task and build its deterministic authenticated handoff."""

    repo_root = repo_root.resolve()
    plan_path = plan_path.resolve()
    _sha(plan_sha256, field="plan_sha256")
    if not plan_path.is_file():
        raise PreconfirmationArtifactError(f"source plan is missing: {plan_path}")
    plan = _load_object(plan_path, field="source preconfirmation plan")
    declared_plan_sha = _self_hash(
        plan, hash_field="plan_sha256", field="source preconfirmation plan"
    )
    if not hmac.compare_digest(declared_plan_sha, plan_sha256):
        raise PreconfirmationArtifactError("source preconfirmation plan SHA mismatch")
    plan_code = _object(plan.get("code"), field="source preconfirmation plan.code")
    run_dir = _repo_path(repo_root, task.get("output_dir"), field="task.output_dir")
    if not run_dir.is_dir():
        raise PreconfirmationArtifactError(f"task output directory is missing: {run_dir}")
    profile = task.get("safety_arm") == PROFILE_ARM

    run_path = run_dir / "run_manifest.json"
    dataset_path = run_dir / "dataset_manifest.json"
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    evaluation_path = run_dir / "evaluation_manifest.json"
    postfit_path = run_dir / "validation_postfit_manifest.json"
    audit_path = run_dir / "audit_validation.jsonl"
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    required = (
        run_path,
        dataset_path,
        checkpoint_manifest_path,
        postfit_path,
        audit_path,
        best_path,
        last_path,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise PreconfirmationArtifactError(
            f"completed task is missing required artifacts: {missing!r}"
        )
    if profile and evaluation_path.exists():
        raise PreconfirmationArtifactError(
            "profile task unexpectedly contains an unauthenticated evaluation manifest"
        )
    if not profile and not evaluation_path.is_file():
        raise PreconfirmationArtifactError("ordinary task is missing evaluation_manifest.json")

    run = _load_object(run_path, field="run manifest")
    dataset = _load_object(dataset_path, field="dataset manifest")
    checkpoint = _load_object(checkpoint_manifest_path, field="checkpoint manifest")
    if run.get("status") != "success" or checkpoint.get("status") != "success":
        raise PreconfirmationArtifactError("run/checkpoint status is not success")
    if run.get("failure") not in (None, {}):
        raise PreconfirmationArtifactError("successful run retains a failure payload")
    if run.get("orchestration_plan_sha256") != plan_sha256:
        raise PreconfirmationArtifactError("run orchestration-plan SHA mismatch")
    if checkpoint.get("orchestration_plan_sha256") != plan_sha256:
        raise PreconfirmationArtifactError("checkpoint orchestration-plan SHA mismatch")
    if profile:
        if run.get("execution_code") != plan_code:
            raise PreconfirmationArtifactError("profile run code state disagrees with plan")
    elif run.get("code") != plan_code or checkpoint.get("code") != plan_code:
        raise PreconfirmationArtifactError("ordinary run/checkpoint code state disagrees with plan")
    if not profile:
        _base_config_reference(repo_root, run, task, field="run manifest")
        _base_config_reference(
            repo_root, checkpoint, task, field="checkpoint manifest"
        )

    resolved = _object(run.get("resolved_config"), field="run.resolved_config")
    values = _object(resolved.get("values"), field="run.resolved_config.values")
    resolved_sha = _sha(resolved.get("sha256"), field="run.resolved_config.sha256")
    if canonical_sha256(values) != resolved_sha:
        raise PreconfirmationArtifactError("resolved-config SHA mismatch")
    _check_task_config(repo_root, values, task)
    if checkpoint.get("resolved_config_sha256") != resolved_sha:
        raise PreconfirmationArtifactError("checkpoint resolved-config SHA mismatch")

    dataset_sha = canonical_sha256(dataset)
    if profile:
        if run.get("dataset_manifest_sha256") != dataset_sha:
            raise PreconfirmationArtifactError("profile run dataset-manifest SHA mismatch")
        isolation = _object(run.get("isolation"), field="profile run.isolation")
        if (
            isolation.get("calibration_materialized") is not False
            or isolation.get("final_test_materialized") is not False
            or isolation.get("oracle_action_safety_margins_forbidden") is not True
        ):
            raise PreconfirmationArtifactError("profile run violates split/oracle isolation")
        if dataset.get("not_materialized") != ["calibration", "test"]:
            raise PreconfirmationArtifactError("profile dataset materialized a held-out split")
        if dataset.get("oracle_action_profiles_materialized") is not False:
            raise PreconfirmationArtifactError("profile dataset materialized oracle profiles")
        if run.get("semantic_arm") != task.get("safety_arm"):
            raise PreconfirmationArtifactError("profile run semantic-arm mismatch")
    else:
        _require_exact_scope(run.get("data_access_scope"), field="run.data_access_scope")
        run_dataset = _object(run.get("dataset"), field="run.dataset")
        if run_dataset.get("manifest_sha256") != dataset_sha:
            raise PreconfirmationArtifactError("run dataset-manifest SHA mismatch")
        access = _object(dataset.get("access_scope"), field="dataset.access_scope")
        if access != {
            "materialized_splits": ["train", "validation"],
            "not_materialized_splits": ["calibration", "test"],
            "oracle_action_profiles_materialized": False,
        }:
            raise PreconfirmationArtifactError("dataset violates train/validation isolation")
        if run.get("training_arm") != task.get("safety_arm"):
            raise PreconfirmationArtifactError("run training-arm mismatch")
    if checkpoint.get("dataset_manifest_sha256") != dataset_sha:
        raise PreconfirmationArtifactError("checkpoint dataset-manifest SHA mismatch")

    best_sha = file_sha256(best_path)
    last_sha = file_sha256(last_path)
    checkpoint_files = _object(checkpoint.get("checkpoints"), field="checkpoint.checkpoints")
    for name, path, digest in (
        ("best", best_path, best_sha),
        ("last", last_path, last_sha),
    ):
        entry = _object(checkpoint_files.get(name), field=f"checkpoint.checkpoints.{name}")
        if _run_reference(run_dir, entry.get("path"), field=f"checkpoint.{name}.path") != path:
            raise PreconfirmationArtifactError(f"checkpoint {name} path mismatch")
        if entry.get("sha256") != digest:
            raise PreconfirmationArtifactError(f"checkpoint {name} file SHA mismatch")

    postfit, postfit_manifest_sha = _postfit_manifest(
        postfit_path,
        task=task,
        checkpoint_sha256=best_sha,
        resolved_values=values,
    )
    postfit_file_sha = file_sha256(postfit_path)
    checkpoint_postfit = _object(
        checkpoint.get("postfit_validation_audit"), field="checkpoint.postfit_validation_audit"
    )
    if (
        _run_reference(
            run_dir,
            checkpoint_postfit.get("path"),
            field="checkpoint.postfit_validation_audit.path",
        )
        != postfit_path
        or checkpoint_postfit.get("sha256") != postfit_file_sha
        or checkpoint_postfit.get("manifest_sha256") != postfit_manifest_sha
        or checkpoint_postfit.get("used_for_fitting_or_checkpoint_selection") is not False
    ):
        raise PreconfirmationArtifactError("checkpoint/post-fit audit provenance mismatch")

    audit_entry = _object(postfit.get("audit_records"), field="postfit.audit_records")
    audit_sha = file_sha256(audit_path)
    records = read_jsonl(audit_path)
    if (
        _run_reference(run_dir, audit_entry.get("path"), field="postfit.audit_records.path")
        != audit_path
        or audit_entry.get("sha256") != audit_sha
        or audit_entry.get("record_count") != len(records)
        or postfit.get("sample_count") != len(records)
    ):
        raise PreconfirmationArtifactError("post-fit validation-record provenance mismatch")
    if {record.split for record in records} != {"validation"}:
        raise PreconfirmationArtifactError("post-fit audit records are not validation-only")
    resolved_data = _object(values.get("data"), field="resolved_config.values.data")
    actions = resolved_data.get("actions")
    if not isinstance(actions, list) or not actions or any(
        record.action_safety_margins is None
        or len(record.action_safety_margins) != len(actions)
        for record in records
    ):
        raise PreconfirmationArtifactError(
            "post-fit audit action-profile width disagrees with the resolved action grid"
        )

    evaluation_entry: dict[str, str] | None = None
    if not profile:
        evaluation = _load_object(evaluation_path, field="evaluation manifest")
        if evaluation.get("status") != "success":
            raise PreconfirmationArtifactError("evaluation status is not success")
        if evaluation.get("code") != plan_code:
            raise PreconfirmationArtifactError("evaluation code state disagrees with plan")
        _base_config_reference(
            repo_root, evaluation, task, field="evaluation manifest"
        )
        _require_exact_scope(
            evaluation.get("data_access_scope"), field="evaluation.data_access_scope"
        )
        if (
            evaluation.get("orchestration_plan_sha256") != plan_sha256
            or evaluation.get("resolved_config_sha256") != resolved_sha
            or evaluation.get("dataset_manifest_sha256") != dataset_sha
            or evaluation.get("checkpoint_sha256") != best_sha
        ):
            raise PreconfirmationArtifactError("evaluation provenance hash mismatch")
        if set(_object(evaluation.get("split_metrics"), field="evaluation.split_metrics")) != {
            "train",
            "validation",
        }:
            raise PreconfirmationArtifactError("evaluation contains a held-out split metric")
        if set(_object(evaluation.get("rollout_metrics"), field="evaluation.rollout_metrics")) != {
            "validation"
        }:
            raise PreconfirmationArtifactError("evaluation contains a held-out rollout")
        audit_records = _object(evaluation.get("audit_records"), field="evaluation.audit_records")
        if set(audit_records) != {"validation"} or _run_reference(
            run_dir, audit_records["validation"], field="evaluation.audit_records.validation"
        ) != audit_path:
            raise PreconfirmationArtifactError("evaluation audit-record provenance mismatch")
        evaluation_postfit = _object(
            evaluation.get("postfit_validation_audit"),
            field="evaluation.postfit_validation_audit",
        )
        if (
            _run_reference(
                run_dir,
                evaluation_postfit.get("path"),
                field="evaluation.postfit_validation_audit.path",
            )
            != postfit_path
            or evaluation_postfit.get("sha256") != postfit_file_sha
            or evaluation_postfit.get("manifest_sha256") != postfit_manifest_sha
        ):
            raise PreconfirmationArtifactError("evaluation/post-fit provenance mismatch")
        if _run_reference(
            run_dir,
            evaluation.get("checkpoint_manifest"),
            field="evaluation.checkpoint_manifest",
        ) != checkpoint_manifest_path:
            raise PreconfirmationArtifactError("evaluation checkpoint-manifest path mismatch")
        evaluation_entry = _artifact_entry(evaluation_path, run_dir)

    profile_handoff = (
        _profile_handoff(repo_root, task, run, checkpoint, values) if profile else None
    )
    utility = _utility_summary(postfit, values)
    task_payload = {
        field: task.get(field)
        for field in (
            "task_id",
            "task_sha256",
            "domain",
            "model_family",
            "history_mode",
            "history_encoder",
            "history_length",
            "latent_dim",
            "safety_arm",
            "seed",
            "regularization_weight",
            "safety_weight",
            "output_dir",
        )
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": TASK_HANDOFF_PROTOCOL,
        "status": "success",
        "plan": {
            "path": plan_path.relative_to(repo_root).as_posix(),
            "file_sha256": file_sha256(plan_path),
            "plan_sha256": plan_sha256,
        },
        "task": task_payload,
        "scope": {
            "selection_split": "validation_only",
            "materialized_splits": ["train", "validation"],
            "calibration_access": False,
            "final_test_access": False,
            "oracle_action_profiles_used_for_fitting_or_selection": False,
            "postfit_physical_profiles_evaluation_only": True,
        },
        "provenance": {
            "resolved_config_sha256": resolved_sha,
            "dataset_manifest_sha256": dataset_sha,
            "selected_checkpoint_sha256": best_sha,
            "validation_postfit_manifest_sha256": postfit_manifest_sha,
        },
        "utility_summary": utility,
        "artifacts": {
            "run_manifest": _artifact_entry(run_path, run_dir),
            "dataset_manifest": _artifact_entry(dataset_path, run_dir),
            "checkpoint_manifest": _artifact_entry(checkpoint_manifest_path, run_dir),
            "evaluation_manifest": evaluation_entry,
            "validation_postfit_manifest": _artifact_entry(postfit_path, run_dir),
            "validation_audit_records": {
                **_artifact_entry(audit_path, run_dir),
                "record_count": len(records),
            },
            "checkpoint_best": _artifact_entry(best_path, run_dir),
            "checkpoint_last": _artifact_entry(last_path, run_dir),
        },
        "profile_handoff": profile_handoff,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def write_preconfirmation_task_manifest(
    *,
    repo_root: Path,
    plan_path: Path,
    plan_sha256: str,
    task: Mapping[str, object],
) -> Path:
    """Build and atomically write the uniform task handoff, refusing overwrite."""

    output_dir = _repo_path(repo_root.resolve(), task.get("output_dir"), field="task.output_dir")
    path = output_dir / TASK_MANIFEST_FILENAME
    if path.exists():
        raise FileExistsError(f"refusing to overwrite task handoff manifest: {path}")
    payload = build_preconfirmation_task_manifest(
        repo_root=repo_root,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        task=task,
    )
    write_json_atomic(path, payload)
    return path


def authenticate_preconfirmation_task_manifest(
    *,
    repo_root: Path,
    plan_path: Path,
    plan_sha256: str,
    task: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    """Authenticate the on-disk envelope and independently rebuild its full contents."""

    output_dir = _repo_path(repo_root.resolve(), task.get("output_dir"), field="task.output_dir")
    path = output_dir / TASK_MANIFEST_FILENAME
    observed = _load_object(path, field="preconfirmation task manifest")
    _self_hash(
        observed,
        hash_field="manifest_sha256",
        field="preconfirmation task manifest",
    )
    expected = build_preconfirmation_task_manifest(
        repo_root=repo_root,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        task=task,
    )
    if observed != expected:
        raise PreconfirmationArtifactError(
            "preconfirmation task manifest does not match revalidated underlying artifacts"
        )
    return observed, path
