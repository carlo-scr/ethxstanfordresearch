"""Observed-rollout coverage-validation gate for predicted action profiles."""

from __future__ import annotations

import dataclasses
import hmac
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_safety.learning.config import LearningConfig
from latent_safety.learning.data import (
    PhysicalState,
    _history_indices,
    _render_trajectory_tensor,
    generate_trajectories,
    observed_constant_action_profile,
)
from latent_safety.learning.models import load_world_model_checkpoint
from latent_safety.learning.profile_artifacts import (
    ASSEMBLED_MANIFEST_FILENAME,
    ProfileArtifactError,
    canonical_sha256,
    file_sha256,
    ids_sha256,
    load_crossfit_label_index,
    verify_teacher_shard,
)
from latent_safety.learning.profile_protocol import (
    COVERAGE_BUNDLES_PER_DOMAIN_SEED,
    PROFILE_PROTOCOL_VERSION,
    REQUIRED_NORMALIZED_P95,
    assign_coverage_teachers,
    evaluate_profile_gate,
)
from latent_safety.learning.profile_teacher import (
    build_teacher_split_spec,
    predict_action_profiles_tensor,
    resolve_teacher_config,
)
from latent_safety.learning.runtime import require_torch, select_device
from latent_safety.manifest import write_json_atomic

COVERAGE_SEED_BASE = 70_000_000
COVERAGE_CENTER_TIMESTEP = 3
COVERAGE_MANIFEST_FILENAME = "profile_coverage_gate.json"
_DOMAIN_SEED_OFFSETS = {
    "controlled_cart_video": 0,
    "controlled_pendulum_video": 100_000,
    "controlled_dubins_navigation_pixels": 200_000,
}


class ProfileCoverageError(ValueError):
    """Raised when coverage bundles or teacher predictions violate the gate protocol."""


@dataclass(frozen=True)
class CoverageBundle:
    bundle_id: str
    scenario: str
    teacher_fold: int
    history: Any
    observed_action_profile: tuple[float, ...]


def coverage_seed(task: str, data_seed: int) -> int:
    """Return the family-independent, domain-separated coverage seed."""

    if task not in _DOMAIN_SEED_OFFSETS:
        raise ProfileCoverageError(f"unsupported coverage task: {task!r}")
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed < 0:
        raise ProfileCoverageError("data_seed must be a non-negative integer")
    return COVERAGE_SEED_BASE + _DOMAIN_SEED_OFFSETS[task] + data_seed


def _observed_constant_action_profile(
    state: PhysicalState,
    config: Any,
) -> tuple[float, ...]:
    """Execute each action branch and record observed margins at t=0,...,H."""

    return observed_constant_action_profile(state, config)


def build_coverage_bundles(
    modules: Any,
    config: LearningConfig,
    *,
    data_seed: int,
    bundle_count: int = COVERAGE_BUNDLES_PER_DOMAIN_SEED,
) -> tuple[CoverageBundle, ...]:
    """Build a disjoint family-independent bundle set from the deployment generator mix.

    The namespace and seed are separate from all four ordinary splits.  Each bundle uses the first
    four behavior-policy frames, branches at timestep three, and executes deterministic physical
    transitions under every constant action.  Targets are constructed from the margins observed on
    those explicit branches; ``action_safety_profile`` is never called.
    """

    if isinstance(bundle_count, bool) or not isinstance(bundle_count, int) or bundle_count < 5:
        raise ProfileCoverageError("bundle_count must be an integer of at least five")
    if config.data.history_length != 4:
        raise ProfileCoverageError("coverage protocol requires stack_h4 histories")
    prefix_horizon = max(config.data.history_length, COVERAGE_CENTER_TIMESTEP + 1)
    coverage_data = dataclasses.replace(
        config.data,
        trajectories=bundle_count,
        horizon=prefix_horizon,
    )
    seed = coverage_seed(config.data.task, data_seed)
    source = generate_trajectories(
        coverage_data,
        seed=seed,
        include_action_profiles=False,
    )
    if len(source) != bundle_count:
        raise ProfileCoverageError(
            f"coverage generator returned {len(source)} trajectories, expected {bundle_count}"
        )
    if any(trajectory.action_safety_margins for trajectory in source):
        raise ProfileCoverageError("coverage generator materialized oracle action profiles")
    bundle_ids = [
        f"coverage-{config.data.task}-seed-{data_seed:03d}-bundle-{index:04d}"
        for index in range(bundle_count)
    ]
    assignments = assign_coverage_teachers(bundle_ids)
    bundles: list[CoverageBundle] = []
    for bundle_id, trajectory in zip(bundle_ids, source, strict=True):
        frames = _render_trajectory_tensor(trajectory, coverage_data, modules.torch)
        history = frames[
            list(_history_indices(COVERAGE_CENTER_TIMESTEP, coverage_data.history_length))
        ]
        center = trajectory.states[COVERAGE_CENTER_TIMESTEP]
        target = _observed_constant_action_profile(center, coverage_data)
        if len(target) != len(coverage_data.actions) or any(
            not math.isfinite(value) for value in target
        ):
            raise ProfileCoverageError("coverage target is incomplete or non-finite")
        bundles.append(
            CoverageBundle(
                bundle_id=bundle_id,
                scenario=trajectory.scenario,
                teacher_fold=assignments[bundle_id],
                history=history,
                observed_action_profile=target,
            )
        )
    if len({bundle.bundle_id for bundle in bundles}) != bundle_count:
        raise ProfileCoverageError("coverage bundle IDs must be unique")
    fold_counts = Counter(bundle.teacher_fold for bundle in bundles)
    if set(fold_counts) != set(range(5)) or max(fold_counts.values()) - min(
        fold_counts.values()
    ) > 0:
        raise ProfileCoverageError("coverage bundles must route equally across five teachers")
    return tuple(bundles)


def load_profile_coverage_gate(
    manifest_path: Path,
    *,
    expected_domain: str,
    expected_model_family: str,
    expected_data_seed: int,
    expected_plan_sha256: str,
    expected_assembly_manifest_sha256: str,
    expected_action_grid: Sequence[float],
    expected_profile_horizon: int,
    expected_margin_scale: float,
    expected_teacher_task_sha256_by_fold: Mapping[int, str],
    allow_engineering_failure: bool = False,
) -> dict[str, Any]:
    """Recompute and authenticate a coverage gate before downstream arm training."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileCoverageError(f"cannot read coverage manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ProfileCoverageError("coverage manifest must contain an object")
    declared_sha = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual_sha = canonical_sha256(unsigned)
    if not isinstance(declared_sha, str) or not hmac.compare_digest(
        declared_sha, actual_sha
    ):
        raise ProfileCoverageError("coverage manifest SHA-256 mismatch")
    engineering = manifest.get("engineering_smoke") is True
    identity = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "plan_sha256": expected_plan_sha256,
        "domain": expected_domain,
        "model_family": expected_model_family,
        "data_seed": expected_data_seed,
        "action_grid": [float(value) for value in expected_action_grid],
        "profile_rollout_horizon": expected_profile_horizon,
        "margin_scale": float(expected_margin_scale),
        "required_normalized_p95": REQUIRED_NORMALIZED_P95,
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ProfileCoverageError(
            "coverage identity mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    assembly = manifest.get("assembly")
    if (
        not isinstance(assembly, dict)
        or assembly.get("manifest_sha256") != expected_assembly_manifest_sha256
    ):
        raise ProfileCoverageError("coverage manifest assembly identity mismatch")
    bundle_count = manifest.get("bundle_count")
    bundles = manifest.get("bundles")
    if (
        isinstance(bundle_count, bool)
        or not isinstance(bundle_count, int)
        or not isinstance(bundles, list)
        or len(bundles) != bundle_count
    ):
        raise ProfileCoverageError("coverage bundle count/inventory mismatch")
    if not engineering and bundle_count != COVERAGE_BUNDLES_PER_DOMAIN_SEED:
        raise ProfileCoverageError("production coverage manifest must contain 200 bundles")
    predictions: dict[str, tuple[float, ...]] = {}
    targets: dict[str, tuple[float, ...]] = {}
    seen_folds: Counter[int] = Counter()
    expected_teacher_hashes = {
        int(fold): str(task_sha256)
        for fold, task_sha256 in expected_teacher_task_sha256_by_fold.items()
    }
    if (
        set(expected_teacher_hashes) != set(range(5))
        or any(len(value) != 64 for value in expected_teacher_hashes.values())
    ):
        raise ProfileCoverageError("expected teacher inventory must cover folds 0 through 4")
    for row in bundles:
        if not isinstance(row, dict):
            raise ProfileCoverageError("coverage bundle entry must be an object")
        bundle_id = row.get("bundle_id")
        fold = row.get("teacher_fold")
        prediction = row.get("prediction")
        target = row.get("observed_target")
        if (
            not isinstance(bundle_id, str)
            or not bundle_id.startswith(f"coverage-{expected_domain}-seed-")
            or bundle_id in predictions
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or not 0 <= fold < 5
            or not isinstance(row.get("teacher_task_sha256"), str)
            or len(row["teacher_task_sha256"]) != 64
            or row["teacher_task_sha256"] != expected_teacher_hashes.get(fold)
        ):
            raise ProfileCoverageError("coverage bundle/teacher identity is invalid")
        converted = []
        for values in (prediction, target):
            if (
                not isinstance(values, list)
                or len(values) != len(expected_action_grid)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise ProfileCoverageError("coverage prediction/target vector is invalid")
            converted.append(tuple(float(value) for value in values))
        predictions[bundle_id], targets[bundle_id] = converted
        expected_error = max(
            abs(left - right)
            for left, right in zip(predictions[bundle_id], targets[bundle_id], strict=True)
        ) / expected_margin_scale
        reported_error = row.get("normalized_max_error")
        if (
            isinstance(reported_error, bool)
            or not isinstance(reported_error, (int, float))
            or not math.isclose(
                float(reported_error), expected_error, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ProfileCoverageError("coverage bundle normalized error mismatch")
        seen_folds[fold] += 1
    if manifest.get("bundle_ids_sha256") != ids_sha256(sorted(predictions)):
        raise ProfileCoverageError("coverage bundle ID checksum mismatch")
    if set(seen_folds) != set(range(5)) or max(seen_folds.values()) != min(
        seen_folds.values()
    ):
        raise ProfileCoverageError("coverage teacher assignment is incomplete or unbalanced")
    recomputed = evaluate_profile_gate(
        predictions,
        targets,
        margin_scale=expected_margin_scale,
        required_max=REQUIRED_NORMALIZED_P95,
    )
    if manifest.get("gate") != json.loads(json.dumps(recomputed.to_dict())):
        raise ProfileCoverageError("coverage aggregate gate metrics mismatch")
    passed = recomputed.passed
    if (
        manifest.get("scientific_gate_passed") is not passed
        or manifest.get("status") != ("gate_passed" if passed else "gate_failed")
    ):
        raise ProfileCoverageError("coverage status disagrees with recomputed gate")
    if (engineering or not passed) and not allow_engineering_failure:
        raise ProfileCoverageError(
            "downstream evidence training requires a passed production coverage gate"
        )
    manifest["verified_manifest_sha256"] = actual_sha
    return manifest


def run_profile_coverage_gate(
    tasks: Sequence[Mapping[str, object]],
    shard_dirs: Sequence[Path],
    *,
    plan_sha256: str,
    base_config: LearningConfig,
    assembled_manifest_path: Path,
    output_dir: Path,
    device: str,
    producing_command: Sequence[str],
    execution_code: Mapping[str, object],
    execution_evidence_eligible: bool,
    engineering_smoke: bool,
    bundle_count: int = COVERAGE_BUNDLES_PER_DOMAIN_SEED,
) -> dict[str, object]:
    """Execute one domain-family-seed coverage cell and persist every bundle decision."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite coverage output: {output_dir}")
    if len(tasks) != 5 or len(shard_dirs) != 5:
        raise ProfileCoverageError("coverage requires exactly five teacher tasks and shards")
    ordered = sorted(
        zip(tasks, shard_dirs, strict=True), key=lambda pair: int(pair[0]["fold_index"])
    )
    if [int(task["fold_index"]) for task, _ in ordered] != list(range(5)):
        raise ProfileCoverageError("coverage teacher folds must be exactly 0 through 4")
    resolved = resolve_teacher_config(
        base_config,
        ordered[0][0],
        device="cpu" if engineering_smoke else device,
        engineering_smoke=engineering_smoke,
    )
    split_spec = build_teacher_split_spec(
        resolved,
        data_seed=int(ordered[0][0]["data_seed"]),
        fold_index=0,
    )
    expected_training_samples = [
        f"{trajectory_id}:{timestep:04d}"
        for trajectory_id in split_spec.all_training_ids
        for timestep in range(resolved.data.horizon)
    ]
    label_index = load_crossfit_label_index(
        assembled_manifest_path,
        expected_domain=resolved.data.task,
        expected_model_family=resolved.model.family,
        expected_data_seed=int(ordered[0][0]["data_seed"]),
        expected_plan_sha256=plan_sha256,
        expected_base_config_sha256=str(ordered[0][0]["base_config_sha256"]),
        expected_action_grid=resolved.data.actions,
        expected_profile_horizon=resolved.data.action_profile_horizon,
        expected_margin_scale=resolved.objective.margin_scale,
        expected_sample_ids=expected_training_samples,
        allow_engineering_smoke=engineering_smoke,
    )
    verified_shards = tuple(
        verify_teacher_shard(
            task,
            plan_sha256=plan_sha256,
            base_config=base_config,
            output_dir=directory,
            engineering_smoke=engineering_smoke,
        )
        for task, directory in ordered
    )
    assembly = json.loads(assembled_manifest_path.read_text(encoding="utf-8"))
    teachers = assembly.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != 5:
        raise ProfileCoverageError("assembled manifest teacher inventory is incomplete")
    assembly_teachers = {int(entry["fold_index"]): entry for entry in teachers}
    for shard in verified_shards:
        entry = assembly_teachers.get(shard.fold_index)
        if (
            not isinstance(entry, dict)
            or entry.get("task_sha256") != shard.task_sha256
            or entry.get("handoff_manifest_sha256") != shard.handoff_manifest_sha256
            or entry.get("checkpoint_manifest_sha256")
            != shard.checkpoint_manifest_sha256
            or entry.get("best_checkpoint_sha256") != shard.best_checkpoint_sha256
        ):
            raise ProfileCoverageError("assembled teacher inventory disagrees with source shard")

    if not engineering_smoke and bundle_count != COVERAGE_BUNDLES_PER_DOMAIN_SEED:
        raise ProfileCoverageError(
            f"production coverage requires exactly {COVERAGE_BUNDLES_PER_DOMAIN_SEED} bundles"
        )
    modules = require_torch()
    selected_device, device_info = select_device(modules.torch, resolved.run.device)
    bundles = build_coverage_bundles(
        modules,
        resolved,
        data_seed=int(ordered[0][0]["data_seed"]),
        bundle_count=bundle_count,
    )
    models: dict[int, Any] = {}
    for (task, directory), shard in zip(ordered, verified_shards, strict=True):
        checkpoint_path = directory / "checkpoint_best.pt"
        if file_sha256(checkpoint_path) != shard.best_checkpoint_sha256:
            raise ProfileCoverageError("best checkpoint changed after shard verification")
        model, checkpoint_config, checkpoint = load_world_model_checkpoint(
            modules,
            checkpoint_path,
            device=selected_device,
        )
        expected_checkpoint_config = resolve_teacher_config(
            base_config,
            task,
            device=checkpoint_config.run.device,
            engineering_smoke=engineering_smoke,
        )
        if (
            checkpoint_config != expected_checkpoint_config
            or checkpoint.get("plan_sha256") != plan_sha256
            or checkpoint.get("task_sha256") != task["task_sha256"]
        ):
            raise ProfileCoverageError("loaded teacher checkpoint identity mismatch")
        models[shard.fold_index] = model

    predictions: dict[str, tuple[float, ...]] = {}
    targets = {
        bundle.bundle_id: bundle.observed_action_profile for bundle in bundles
    }
    with modules.torch.no_grad():
        for fold in range(5):
            assigned = [bundle for bundle in bundles if bundle.teacher_fold == fold]
            histories = modules.torch.stack([bundle.history for bundle in assigned], dim=0)
            profile_tensor = predict_action_profiles_tensor(
                modules,
                models[fold],
                histories,
                selected_device,
                actions=resolved.data.actions,
                horizon=resolved.data.action_profile_horizon,
            )
            profile_rows = profile_tensor.detach().cpu().tolist()
            for bundle, profile in zip(assigned, profile_rows, strict=True):
                predictions[bundle.bundle_id] = tuple(float(value) for value in profile)
    if set(predictions) != set(targets):
        raise ProfileCoverageError("coverage predictions do not cover every bundle exactly")
    gate = evaluate_profile_gate(
        predictions,
        targets,
        margin_scale=resolved.objective.margin_scale,
        required_max=REQUIRED_NORMALIZED_P95,
    )
    errors = {
        bundle_id: max(
            abs(prediction - target)
            for prediction, target in zip(
                predictions[bundle_id], targets[bundle_id], strict=True
            )
        )
        / resolved.objective.margin_scale
        for bundle_id in sorted(targets)
    }
    evidence_eligible = (
        not engineering_smoke
        and execution_evidence_eligible
        and label_index.evidence_eligible
        and all(shard.evidence_eligible for shard in verified_shards)
        and gate.passed
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "status": "gate_passed" if gate.passed else "gate_failed",
        "scientific_gate_passed": gate.passed,
        "engineering_smoke": engineering_smoke,
        "evidence_eligible": evidence_eligible,
        "plan_sha256": plan_sha256,
        "domain": resolved.data.task,
        "model_family": resolved.model.family,
        "data_seed": int(ordered[0][0]["data_seed"]),
        "assembly": {
            "path": ASSEMBLED_MANIFEST_FILENAME,
            "manifest_sha256": label_index.manifest_sha256,
            "file_sha256": file_sha256(assembled_manifest_path),
        },
        "generator": {
            "name": "coverage_validation_observed_rollouts_v1",
            "seed": coverage_seed(resolved.data.task, int(ordered[0][0]["data_seed"])),
            "seed_rule": (
                "70000000 + domain_offset{cart:0,pendulum:100000,dubins:200000} "
                "+ data_seed"
            ),
            "namespace": "coverage-{domain}-seed-{data_seed}-bundle-{index}",
            "family_independent": True,
            "deployment_scenario_mix": True,
            "history_length": 4,
            "center_timestep": COVERAGE_CENTER_TIMESTEP,
            "branch_process_noise": 0.0,
            "target": "minimum_observed_margin_over_explicit_t_0_through_H_rollout",
            "oracle_action_safety_profile_called": False,
        },
        "teacher_assignment": {
            "rule": "sorted bundle index modulo five",
            "ensemble": False,
            "fold_counts": {
                str(fold): sum(bundle.teacher_fold == fold for bundle in bundles)
                for fold in range(5)
            },
        },
        "action_grid": list(resolved.data.actions),
        "profile_rollout_horizon": resolved.data.action_profile_horizon,
        "margin_scale": resolved.objective.margin_scale,
        "required_normalized_p95": REQUIRED_NORMALIZED_P95,
        "bundle_count": len(bundles),
        "bundle_ids_sha256": ids_sha256([bundle.bundle_id for bundle in bundles]),
        "scenario_counts": dict(sorted(Counter(bundle.scenario for bundle in bundles).items())),
        "gate": gate.to_dict(),
        "bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "scenario": bundle.scenario,
                "teacher_fold": bundle.teacher_fold,
                "teacher_task_sha256": verified_shards[bundle.teacher_fold].task_sha256,
                "prediction": list(predictions[bundle.bundle_id]),
                "observed_target": list(targets[bundle.bundle_id]),
                "normalized_max_error": errors[bundle.bundle_id],
            }
            for bundle in bundles
        ],
        "device": {
            "selected": device_info.selected,
            "accelerator_name": device_info.accelerator_name,
        },
        "execution_code": dict(execution_code),
        "producing_command": list(producing_command),
        "remaining_scope": {
            "teacher_error_coverage_gate_complete": True,
            "latent_eligible_center_coverage_is_separate": True,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / COVERAGE_MANIFEST_FILENAME
    write_json_atomic(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "passed": gate.passed,
        "normalized_p95_error": gate.normalized_p95_error,
        "bundle_count": gate.bundle_count,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "evidence_eligible": evidence_eligible,
    }
