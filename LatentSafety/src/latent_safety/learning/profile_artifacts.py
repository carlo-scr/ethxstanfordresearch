"""Authenticated assembly and ingestion for cross-fitted predicted-profile labels."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from latent_safety.learning.config import LearningConfig, canonical_config_json, parse_config
from latent_safety.learning.data import _render_trajectory_tensor, generate_trajectories
from latent_safety.learning.profile_protocol import (
    FOLD_COUNT,
    PROFILE_PROTOCOL_VERSION,
    assign_trajectory_folds,
    build_crossfit_manifest,
)
from latent_safety.learning.profile_teacher import (
    ObservedMarginDataset,
    TeacherSplitSpec,
    build_teacher_split_spec,
    resolve_teacher_config,
    validate_task_split_hashes,
)
from latent_safety.manifest import write_json_atomic

PREDICTED_PROFILE_ARM = "nonprivileged_predicted_action_profile"
ASSEMBLED_LABEL_FILENAME = "crossfitted_profile_labels.jsonl"
ASSEMBLED_MANIFEST_FILENAME = "crossfit_label_manifest.json"


class ProfileArtifactError(ValueError):
    """Raised when a teacher or assembled-label artifact fails authentication."""


@dataclass(frozen=True)
class VerifiedTeacherShard:
    fold_index: int
    task_id: int
    task_sha256: str
    output_dir: Path
    split_spec: TeacherSplitSpec
    labels: tuple[dict[str, object], ...]
    checkpoint_manifest_sha256: str
    best_checkpoint_sha256: str
    label_shard_sha256: str
    label_manifest_sha256: str
    handoff_manifest_sha256: str
    evidence_eligible: bool


@dataclass(frozen=True)
class CrossfitLabelIndex:
    semantic_arm: str
    domain: str
    model_family: str
    data_seed: int
    action_grid: tuple[float, ...]
    profile_horizon: int
    margin_scale: float
    manifest_sha256: str
    evidence_eligible: bool
    profiles: Mapping[str, tuple[float, ...]]
    teacher_folds: Mapping[str, int]
    teacher_task_sha256_by_fold: Mapping[int, str]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.profiles))


def canonical_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    return canonical_sha256(sorted(ids))


def _json_compatible(payload: object) -> object:
    return json.loads(json.dumps(payload, sort_keys=True))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileArtifactError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProfileArtifactError(f"{label} must contain a JSON object: {path}")
    return payload


def _safe_child(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProfileArtifactError(f"{label}.path must be a non-empty relative string")
    declared = Path(value)
    if declared.is_absolute():
        raise ProfileArtifactError(f"{label}.path must be relative")
    resolved = (root / declared).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ProfileArtifactError(f"{label}.path escapes its artifact root") from error
    return resolved


def _verify_artifact_entry(
    root: Path,
    entry: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ProfileArtifactError(f"{label} must contain exactly path and sha256")
    path = _safe_child(root, entry.get("path"), label=label)
    declared = entry.get("sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ProfileArtifactError(f"{label}.sha256 must be a SHA-256 string")
    if not path.is_file():
        raise ProfileArtifactError(f"{label} file is missing: {path}")
    actual = file_sha256(path)
    if not hmac.compare_digest(declared, actual):
        raise ProfileArtifactError(
            f"{label}.sha256 mismatch: declared {declared}, recomputed {actual}"
        )
    return path, actual


def _parse_prediction_rows(
    path: Path,
    *,
    action_count: int,
    expected_sample_ids: set[str],
    expected_trajectory_ids: set[str],
    trajectory_horizon: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ProfileArtifactError(f"cannot read label shard {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProfileArtifactError(
                f"invalid JSON in {path.name} line {line_number}: {error}"
            ) from error
        required = {
            "sample_id",
            "trajectory_id",
            "timestep",
            "predicted_action_profile",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ProfileArtifactError(
                f"{path.name} line {line_number} must contain exactly {sorted(required)}"
            )
        sample_id = row["sample_id"]
        trajectory_id = row["trajectory_id"]
        timestep = row["timestep"]
        profile = row["predicted_action_profile"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ProfileArtifactError("label sample_id must be a non-empty string")
        if not isinstance(trajectory_id, str) or trajectory_id not in expected_trajectory_ids:
            raise ProfileArtifactError(f"unexpected label trajectory_id: {trajectory_id!r}")
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise ProfileArtifactError("label timestep must be an integer")
        if not 0 <= timestep < trajectory_horizon:
            raise ProfileArtifactError(f"label timestep is outside the trajectory: {timestep}")
        if sample_id != f"{trajectory_id}:{timestep:04d}":
            raise ProfileArtifactError("label sample_id disagrees with trajectory_id/timestep")
        if (
            not isinstance(profile, list)
            or len(profile) != action_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in profile
            )
        ):
            raise ProfileArtifactError(
                "predicted_action_profile must contain one finite number per action"
            )
        rows.append(
            {
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "timestep": timestep,
                "predicted_action_profile": [float(value) for value in profile],
            }
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ProfileArtifactError("label shard contains duplicate sample IDs")
    if set(sample_ids) != expected_sample_ids:
        missing = sorted(expected_sample_ids - set(sample_ids))
        unexpected = sorted(set(sample_ids) - expected_sample_ids)
        raise ProfileArtifactError(
            "label shard sample set mismatch; "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    return tuple(rows)


def verify_teacher_shard(
    task: Mapping[str, object],
    *,
    plan_sha256: str,
    base_config: LearningConfig,
    output_dir: Path,
    engineering_smoke: bool,
) -> VerifiedTeacherShard:
    """Authenticate one teacher handoff and its exact held-out sample partition."""

    fold_index = int(task["fold_index"])
    resolved = resolve_teacher_config(
        base_config,
        task,
        device="cpu" if engineering_smoke else base_config.run.device,
        engineering_smoke=engineering_smoke,
    )
    split_spec = build_teacher_split_spec(
        resolved,
        data_seed=int(task["data_seed"]),
        fold_index=fold_index,
    )
    if not engineering_smoke:
        validate_task_split_hashes(task, split_spec)
    expected_samples = {
        f"{trajectory_id}:{timestep:04d}"
        for trajectory_id in split_spec.held_out_ids
        for timestep in range(resolved.data.horizon)
    }

    handoff_path = output_dir / "teacher_handoff_manifest.json"
    handoff = _load_json(handoff_path, label="teacher handoff manifest")
    if (
        handoff.get("schema_version") != 1
        or handoff.get("protocol_version") != PROFILE_PROTOCOL_VERSION
        or handoff.get("status") != "single_teacher_success"
        or handoff.get("plan_sha256") != plan_sha256
        or handoff.get("task_id") != task["task_id"]
        or handoff.get("task_sha256") != task["task_sha256"]
        or handoff.get("engineering_smoke") is not engineering_smoke
    ):
        raise ProfileArtifactError(
            f"teacher handoff identity/status mismatch for fold {fold_index}"
        )
    if handoff.get("evidence_eligible") is not (not engineering_smoke):
        # Production tasks may be explicitly unversioned engineering runs.  Such runs remain
        # assemblable only as non-evidence, while smoke handoffs must never claim eligibility.
        if engineering_smoke or handoff.get("evidence_eligible") is not False:
            raise ProfileArtifactError("teacher handoff evidence eligibility is inconsistent")
    artifacts = handoff.get("artifacts")
    expected_artifacts = {
        "checkpoint_manifest",
        "crossfit_protocol_manifest",
        "dataset_manifest",
        "history",
        "label_shard_manifest",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ProfileArtifactError("teacher handoff artifact inventory is incomplete")
    verified_artifacts = {
        name: _verify_artifact_entry(output_dir, artifacts[name], label=name)
        for name in sorted(expected_artifacts)
    }

    run_manifest = _load_json(output_dir / "run_manifest.json", label="run manifest")
    run_handoff = run_manifest.get("handoff_manifest")
    if (
        run_manifest.get("status") != "success"
        or run_manifest.get("engineering_smoke") is not engineering_smoke
        or not isinstance(run_handoff, dict)
        or run_handoff.get("path") != handoff_path.name
        or run_handoff.get("sha256") != file_sha256(handoff_path)
        or run_manifest.get("plan", {}).get("sha256") != plan_sha256
        or run_manifest.get("plan", {}).get("task_sha256") != task["task_sha256"]
    ):
        raise ProfileArtifactError("run manifest does not authenticate the teacher handoff")
    resolved_manifest = run_manifest.get("resolved_config")
    if not isinstance(resolved_manifest, dict) or not isinstance(
        resolved_manifest.get("values"), dict
    ):
        raise ProfileArtifactError("teacher resolved-config provenance is missing")
    try:
        actual_resolved = parse_config(resolved_manifest["values"])
    except ValueError as error:
        raise ProfileArtifactError(f"teacher resolved config is invalid: {error}") from error
    resolved = resolve_teacher_config(
        base_config,
        task,
        device=actual_resolved.run.device,
        engineering_smoke=engineering_smoke,
    )
    expected_values = _json_compatible(resolved.to_dict())
    expected_resolved_sha = hashlib.sha256(
        canonical_config_json(resolved).encode("utf-8")
    ).hexdigest()
    if (
        not isinstance(resolved_manifest, dict)
        or resolved_manifest.get("values") != expected_values
        or resolved_manifest.get("sha256") != expected_resolved_sha
    ):
        raise ProfileArtifactError("teacher resolved-config provenance mismatch")

    dataset = _load_json(verified_artifacts["dataset_manifest"][0], label="dataset manifest")
    trajectory_sets = dataset.get("trajectory_sets")
    signals = dataset.get("signals")
    if (
        dataset.get("task") != task["domain"]
        or dataset.get("data_seed") != task["data_seed"]
        or dataset.get("fold_index") != fold_index
        or dataset.get("engineering_smoke") is not engineering_smoke
        or dataset.get("split_access")
        != {"materialized": ["train", "validation"], "not_materialized": ["calibration", "test"]}
        or not isinstance(signals, dict)
        or signals.get("oracle_action_profiles_materialized") is not False
        or signals.get("hidden_state_exposed_to_model") is not False
        or not isinstance(trajectory_sets, dict)
    ):
        raise ProfileArtifactError("teacher dataset access-control manifest mismatch")
    expected_sets = {
        "all_training": (len(split_spec.all_training_ids), split_spec.all_training_ids_sha256),
        "fitting": (len(split_spec.fitting_ids), split_spec.fitting_ids_sha256),
        "held_out": (len(split_spec.held_out_ids), split_spec.held_out_ids_sha256),
        "ordinary_validation": (
            len(split_spec.validation_ids),
            split_spec.validation_ids_sha256,
        ),
    }
    for name, (count, digest) in expected_sets.items():
        if trajectory_sets.get(name) != {"count": count, "sha256": digest}:
            raise ProfileArtifactError(f"dataset trajectory set mismatch: {name}")

    crossfit = _load_json(
        verified_artifacts["crossfit_protocol_manifest"][0],
        label="crossfit protocol manifest",
    )
    expected_crossfit, expected_crossfit_sha = build_crossfit_manifest(
        task=resolved.data.task,
        model_family=resolved.model.family,
        data_seed=int(task["data_seed"]),
        training_trajectory_ids=split_spec.all_training_ids,
        action_grid=resolved.data.actions,
        horizon=resolved.data.action_profile_horizon,
        margin_scale=resolved.objective.margin_scale,
        inherited_config_sha256=str(task["base_config_sha256"]),
    )
    if crossfit != _json_compatible(expected_crossfit):
        raise ProfileArtifactError("crossfit protocol manifest is not canonical")

    checkpoint_path, checkpoint_manifest_file_sha = verified_artifacts[
        "checkpoint_manifest"
    ]
    checkpoint = _load_json(checkpoint_path, label="checkpoint manifest")
    if (
        checkpoint.get("status") != "success"
        or checkpoint.get("plan_sha256") != plan_sha256
        or checkpoint.get("task_sha256") != task["task_sha256"]
        or checkpoint.get("resolved_config_sha256") != expected_resolved_sha
        or checkpoint.get("crossfit_protocol_manifest_sha256") != expected_crossfit_sha
        or checkpoint.get("selection", {}).get("held_out_fold_used") is not False
        or checkpoint.get("selection", {}).get("calibration_or_test_used") is not False
    ):
        raise ProfileArtifactError("checkpoint manifest provenance/selection mismatch")
    checkpoints = checkpoint.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"best", "last"}:
        raise ProfileArtifactError("checkpoint inventory must contain exactly best and last")
    verified_checkpoints = {
        name: _verify_artifact_entry(output_dir, checkpoints[name], label=f"checkpoint.{name}")
        for name in ("best", "last")
    }

    label_manifest_path, label_manifest_file_sha = verified_artifacts[
        "label_shard_manifest"
    ]
    label_manifest = _load_json(label_manifest_path, label="label shard manifest")
    if (
        label_manifest.get("status") != "unvalidated_crossfit_label_shard"
        or label_manifest.get("plan_sha256") != plan_sha256
        or label_manifest.get("task_sha256") != task["task_sha256"]
        or label_manifest.get("action_grid") != list(resolved.data.actions)
        or label_manifest.get("horizon") != resolved.data.action_profile_horizon
        or label_manifest.get("held_out_trajectory_ids_sha256")
        != split_spec.held_out_ids_sha256
        or label_manifest.get("checkpoint_manifest")
        != {"path": checkpoint_path.name, "sha256": checkpoint_manifest_file_sha}
    ):
        raise ProfileArtifactError("label shard manifest provenance mismatch")
    label_entry = label_manifest.get("label_shard")
    if not isinstance(label_entry, dict) or set(label_entry) != {
        "path",
        "sha256",
        "record_count",
        "sample_ids_sha256",
    }:
        raise ProfileArtifactError("label shard entry is malformed")
    label_path = _safe_child(output_dir, label_entry.get("path"), label="label_shard")
    if not label_path.is_file():
        raise ProfileArtifactError("held-out label shard is missing")
    label_sha = file_sha256(label_path)
    if not hmac.compare_digest(str(label_entry.get("sha256")), label_sha):
        raise ProfileArtifactError("held-out label shard SHA-256 mismatch")
    labels = _parse_prediction_rows(
        label_path,
        action_count=len(resolved.data.actions),
        expected_sample_ids=expected_samples,
        expected_trajectory_ids=set(split_spec.held_out_ids),
        trajectory_horizon=resolved.data.horizon,
    )
    if (
        label_entry.get("record_count") != len(labels)
        or label_entry.get("sample_ids_sha256")
        != ids_sha256([str(row["sample_id"]) for row in labels])
    ):
        raise ProfileArtifactError("label shard count/sample checksum mismatch")

    return VerifiedTeacherShard(
        fold_index=fold_index,
        task_id=int(task["task_id"]),
        task_sha256=str(task["task_sha256"]),
        output_dir=output_dir,
        split_spec=split_spec,
        labels=labels,
        checkpoint_manifest_sha256=checkpoint_manifest_file_sha,
        best_checkpoint_sha256=verified_checkpoints["best"][1],
        label_shard_sha256=label_sha,
        label_manifest_sha256=label_manifest_file_sha,
        handoff_manifest_sha256=file_sha256(handoff_path),
        evidence_eligible=bool(handoff.get("evidence_eligible")),
    )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def assemble_crossfit_label_shards(
    tasks: Sequence[Mapping[str, object]],
    shard_dirs: Sequence[Path],
    *,
    plan_sha256: str,
    base_config: LearningConfig,
    output_dir: Path,
    producing_command: Sequence[str],
    engineering_smoke: bool,
    execution_code: Mapping[str, object],
    execution_evidence_eligible: bool,
) -> dict[str, object]:
    """Verify and assemble exactly five complementary teacher shards."""

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite assembled profile-label output: {output_dir}"
        )
    if len(tasks) != FOLD_COUNT or len(shard_dirs) != FOLD_COUNT:
        raise ProfileArtifactError(f"exactly {FOLD_COUNT} tasks and shard directories are required")
    ordered_pairs = sorted(
        zip(tasks, shard_dirs, strict=True), key=lambda pair: int(pair[0]["fold_index"])
    )
    folds = [int(task["fold_index"]) for task, _ in ordered_pairs]
    if folds != list(range(FOLD_COUNT)):
        raise ProfileArtifactError("teacher shards must cover folds 0 through 4 exactly once")
    identity_fields = ("domain", "model_family", "data_seed", "base_config_sha256")
    for field in identity_fields:
        values = {task.get(field) for task, _ in ordered_pairs}
        if len(values) != 1:
            raise ProfileArtifactError(f"teacher tasks disagree on {field}")
    if len({str(path.resolve()) for _, path in ordered_pairs}) != FOLD_COUNT:
        raise ProfileArtifactError("teacher shard directories must be unique")

    verified = tuple(
        verify_teacher_shard(
            task,
            plan_sha256=plan_sha256,
            base_config=base_config,
            output_dir=directory,
            engineering_smoke=engineering_smoke,
        )
        for task, directory in ordered_pairs
    )
    all_training_ids = verified[0].split_spec.all_training_ids
    if any(shard.split_spec.all_training_ids != all_training_ids for shard in verified):
        raise ProfileArtifactError("teacher shards do not share one training trajectory set")
    assignments = assign_trajectory_folds(all_training_ids)
    combined_rows: list[dict[str, object]] = []
    seen_samples: set[str] = set()
    for shard in verified:
        expected_fold_ids = {
            trajectory_id
            for trajectory_id, fold in assignments.items()
            if fold == shard.fold_index
        }
        if set(shard.split_spec.held_out_ids) != expected_fold_ids:
            raise ProfileArtifactError("teacher held-out partition disagrees with fold assignment")
        for row in shard.labels:
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise ProfileArtifactError(f"duplicate sample across teacher shards: {sample_id}")
            seen_samples.add(sample_id)
            combined_rows.append(
                {
                    **row,
                    "teacher_fold": shard.fold_index,
                    "teacher_task_sha256": shard.task_sha256,
                }
            )
    trajectory_horizon = resolve_teacher_config(
        base_config,
        ordered_pairs[0][0],
        device="cpu" if engineering_smoke else base_config.run.device,
        engineering_smoke=engineering_smoke,
    ).data.horizon
    expected_samples = {
        f"{trajectory_id}:{timestep:04d}"
        for trajectory_id in all_training_ids
        for timestep in range(trajectory_horizon)
    }
    if seen_samples != expected_samples:
        missing = sorted(expected_samples - seen_samples)
        unexpected = sorted(seen_samples - expected_samples)
        raise ProfileArtifactError(
            "five shards do not exactly cover the training samples; "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    combined_rows.sort(key=lambda row: str(row["sample_id"]))

    resolved = resolve_teacher_config(
        base_config,
        ordered_pairs[0][0],
        device="cpu" if engineering_smoke else base_config.run.device,
        engineering_smoke=engineering_smoke,
    )
    evidence_eligible = (
        not engineering_smoke
        and execution_evidence_eligible
        and all(shard.evidence_eligible for shard in verified)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    labels_path = output_dir / ASSEMBLED_LABEL_FILENAME
    _write_jsonl_atomic(labels_path, combined_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "status": (
            "engineering_smoke_complete_not_evidence"
            if engineering_smoke
            else "complete_crossfit_label_manifest"
        ),
        "semantic_arm": PREDICTED_PROFILE_ARM,
        "plan_sha256": plan_sha256,
        "domain": ordered_pairs[0][0]["domain"],
        "model_family": ordered_pairs[0][0]["model_family"],
        "data_seed": ordered_pairs[0][0]["data_seed"],
        "base_config_sha256": ordered_pairs[0][0]["base_config_sha256"],
        "engineering_smoke": engineering_smoke,
        "evidence_eligible": evidence_eligible,
        "execution_code": dict(execution_code),
        "action_grid": list(resolved.data.actions),
        "profile_rollout_horizon": resolved.data.action_profile_horizon,
        "trajectory_sample_horizon": trajectory_horizon,
        "margin_scale": resolved.objective.margin_scale,
        "training_trajectories": {
            "count": len(all_training_ids),
            "sha256": ids_sha256(all_training_ids),
        },
        "teachers": [
            {
                "fold_index": shard.fold_index,
                "task_id": shard.task_id,
                "task_sha256": shard.task_sha256,
                "planned_output_dir": ordered_pairs[index][0]["output_dir"],
                "handoff_manifest_sha256": shard.handoff_manifest_sha256,
                "checkpoint_manifest_sha256": shard.checkpoint_manifest_sha256,
                "best_checkpoint_sha256": shard.best_checkpoint_sha256,
                "label_manifest_sha256": shard.label_manifest_sha256,
                "label_shard_sha256": shard.label_shard_sha256,
                "held_out_trajectory_ids_sha256": shard.split_spec.held_out_ids_sha256,
                "held_out_sample_count": len(shard.labels),
            }
            for index, shard in enumerate(verified)
        ],
        "combined_labels": {
            "path": labels_path.name,
            "sha256": file_sha256(labels_path),
            "record_count": len(combined_rows),
            "sample_ids_sha256": ids_sha256(sorted(seen_samples)),
        },
        "isolation": {
            "input_split": "training_only_crossfit",
            "ordinary_validation_included": False,
            "calibration_included": False,
            "final_test_included": False,
            "oracle_action_profiles_included": False,
        },
        "downstream": {
            "ingestion_ready": True,
            "target_key": "predicted_action_profile_target",
            "semantic_arm": PREDICTED_PROFILE_ARM,
            "coverage_gate_complete": False,
        },
        "producing_command": list(producing_command),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / ASSEMBLED_MANIFEST_FILENAME
    write_json_atomic(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "label_count": len(combined_rows),
        "trajectory_count": len(all_training_ids),
        "evidence_eligible": evidence_eligible,
    }


def _parse_combined_rows(
    path: Path,
    *,
    action_count: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ProfileArtifactError(f"cannot read assembled labels: {error}") from error
    required = {
        "sample_id",
        "trajectory_id",
        "timestep",
        "predicted_action_profile",
        "teacher_fold",
        "teacher_task_sha256",
    }
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProfileArtifactError(
                f"invalid assembled-label JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict) or set(row) != required:
            raise ProfileArtifactError("assembled label row schema mismatch")
        profile = row["predicted_action_profile"]
        fold = row["teacher_fold"]
        if (
            not isinstance(row["sample_id"], str)
            or not isinstance(row["trajectory_id"], str)
            or isinstance(row["timestep"], bool)
            or not isinstance(row["timestep"], int)
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or not 0 <= fold < FOLD_COUNT
            or not isinstance(row["teacher_task_sha256"], str)
            or len(row["teacher_task_sha256"]) != 64
            or not isinstance(profile, list)
            or len(profile) != action_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in profile
            )
        ):
            raise ProfileArtifactError("assembled label row contains invalid values")
        if row["sample_id"] != f"{row['trajectory_id']}:{row['timestep']:04d}":
            raise ProfileArtifactError("assembled sample identity is inconsistent")
        rows.append(row)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ProfileArtifactError("assembled labels contain duplicate sample IDs")
    return tuple(rows)


def load_crossfit_label_index(
    manifest_path: Path,
    *,
    expected_domain: str,
    expected_model_family: str,
    expected_data_seed: int,
    expected_plan_sha256: str,
    expected_base_config_sha256: str,
    expected_action_grid: Sequence[float],
    expected_profile_horizon: int,
    expected_margin_scale: float,
    expected_sample_ids: Sequence[str],
    allow_engineering_smoke: bool = False,
) -> CrossfitLabelIndex:
    """Load labels only after re-verifying every identity, count, and checksum."""

    manifest = _load_json(manifest_path, label="assembled label manifest")
    declared_manifest_sha = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual_manifest_sha = canonical_sha256(unsigned)
    if (
        not isinstance(declared_manifest_sha, str)
        or not hmac.compare_digest(declared_manifest_sha, actual_manifest_sha)
    ):
        raise ProfileArtifactError("assembled manifest SHA-256 mismatch")
    engineering_smoke = manifest.get("engineering_smoke") is True
    expected_status = (
        "engineering_smoke_complete_not_evidence"
        if engineering_smoke
        else "complete_crossfit_label_manifest"
    )
    if engineering_smoke and not allow_engineering_smoke:
        raise ProfileArtifactError("engineering-smoke labels are forbidden for evidence training")
    expected_identity = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "status": expected_status,
        "semantic_arm": PREDICTED_PROFILE_ARM,
        "plan_sha256": expected_plan_sha256,
        "domain": expected_domain,
        "model_family": expected_model_family,
        "data_seed": expected_data_seed,
        "base_config_sha256": expected_base_config_sha256,
        "action_grid": [float(value) for value in expected_action_grid],
        "profile_rollout_horizon": expected_profile_horizon,
        "margin_scale": float(expected_margin_scale),
    }
    mismatches = {
        field: {"expected": value, "found": manifest.get(field)}
        for field, value in expected_identity.items()
        if manifest.get(field) != value
    }
    if mismatches:
        raise ProfileArtifactError(
            "assembled manifest identity mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    teachers = manifest.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != FOLD_COUNT:
        raise ProfileArtifactError("assembled manifest must authenticate exactly five teachers")
    teacher_by_fold: dict[int, dict[str, object]] = {}
    for teacher in teachers:
        if not isinstance(teacher, dict):
            raise ProfileArtifactError("assembled teacher entry must be an object")
        fold = teacher.get("fold_index")
        task_sha = teacher.get("task_sha256")
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or not 0 <= fold < FOLD_COUNT
            or fold in teacher_by_fold
            or not isinstance(task_sha, str)
            or len(task_sha) != 64
        ):
            raise ProfileArtifactError("assembled teacher fold/task identity is invalid")
        teacher_by_fold[fold] = teacher
    if set(teacher_by_fold) != set(range(FOLD_COUNT)):
        raise ProfileArtifactError("assembled teacher folds are incomplete")

    combined = manifest.get("combined_labels")
    if not isinstance(combined, dict) or set(combined) != {
        "path",
        "sha256",
        "record_count",
        "sample_ids_sha256",
    }:
        raise ProfileArtifactError("combined_labels entry is malformed")
    labels_path = _safe_child(manifest_path.parent, combined.get("path"), label="combined_labels")
    if not labels_path.is_file():
        raise ProfileArtifactError("assembled label file is missing")
    actual_labels_sha = file_sha256(labels_path)
    if not hmac.compare_digest(str(combined.get("sha256")), actual_labels_sha):
        raise ProfileArtifactError("assembled label file SHA-256 mismatch")
    rows = _parse_combined_rows(labels_path, action_count=len(expected_action_grid))
    expected_samples = list(expected_sample_ids)
    if (
        not expected_samples
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in expected_samples)
        or len(expected_samples) != len(set(expected_samples))
    ):
        raise ProfileArtifactError("expected_sample_ids must be unique non-empty strings")
    row_ids = [str(row["sample_id"]) for row in rows]
    if set(row_ids) != set(expected_samples):
        missing = sorted(set(expected_samples) - set(row_ids))
        unexpected = sorted(set(row_ids) - set(expected_samples))
        raise ProfileArtifactError(
            "assembled labels do not exactly match downstream training samples; "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    if (
        combined.get("record_count") != len(rows)
        or combined.get("sample_ids_sha256") != ids_sha256(row_ids)
    ):
        raise ProfileArtifactError("assembled label count/sample checksum mismatch")
    profiles: dict[str, tuple[float, ...]] = {}
    teacher_folds: dict[str, int] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        fold = int(row["teacher_fold"])
        if row["teacher_task_sha256"] != teacher_by_fold[fold]["task_sha256"]:
            raise ProfileArtifactError("label row teacher hash disagrees with teacher inventory")
        profiles[sample_id] = tuple(float(value) for value in row["predicted_action_profile"])
        teacher_folds[sample_id] = fold
    return CrossfitLabelIndex(
        semantic_arm=PREDICTED_PROFILE_ARM,
        domain=expected_domain,
        model_family=expected_model_family,
        data_seed=expected_data_seed,
        action_grid=tuple(float(value) for value in expected_action_grid),
        profile_horizon=expected_profile_horizon,
        margin_scale=float(expected_margin_scale),
        manifest_sha256=actual_manifest_sha,
        evidence_eligible=bool(manifest.get("evidence_eligible")),
        profiles=MappingProxyType(profiles),
        teacher_folds=MappingProxyType(teacher_folds),
        teacher_task_sha256_by_fold=MappingProxyType(
            {
                fold: str(teacher_by_fold[fold]["task_sha256"])
                for fold in range(FOLD_COUNT)
            }
        ),
    )


class PredictedProfileDataset:
    """Attach authenticated cross-fitted labels to an observed-only training dataset."""

    semantic_arm = PREDICTED_PROFILE_ARM

    def __init__(self, observed_dataset: Any, labels: CrossfitLabelIndex) -> None:
        sample_ids = getattr(observed_dataset, "sample_ids", None)
        if not isinstance(sample_ids, tuple):
            raise ProfileArtifactError("observed dataset must expose an immutable sample_ids tuple")
        if len(sample_ids) != len(set(sample_ids)):
            raise ProfileArtifactError("observed dataset contains duplicate sample IDs")
        if set(sample_ids) != set(labels.profiles):
            raise ProfileArtifactError(
                "observed dataset and authenticated labels must have identical sample IDs"
            )
        self.observed_dataset = observed_dataset
        self.labels = labels
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.observed_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.observed_dataset[index])
        forbidden = {"state_features", "action_safety_margins"} & set(sample)
        if forbidden:
            raise ProfileArtifactError(
                f"observed dataset exposed forbidden supervision fields: {sorted(forbidden)}"
            )
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in self.labels.profiles:
            raise ProfileArtifactError("dataset sample is missing an authenticated profile label")
        history = sample.get("history")
        if history is None or not hasattr(history, "new_tensor"):
            raise ProfileArtifactError("dataset history must be a tensor-like object")
        sample["predicted_action_profile_target"] = history.new_tensor(
            self.labels.profiles[sample_id]
        )
        sample["profile_label_source"] = PREDICTED_PROFILE_ARM
        sample["profile_teacher_fold"] = self.labels.teacher_folds[sample_id]
        return sample


def build_ingested_profile_training_dataset(
    modules: Any,
    config: LearningConfig,
    *,
    data_seed: int,
    labels: CrossfitLabelIndex,
) -> PredictedProfileDataset:
    """Materialize only the training split and attach authenticated predicted profiles."""

    if (
        labels.domain != config.data.task
        or labels.model_family != config.model.family
        or labels.data_seed != data_seed
        or labels.action_grid != config.data.actions
        or labels.profile_horizon != config.data.action_profile_horizon
        or labels.margin_scale != config.objective.margin_scale
    ):
        raise ProfileArtifactError("label index disagrees with downstream training config")
    trajectories = generate_trajectories(
        config.data,
        seed=data_seed,
        include_splits=("train",),
        include_action_profiles=False,
    )
    if any(trajectory.action_safety_margins for trajectory in trajectories):
        raise ProfileArtifactError("oracle profiles were materialized during downstream ingestion")
    rendered = tuple(
        _render_trajectory_tensor(trajectory, config.data, modules.torch)
        for trajectory in trajectories
    )
    observed = ObservedMarginDataset(
        trajectories,
        tuple(range(len(trajectories))),
        config.data,
        rendered,
    )
    return PredictedProfileDataset(observed, labels)
