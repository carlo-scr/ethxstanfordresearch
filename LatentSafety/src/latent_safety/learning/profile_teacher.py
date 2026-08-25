"""Single-fold trainer for the nonprivileged predicted-profile protocol.

This module deliberately trains one task at a time.  It consumes only rendered pixel histories,
behavior actions, next rendered histories, and observed safety margins from the training/ordinary-
validation splits.  Calibration and final-test trajectories, privileged states, known dynamics,
and oracle counterfactual profiles are never materialized for a teacher run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_safety.learning.config import (
    LearningConfig,
    canonical_config_json,
    validate_config,
)
from latent_safety.learning.data import (
    ControlledTrajectory,
    SampleRef,
    _history_indices,
    _render_trajectory_tensor,
    generate_trajectories,
    trajectory_split_assignment,
)
from latent_safety.learning.models import build_world_model, model_metadata
from latent_safety.learning.profile_protocol import (
    PROFILE_PROTOCOL_VERSION,
    assign_trajectory_folds,
    build_crossfit_manifest,
    teacher_seed,
)
from latent_safety.learning.runtime import require_torch, seed_everything, select_device
from latent_safety.learning.trainer import (
    _evaluate_loader,
    _make_loaders,
    _save_checkpoint_atomic,
    _sha256,
    _state_sha256,
    _train_epoch,
    _with_world_model_utility,
)
from latent_safety.manifest import write_json_atomic

_ALLOWED_SPLITS = ("train", "validation")
_PERMITTED_FIELDS = (
    "sample_id",
    "trajectory_id",
    "timestep",
    "history",
    "next_history",
    "action",
    "safety_margin",
)


class ProfileTeacherError(ValueError):
    """Raised when a planned teacher task or its data violates the frozen protocol."""


@dataclass(frozen=True)
class TeacherSplitSpec:
    all_training_ids: tuple[str, ...]
    fitting_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    all_training_ids_sha256: str
    fitting_ids_sha256: str
    held_out_ids_sha256: str
    validation_ids_sha256: str


@dataclass(frozen=True)
class TeacherDatasetBundle:
    datasets: dict[str, Any]
    split_trajectory_ids: dict[str, tuple[str, ...]]


class ObservedMarginDataset:
    """A strict field allowlist over rendered controlled trajectories."""

    def __init__(
        self,
        trajectories: tuple[ControlledTrajectory, ...],
        trajectory_indices: tuple[int, ...],
        config: Any,
        rendered_frames: tuple[Any, ...],
    ) -> None:
        self.trajectories = trajectories
        self.trajectory_indices = trajectory_indices
        self.config = config
        self.rendered_frames = rendered_frames
        self.refs = tuple(
            SampleRef(trajectory_index=index, timestep=timestep)
            for index in trajectory_indices
            for timestep in range(config.horizon)
        )
        self.sample_ids = tuple(
            f"{trajectories[ref.trajectory_index].trajectory_id}:{ref.timestep:04d}"
            for ref in self.refs
        )

    def __len__(self) -> int:
        return len(self.refs)

    def _history(self, trajectory_index: int, timestep: int) -> Any:
        indices = list(_history_indices(timestep, self.config.history_length))
        return self.rendered_frames[trajectory_index][indices]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        trajectory = self.trajectories[ref.trajectory_index]
        timestep = ref.timestep
        frames = self.rendered_frames[ref.trajectory_index]
        sample = {
            "sample_id": f"{trajectory.trajectory_id}:{timestep:04d}",
            "trajectory_id": trajectory.trajectory_id,
            "timestep": timestep,
            "history": self._history(ref.trajectory_index, timestep),
            "next_history": self._history(ref.trajectory_index, timestep + 1),
            "action": frames.new_tensor([trajectory.actions[timestep]]),
            "safety_margin": frames.new_tensor(trajectory.safety_margins[timestep]),
        }
        if tuple(sample) != _PERMITTED_FIELDS:
            raise AssertionError("observed-margin dataset field allowlist drifted")
        return sample


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return _canonical_sha256(sorted(ids))


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build_teacher_split_spec(
    config: LearningConfig,
    *,
    data_seed: int,
    fold_index: int,
) -> TeacherSplitSpec:
    """Resolve the exact cross-fit and ordinary-validation trajectory sets."""

    split_assignment = trajectory_split_assignment(config.data, seed=data_seed)
    all_training_ids = tuple(
        sorted(
            trajectory_id
            for trajectory_id, split in split_assignment.items()
            if split == "train"
        )
    )
    validation_ids = tuple(
        sorted(
            trajectory_id
            for trajectory_id, split in split_assignment.items()
            if split == "validation"
        )
    )
    fold_assignment = assign_trajectory_folds(all_training_ids)
    held_out_ids = tuple(
        sorted(
            trajectory_id
            for trajectory_id, assigned_fold in fold_assignment.items()
            if assigned_fold == fold_index
        )
    )
    fitting_ids = tuple(sorted(set(all_training_ids) - set(held_out_ids)))
    if not fitting_ids or not held_out_ids or not validation_ids:
        raise ProfileTeacherError(
            "teacher fitting, held-out, and validation sets must be non-empty"
        )
    if (
        set(fitting_ids) & set(held_out_ids)
        or set(fitting_ids) & set(validation_ids)
        or set(held_out_ids) & set(validation_ids)
    ):
        raise ProfileTeacherError("teacher trajectory sets must be pairwise disjoint")
    if set(fitting_ids) | set(held_out_ids) != set(all_training_ids):
        raise ProfileTeacherError("fitting and held-out sets must partition the training split")
    return TeacherSplitSpec(
        all_training_ids=all_training_ids,
        fitting_ids=fitting_ids,
        held_out_ids=held_out_ids,
        validation_ids=validation_ids,
        all_training_ids_sha256=_ids_sha256(all_training_ids),
        fitting_ids_sha256=_ids_sha256(fitting_ids),
        held_out_ids_sha256=_ids_sha256(held_out_ids),
        validation_ids_sha256=_ids_sha256(validation_ids),
    )


def validate_task_split_hashes(
    task: Mapping[str, object], split_spec: TeacherSplitSpec
) -> None:
    """Fail closed unless a production task exactly matches the planned fold hashes."""

    expected = {
        "training_trajectory_count": len(split_spec.fitting_ids),
        "training_trajectory_ids_sha256": split_spec.fitting_ids_sha256,
        "held_out_trajectory_count": len(split_spec.held_out_ids),
        "held_out_trajectory_ids_sha256": split_spec.held_out_ids_sha256,
    }
    mismatches = {
        key: {"planned": task.get(key), "resolved": value}
        for key, value in expected.items()
        if task.get(key) != value
    }
    if mismatches:
        raise ProfileTeacherError(
            "planned teacher trajectory counts/hashes do not match the resolved fold: "
            + json.dumps(mismatches, sort_keys=True)
        )


def resolve_teacher_config(
    base: LearningConfig,
    task: Mapping[str, object],
    *,
    device: str,
    engineering_smoke: bool = False,
) -> LearningConfig:
    """Apply only the registered teacher overrides to a verified domain config."""

    family = str(task["model_family"])
    kl_weight = 0.0 if family == "ae" else base.objective.kl_weight
    experiment = (
        f"e2_profile_teacher_{task['domain']}_{family}_"
        f"data_seed_{task['data_seed']}_fold_{task['fold_index']}"
    )
    resolved = dataclasses.replace(
        base,
        experiment=experiment,
        status="profile_teacher_task",
        run=dataclasses.replace(
            base.run,
            seed=int(task["teacher_seed"]),
            output_dir=str(task["output_dir"]),
            device=device,
            epochs=int(task["epochs"]),
        ),
        data=dataclasses.replace(base.data, history_length=4),
        model=dataclasses.replace(
            base.model,
            family=family,
            history_encoder="stack",
            latent_dim=int(task["latent_dim"]),
            hidden_dim=int(task["hidden_dim"]),
            transition_hidden_dim=int(task["transition_hidden_dim"]),
        ),
        objective=dataclasses.replace(
            base.objective,
            safety_arm="h_prediction",
            safety_weight=float(task["safety_weight"]),
            kl_weight=kl_weight,
        ),
        evaluation=dataclasses.replace(base.evaluation, emit_audit_records=False),
    )
    if engineering_smoke:
        resolved = dataclasses.replace(
            resolved,
            status="engineering_smoke_only",
            run=dataclasses.replace(
                resolved.run,
                output_dir=str(task["output_dir"]) + "/engineering_smoke",
                device="cpu",
                epochs=1,
                batch_size=8,
                num_workers=0,
                amp=False,
                log_every=1,
            ),
            data=dataclasses.replace(
                resolved.data,
                trajectories=20,
                horizon=8,
                image_size=16,
            ),
            evaluation=dataclasses.replace(
                resolved.evaluation,
                rollout_horizons=(1, 2),
                max_rollout_cases=8,
                emit_audit_records=False,
                max_audit_records_per_split=16,
            ),
        )
    validate_config(resolved)
    required = {
        "domain": resolved.data.task,
        "history_mode": f"{resolved.model.history_encoder}_h{resolved.data.history_length}",
        "objective": resolved.objective.safety_arm,
        "teacher_seed": resolved.run.seed,
        "epochs": resolved.run.epochs,
    }
    if not engineering_smoke:
        for field, actual in required.items():
            if task.get(field) != actual:
                raise ProfileTeacherError(
                    f"task.{field}={task.get(field)!r} disagrees with resolved value {actual!r}"
                )
    return resolved


def _build_dataset_bundle(
    modules: Any,
    config: LearningConfig,
    *,
    data_seed: int,
    split_spec: TeacherSplitSpec,
) -> TeacherDatasetBundle:
    trajectories = generate_trajectories(
        config.data,
        seed=data_seed,
        include_splits=_ALLOWED_SPLITS,
        include_action_profiles=False,
    )
    if any(trajectory.action_safety_margins for trajectory in trajectories):
        raise ProfileTeacherError("oracle counterfactual profiles were materialized")
    if {trajectory.split for trajectory in trajectories} - set(_ALLOWED_SPLITS):
        raise ProfileTeacherError("calibration or final-test trajectories were materialized")
    actual_train = {
        trajectory.trajectory_id for trajectory in trajectories if trajectory.split == "train"
    }
    actual_validation = {
        trajectory.trajectory_id
        for trajectory in trajectories
        if trajectory.split == "validation"
    }
    if actual_train != set(split_spec.all_training_ids):
        raise ProfileTeacherError("generated training trajectories disagree with split preflight")
    if actual_validation != set(split_spec.validation_ids):
        raise ProfileTeacherError("generated validation trajectories disagree with split preflight")

    rendered = tuple(
        _render_trajectory_tensor(trajectory, config.data, modules.torch)
        for trajectory in trajectories
    )
    index_by_id = {
        trajectory.trajectory_id: index for index, trajectory in enumerate(trajectories)
    }
    split_ids = {
        "train": split_spec.fitting_ids,
        "held_out": split_spec.held_out_ids,
        "validation": split_spec.validation_ids,
    }
    datasets = {
        split: ObservedMarginDataset(
            trajectories,
            tuple(index_by_id[trajectory_id] for trajectory_id in ids),
            config.data,
            rendered,
        )
        for split, ids in split_ids.items()
    }
    return TeacherDatasetBundle(datasets=datasets, split_trajectory_ids=split_ids)


def predict_action_profiles_tensor(
    modules: Any,
    model: Any,
    history: Any,
    device: Any,
    *,
    actions: Sequence[float],
    horizon: int,
) -> Any:
    """Predict constant-action minimum margins from posterior-mean latent rollouts."""

    torch = modules.torch
    history = history.to(device)
    initial_latent = model.encode(history, sample=False)["z"]
    current_margin = model.margin_head(initial_latent).squeeze(-1)
    predictions = []
    for action in actions:
        latent = initial_latent
        margins = [current_margin]
        action_batch = torch.full(
            (int(initial_latent.shape[0]), 1),
            float(action),
            device=device,
            dtype=initial_latent.dtype,
        )
        for _ in range(horizon):
            latent = model.predict_next(latent, action_batch)
            margins.append(model.margin_head(latent).squeeze(-1))
        predictions.append(torch.stack(margins, dim=0).amin(dim=0))
    profiles = torch.stack(predictions, dim=1)
    if not bool(torch.isfinite(profiles).all().item()):
        raise FloatingPointError("profile prediction contains a non-finite value")
    return profiles


def _predict_held_out_profiles(
    modules: Any,
    model: Any,
    loader: Any,
    device: Any,
    *,
    actions: Sequence[float],
    horizon: int,
) -> tuple[dict[str, object], ...]:
    torch = modules.torch
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            profiles = predict_action_profiles_tensor(
                modules,
                model,
                raw_batch["history"],
                device,
                actions=actions,
                horizon=horizon,
            )
            profile_rows = profiles.detach().cpu().tolist()
            timesteps = raw_batch["timestep"].tolist()
            for sample_id, trajectory_id, timestep, profile in zip(
                raw_batch["sample_id"],
                raw_batch["trajectory_id"],
                timesteps,
                profile_rows,
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "trajectory_id": str(trajectory_id),
                        "timestep": int(timestep),
                        "predicted_action_profile": [float(value) for value in profile],
                    }
                )
    rows.sort(key=lambda row: (str(row["trajectory_id"]), int(row["timestep"])))
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ProfileTeacherError("held-out label shard contains duplicate sample IDs")
    return tuple(rows)


def _environment_payload(
    modules: Any,
    *,
    device_info: Any,
    determinism: Mapping[str, object],
) -> dict[str, object]:
    torch = modules.torch
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "device": device_info.selected,
        "accelerator_name": device_info.accelerator_name,
        "determinism": dict(determinism),
    }


def run_profile_teacher_task(
    base_config: LearningConfig,
    task: Mapping[str, object],
    *,
    base_config_path: str,
    plan_sha256: str,
    planned_code: Mapping[str, object],
    output_dir: Path,
    device: str,
    producing_command: Sequence[str],
    evidence_eligible: bool,
    engineering_smoke: bool = False,
) -> dict[str, object]:
    """Train one fold teacher and atomically hand off its checkpoint and label shard."""

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite teacher output path, even if empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run_manifest_path = output_dir / "run_manifest.json"
    task_sha256 = str(task["task_sha256"])
    resolved = resolve_teacher_config(
        base_config,
        task,
        device=device,
        engineering_smoke=engineering_smoke,
    )
    data_seed = int(task["data_seed"])
    fold_index = int(task["fold_index"])
    split_spec = build_teacher_split_spec(
        resolved,
        data_seed=data_seed,
        fold_index=fold_index,
    )
    if not engineering_smoke:
        validate_task_split_hashes(task, split_spec)
    resolved_config_json = canonical_config_json(resolved)
    resolved_config_sha256 = hashlib.sha256(
        resolved_config_json.encode("utf-8")
    ).hexdigest()
    run_manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": PROFILE_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": resolved.experiment,
        "status": "initializing",
        "evidence_eligible": evidence_eligible and not engineering_smoke,
        "engineering_smoke": engineering_smoke,
        "plan": {
            "sha256": plan_sha256,
            "planned_code": dict(planned_code),
            "task_id": task["task_id"],
            "task_sha256": task_sha256,
        },
        "base_config": {
            "path": base_config_path,
            "sha256": task["base_config_sha256"],
        },
        "resolved_config": {
            "sha256": resolved_config_sha256,
            "values": resolved.to_dict(),
        },
        "seeds": {
            "data": data_seed,
            "teacher": resolved.run.seed,
            "registered_teacher_rule": teacher_seed(data_seed, fold_index),
        },
        "data_access": {
            "materialized_splits": list(_ALLOWED_SPLITS),
            "excluded_splits": ["calibration", "test"],
            "permitted_sample_fields": list(_PERMITTED_FIELDS),
            "oracle_action_profiles_materialized": False,
            "hidden_state_exposed_to_model": False,
        },
        "producing_command": list(producing_command),
        "failure": None,
    }
    write_json_atomic(run_manifest_path, run_manifest)
    try:
        modules = require_torch()
        torch = modules.torch
        selected_device, device_info = select_device(torch, resolved.run.device)
        determinism = seed_everything(
            torch,
            seed=resolved.run.seed,
            deterministic=resolved.run.deterministic,
            warn_only=resolved.run.deterministic_warn_only,
        )
        bundle = _build_dataset_bundle(
            modules,
            resolved,
            data_seed=data_seed,
            split_spec=split_spec,
        )
        loader_result = _make_loaders(
            modules,
            bundle,
            resolved,
            pin_memory=selected_device.type == "cuda",
        )
        # The shared trainer also returns its generator for exact-resume support.  Profile teachers
        # do not resume, but consume the same loader construction without discarding determinism.
        loaders = loader_result[0] if isinstance(loader_result, tuple) else loader_result
        crossfit_payload, crossfit_sha256 = build_crossfit_manifest(
            task=resolved.data.task,
            model_family=resolved.model.family,
            data_seed=data_seed,
            training_trajectory_ids=split_spec.all_training_ids,
            action_grid=resolved.data.actions,
            horizon=resolved.data.action_profile_horizon,
            margin_scale=resolved.objective.margin_scale,
            inherited_config_sha256=str(task["base_config_sha256"]),
        )
        crossfit_path = output_dir / "crossfit_protocol_manifest.json"
        write_json_atomic(crossfit_path, crossfit_payload)
        dataset_manifest = {
            "schema_version": 1,
            "protocol_version": PROFILE_PROTOCOL_VERSION,
            "generator": "controlled_observed_margin_teacher_v1",
            "data_seed": data_seed,
            "task": resolved.data.task,
            "fold_index": fold_index,
            "split_access": {
                "materialized": list(_ALLOWED_SPLITS),
                "not_materialized": ["calibration", "test"],
            },
            "signals": {
                "permitted_sample_fields": list(_PERMITTED_FIELDS),
                "oracle_action_profiles_materialized": False,
                "hidden_state_exposed_to_model": False,
            },
            "trajectory_sets": {
                "all_training": {
                    "count": len(split_spec.all_training_ids),
                    "sha256": split_spec.all_training_ids_sha256,
                },
                "fitting": {
                    "count": len(split_spec.fitting_ids),
                    "sha256": split_spec.fitting_ids_sha256,
                },
                "held_out": {
                    "count": len(split_spec.held_out_ids),
                    "sha256": split_spec.held_out_ids_sha256,
                },
                "ordinary_validation": {
                    "count": len(split_spec.validation_ids),
                    "sha256": split_spec.validation_ids_sha256,
                },
            },
            "sample_counts": {
                split: len(dataset) for split, dataset in bundle.datasets.items()
            },
            "crossfit_protocol_manifest_sha256": crossfit_sha256,
            "engineering_smoke": engineering_smoke,
            "production_plan_split_hashes_enforced": not engineering_smoke,
        }
        dataset_manifest_sha256 = _canonical_sha256(dataset_manifest)
        dataset_path = output_dir / "dataset_manifest.json"
        write_json_atomic(dataset_path, dataset_manifest)

        model = build_world_model(modules, resolved.model, resolved.data).to(selected_device)
        model_info = model_metadata(model, resolved.model, resolved.data)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved.run.learning_rate,
            weight_decay=resolved.run.weight_decay,
        )
        amp_enabled = resolved.run.amp and selected_device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        run_manifest.update(
            {
                "status": "running",
                "environment": _environment_payload(
                    modules,
                    device_info=device_info,
                    determinism=determinism,
                ),
                "architecture": model_info,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "crossfit_protocol_manifest_sha256": crossfit_sha256,
                "selection": {
                    "split": "ordinary_validation",
                    "metric": "world_model_utility",
                    "mode": "min",
                    "tie_break": "earliest_epoch",
                    "held_out_fold_used": False,
                    "calibration_or_test_used": False,
                },
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)

        best_path = output_dir / "checkpoint_best.pt"
        last_path = output_dir / "checkpoint_last.pt"
        best_metric = math.inf
        best_epoch = -1
        best_validation: dict[str, float | int] | None = None
        history: list[dict[str, object]] = []
        for epoch in range(1, resolved.run.epochs + 1):
            train_metrics = _with_world_model_utility(
                _train_epoch(
                    modules,
                    model,
                    loaders["train"],
                    optimizer,
                    scaler,
                    selected_device,
                    resolved,
                ),
                resolved,
            )
            validation_metrics, _ = _evaluate_loader(
                modules,
                model,
                loaders["validation"],
                selected_device,
                resolved,
                split="validation",
                collect_records=False,
            )
            validation_metrics = _with_world_model_utility(validation_metrics, resolved)
            epoch_payload: dict[str, object] = {
                "epoch": epoch,
                "train": train_metrics,
                "ordinary_validation": validation_metrics,
            }
            history.append(epoch_payload)
            if epoch == 1 or epoch % resolved.run.log_every == 0 or epoch == resolved.run.epochs:
                print(json.dumps(epoch_payload, sort_keys=True), flush=True)
            selection_value = float(validation_metrics["world_model_utility"])
            checkpoint_payload = {
                "schema_version": 1,
                "protocol_version": PROFILE_PROTOCOL_VERSION,
                "experiment": resolved.experiment,
                "epoch": epoch,
                "config": resolved.to_dict(),
                "plan_sha256": plan_sha256,
                "task_sha256": task_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "crossfit_protocol_manifest_sha256": crossfit_sha256,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ordinary_validation": validation_metrics,
            }
            _save_checkpoint_atomic(torch, last_path, checkpoint_payload)
            if selection_value < best_metric:
                best_metric = selection_value
                best_epoch = epoch
                best_validation = validation_metrics
                _save_checkpoint_atomic(torch, best_path, checkpoint_payload)
        if best_epoch < 1 or best_validation is None:
            raise RuntimeError("teacher training completed without a selectable checkpoint")
        history_path = output_dir / "history.jsonl"
        _write_jsonl_atomic(history_path, history)
        selected_checkpoint = torch.load(
            best_path,
            map_location=selected_device,
            weights_only=False,
        )
        model.load_state_dict(selected_checkpoint["model_state"])

        label_rows = _predict_held_out_profiles(
            modules,
            model,
            loaders["held_out"],
            selected_device,
            actions=resolved.data.actions,
            horizon=resolved.data.action_profile_horizon,
        )
        expected_labels = len(split_spec.held_out_ids) * resolved.data.horizon
        if len(label_rows) != expected_labels:
            raise ProfileTeacherError(
                f"held-out label shard has {len(label_rows)} rows, expected {expected_labels}"
            )
        if {str(row["trajectory_id"]) for row in label_rows} != set(
            split_spec.held_out_ids
        ):
            raise ProfileTeacherError("label shard trajectory IDs disagree with held-out fold")
        label_path = output_dir / "heldout_profile_predictions.jsonl"
        _write_jsonl_atomic(label_path, label_rows)

        checkpoint_manifest = {
            "schema_version": 1,
            "protocol_version": PROFILE_PROTOCOL_VERSION,
            "status": "success",
            "plan_sha256": plan_sha256,
            "task_sha256": task_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "crossfit_protocol_manifest_sha256": crossfit_sha256,
            "architecture": model_info,
            "selection": {
                "metric": "ordinary_validation.world_model_utility",
                "mode": "min",
                "tie_break": "earliest_epoch",
                "selected_epoch": best_epoch,
                "selected_value": best_metric,
                "held_out_fold_used": False,
                "calibration_or_test_used": False,
            },
            "checkpoints": {
                "best": {"path": best_path.name, "sha256": _sha256(best_path)},
                "last": {"path": last_path.name, "sha256": _sha256(last_path)},
            },
            "state_hashes": {
                "selected_model_state": _state_sha256(
                    torch, selected_checkpoint["model_state"]
                ),
                "selected_optimizer_state": _state_sha256(
                    torch, selected_checkpoint["optimizer_state"]
                ),
            },
        }
        checkpoint_manifest_path = output_dir / "checkpoint_manifest.json"
        write_json_atomic(checkpoint_manifest_path, checkpoint_manifest)
        checkpoint_manifest_sha256 = _sha256(checkpoint_manifest_path)

        label_manifest = {
            "schema_version": 1,
            "protocol_version": PROFILE_PROTOCOL_VERSION,
            "status": "unvalidated_crossfit_label_shard",
            "plan_sha256": plan_sha256,
            "task_sha256": task_sha256,
            "checkpoint_manifest": {
                "path": checkpoint_manifest_path.name,
                "sha256": checkpoint_manifest_sha256,
            },
            "label_shard": {
                "path": label_path.name,
                "sha256": _sha256(label_path),
                "record_count": len(label_rows),
                "sample_ids_sha256": _ids_sha256(
                    [str(row["sample_id"]) for row in label_rows]
                ),
            },
            "held_out_trajectory_ids_sha256": split_spec.held_out_ids_sha256,
            "action_grid": list(resolved.data.actions),
            "horizon": resolved.data.action_profile_horizon,
            "inference": {
                "history_mode": "stack_h4",
                "latent": "posterior_mean",
                "transition": "deterministic",
                "profile": "minimum_predicted_margin_over_t_0_through_H",
                "ensemble": False,
            },
            "eligibility": {
                "five_fold_assembly_complete": False,
                "coverage_gate_complete": False,
                "paper_evidence": False,
            },
        }
        label_manifest_path = output_dir / "label_shard_manifest.json"
        write_json_atomic(label_manifest_path, label_manifest)

        handoff_manifest = {
            "schema_version": 1,
            "protocol_version": PROFILE_PROTOCOL_VERSION,
            "status": "single_teacher_success",
            "evidence_eligible": evidence_eligible and not engineering_smoke,
            "engineering_smoke": engineering_smoke,
            "plan_sha256": plan_sha256,
            "task_id": task["task_id"],
            "task_sha256": task_sha256,
            "artifacts": {
                "checkpoint_manifest": {
                    "path": checkpoint_manifest_path.name,
                    "sha256": checkpoint_manifest_sha256,
                },
                "label_shard_manifest": {
                    "path": label_manifest_path.name,
                    "sha256": _sha256(label_manifest_path),
                },
                "history": {
                    "path": history_path.name,
                    "sha256": _sha256(history_path),
                },
                "dataset_manifest": {
                    "path": dataset_path.name,
                    "sha256": _sha256(dataset_path),
                },
                "crossfit_protocol_manifest": {
                    "path": crossfit_path.name,
                    "sha256": _sha256(crossfit_path),
                },
            },
            "remaining_blockers": [
                "assemble_and_checksum_all_five_held_out_label_shards",
                "run_the_registered_coverage_validation_gate",
            ],
        }
        handoff_path = output_dir / "teacher_handoff_manifest.json"
        write_json_atomic(handoff_path, handoff_manifest)
        run_manifest.update(
            {
                "status": "success",
                "wall_time_seconds": time.monotonic() - started,
                "selected_epoch": best_epoch,
                "best_validation_world_model_utility": best_metric,
                "handoff_manifest": {
                    "path": handoff_path.name,
                    "sha256": _sha256(handoff_path),
                },
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "selected_epoch": best_epoch,
            "held_out_label_count": len(label_rows),
            "handoff_manifest_sha256": _sha256(handoff_path),
            "evidence_eligible": evidence_eligible and not engineering_smoke,
        }
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "wall_time_seconds": time.monotonic() - started,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        raise
