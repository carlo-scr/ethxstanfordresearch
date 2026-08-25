"""Specialized downstream runner for the nonprivileged predicted-profile arm."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_safety.learning.config import (
    LearningConfig,
    canonical_config_json,
    validate_config,
)
from latent_safety.learning.data import (
    ControlledVideoDataset,
    _render_trajectory_tensor,
    generate_trajectories,
)
from latent_safety.learning.models import build_world_model, model_metadata
from latent_safety.learning.profile_artifacts import (
    PREDICTED_PROFILE_ARM,
    build_ingested_profile_training_dataset,
    canonical_sha256,
    file_sha256,
    ids_sha256,
    load_crossfit_label_index,
)
from latent_safety.learning.profile_coverage import load_profile_coverage_gate
from latent_safety.learning.profile_teacher import (
    ObservedMarginDataset,
    build_teacher_split_spec,
    resolve_teacher_config,
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
from latent_safety.learning.validation_audit import run_validation_postfit_audit
from latent_safety.manifest import write_json_atomic


class PredictedProfileRunError(ValueError):
    """Raised when downstream profile-arm execution violates artifact isolation."""


@dataclass(frozen=True)
class _DownstreamBundle:
    datasets: dict[str, Any]
    split_trajectory_ids: dict[str, tuple[str, ...]]


def resolve_predicted_profile_config(
    base: LearningConfig,
    task: Mapping[str, object],
    *,
    device: str,
    safety_weight: float,
    engineering_smoke: bool,
) -> LearningConfig:
    """Resolve the same-backbone arm while preserving a distinct semantic objective."""

    if not math.isfinite(safety_weight) or safety_weight <= 0.0:
        raise PredictedProfileRunError("safety_weight must be finite and positive")
    teacher_shape = resolve_teacher_config(
        base,
        task,
        device=device,
        engineering_smoke=engineering_smoke,
    )
    weight_token = format(safety_weight, ".8g").replace(".", "p")
    resolved = dataclasses.replace(
        teacher_shape,
        experiment=(
            f"e2_{PREDICTED_PROFILE_ARM}_{task['domain']}_{task['model_family']}_"
            f"data_seed_{task['data_seed']}_weight_{weight_token}"
        ),
        status=(
            "engineering_smoke_only" if engineering_smoke else "preconfirmation_or_confirmatory"
        ),
        run=dataclasses.replace(
            teacher_shape.run,
            seed=int(task["data_seed"]),
            output_dir=(
                "runs/e2_frontier/predicted_profile_arm/engineering_smoke"
                if engineering_smoke
                else (
                    "runs/e2_frontier/predicted_profile_arm/"
                    f"{task['domain']}/{task['model_family']}/"
                    f"data_seed_{task['data_seed']}/weight_{weight_token}"
                )
            ),
        ),
        objective=dataclasses.replace(
            teacher_shape.objective,
            safety_arm=PREDICTED_PROFILE_ARM,
            safety_weight=float(safety_weight),
        ),
    )
    validate_config(resolved)
    return resolved


def _build_validation_dataset(
    modules: Any,
    config: LearningConfig,
    *,
    data_seed: int,
) -> tuple[
    ObservedMarginDataset,
    ControlledVideoDataset,
    tuple[Any, ...],
    tuple[str, ...],
]:
    trajectories = generate_trajectories(
        config.data,
        seed=data_seed,
        include_splits=("validation",),
        include_action_profiles=False,
    )
    if any(trajectory.action_safety_margins for trajectory in trajectories):
        raise PredictedProfileRunError("ordinary validation materialized oracle profiles")
    if any(trajectory.split != "validation" for trajectory in trajectories):
        raise PredictedProfileRunError("ordinary validation dataset contains another split")
    rendered = tuple(
        _render_trajectory_tensor(trajectory, config.data, modules.torch)
        for trajectory in trajectories
    )
    dataset = ObservedMarginDataset(
        trajectories,
        tuple(range(len(trajectories))),
        config.data,
        rendered,
    )
    audit_dataset = ControlledVideoDataset(
        trajectories,
        tuple(range(len(trajectories))),
        config.data,
        rendered,
    )
    return (
        dataset,
        audit_dataset,
        trajectories,
        tuple(trajectory.trajectory_id for trajectory in trajectories),
    )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run_predicted_profile_arm(
    base_config: LearningConfig,
    tasks: Sequence[Mapping[str, object]],
    *,
    teacher_plan_sha256: str,
    orchestration_plan_sha256: str | None,
    assembled_manifest_path: Path,
    coverage_manifest_path: Path,
    output_dir: Path,
    device: str,
    safety_weight: float,
    producing_command: Sequence[str],
    execution_code: Mapping[str, object],
    execution_evidence_eligible: bool,
    engineering_smoke: bool,
) -> dict[str, object]:
    """Train the proposed arm using only authenticated cross-fitted training labels."""

    for label, value, required in (
        ("teacher_plan_sha256", teacher_plan_sha256, True),
        (
            "orchestration_plan_sha256",
            orchestration_plan_sha256,
            not engineering_smoke,
        ),
    ):
        if value is None and not required:
            continue
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PredictedProfileRunError(
                f"{label} must be a lowercase SHA-256 digest"
            )
    if len(tasks) != 5:
        raise PredictedProfileRunError("downstream profile training requires five plan tasks")
    ordered_tasks = sorted(tasks, key=lambda item: int(item["fold_index"]))
    if [int(item["fold_index"]) for item in ordered_tasks] != list(range(5)):
        raise PredictedProfileRunError("downstream teacher tasks must cover folds 0 through 4")
    for field in ("domain", "model_family", "data_seed", "base_config_sha256"):
        if len({item.get(field) for item in ordered_tasks}) != 1:
            raise PredictedProfileRunError(f"downstream teacher tasks disagree on {field}")
    task = ordered_tasks[0]
    expected_teacher_task_sha256_by_fold = {
        int(item["fold_index"]): str(item["task_sha256"])
        for item in ordered_tasks
    }
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite predicted-profile run: {output_dir}")
    resolved = resolve_predicted_profile_config(
        base_config,
        task,
        device="cpu" if engineering_smoke else device,
        safety_weight=safety_weight,
        engineering_smoke=engineering_smoke,
    )
    data_seed = int(task["data_seed"])
    split_spec = build_teacher_split_spec(
        resolved,
        data_seed=data_seed,
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
        expected_data_seed=data_seed,
        expected_plan_sha256=teacher_plan_sha256,
        expected_base_config_sha256=str(task["base_config_sha256"]),
        expected_action_grid=resolved.data.actions,
        expected_profile_horizon=resolved.data.action_profile_horizon,
        expected_margin_scale=resolved.objective.margin_scale,
        expected_sample_ids=expected_training_samples,
        allow_engineering_smoke=engineering_smoke,
    )
    if dict(label_index.teacher_task_sha256_by_fold) != (
        expected_teacher_task_sha256_by_fold
    ):
        raise PredictedProfileRunError(
            "assembled label teacher inventory disagrees with the verified teacher plan"
        )
    coverage = load_profile_coverage_gate(
        coverage_manifest_path,
        expected_domain=resolved.data.task,
        expected_model_family=resolved.model.family,
        expected_data_seed=data_seed,
        expected_plan_sha256=teacher_plan_sha256,
        expected_assembly_manifest_sha256=label_index.manifest_sha256,
        expected_action_grid=resolved.data.actions,
        expected_profile_horizon=resolved.data.action_profile_horizon,
        expected_margin_scale=resolved.objective.margin_scale,
        expected_teacher_task_sha256_by_fold=expected_teacher_task_sha256_by_fold,
        allow_engineering_failure=engineering_smoke,
    )
    coverage_assembly = coverage.get("assembly")
    if (
        not isinstance(coverage_assembly, dict)
        or coverage_assembly.get("file_sha256")
        != file_sha256(assembled_manifest_path)
    ):
        raise PredictedProfileRunError(
            "coverage gate assembly file checksum disagrees with the supplied labels"
        )
    scientific_gate_passed = bool(coverage["scientific_gate_passed"])
    evidence_eligible = (
        not engineering_smoke
        and execution_evidence_eligible
        and label_index.evidence_eligible
        and scientific_gate_passed
        and bool(coverage.get("evidence_eligible"))
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    run_manifest_path = output_dir / "run_manifest.json"
    started = time.monotonic()
    resolved_sha = hashlib.sha256(
        canonical_config_json(resolved).encode("utf-8")
    ).hexdigest()
    run_manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": resolved.experiment,
        "semantic_arm": PREDICTED_PROFILE_ARM,
        "status": "initializing",
        "engineering_smoke": engineering_smoke,
        "evidence_eligible": evidence_eligible,
        "teacher_plan_sha256": teacher_plan_sha256,
        "orchestration_plan_sha256": orchestration_plan_sha256,
        "task_cell": {
            "domain": task["domain"],
            "model_family": task["model_family"],
            "data_seed": data_seed,
        },
        "resolved_config": {"sha256": resolved_sha, "values": resolved.to_dict()},
        "label_manifest": {
            "manifest_sha256": label_index.manifest_sha256,
            "file_sha256": file_sha256(assembled_manifest_path),
        },
        "coverage_gate": {
            "manifest_sha256": coverage["verified_manifest_sha256"],
            "file_sha256": file_sha256(coverage_manifest_path),
            "scientific_gate_passed": scientific_gate_passed,
        },
        "isolation": {
            "training_source": "crossfitted_predicted_action_profile_target",
            "oracle_action_safety_margins_forbidden": True,
            "ordinary_validation_safety_target": "none_world_model_utility_only",
            "calibration_materialized": False,
            "final_test_materialized": False,
        },
        "execution_code": dict(execution_code),
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
        train_dataset = build_ingested_profile_training_dataset(
            modules,
            resolved,
            data_seed=data_seed,
            labels=label_index,
        )
        (
            validation_dataset,
            validation_audit_dataset,
            validation_trajectories,
            validation_ids,
        ) = _build_validation_dataset(
            modules,
            resolved,
            data_seed=data_seed,
        )
        if set(validation_ids) & set(split_spec.all_training_ids):
            raise PredictedProfileRunError("training and validation trajectories overlap")
        bundle = _DownstreamBundle(
            datasets={"train": train_dataset, "validation": validation_dataset},
            split_trajectory_ids={
                "train": split_spec.all_training_ids,
                "validation": validation_ids,
            },
        )
        loaders, _ = _make_loaders(
            modules,
            bundle,
            resolved,
            pin_memory=selected_device.type == "cuda",
        )
        dataset_manifest = {
            "schema_version": 1,
            "semantic_arm": PREDICTED_PROFILE_ARM,
            "data_seed": data_seed,
            "training": {
                "trajectory_count": len(split_spec.all_training_ids),
                "trajectory_ids_sha256": split_spec.all_training_ids_sha256,
                "sample_count": len(train_dataset),
                "sample_ids_sha256": ids_sha256(train_dataset.sample_ids),
                "target_key": "predicted_action_profile_target",
                "label_manifest_sha256": label_index.manifest_sha256,
            },
            "ordinary_validation": {
                "trajectory_count": len(validation_ids),
                "trajectory_ids_sha256": ids_sha256(validation_ids),
                "sample_count": len(validation_dataset),
                "predicted_profile_target_present": False,
                "observed_margin_present_but_not_used_for_selection": True,
                "postfit_physical_profile_audit_planned": True,
            },
            "not_materialized": ["calibration", "test"],
            "oracle_action_profiles_materialized": False,
        }
        dataset_manifest_sha = canonical_sha256(dataset_manifest)
        dataset_path = output_dir / "dataset_manifest.json"
        write_json_atomic(dataset_path, dataset_manifest)

        model = build_world_model(modules, resolved.model, resolved.data).to(selected_device)
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
        utility_config = dataclasses.replace(
            resolved,
            objective=dataclasses.replace(
                resolved.objective,
                safety_arm="none",
                safety_weight=0.0,
            ),
        )
        model_info = model_metadata(model, resolved.model, resolved.data)
        run_manifest.update(
            {
                "status": "running",
                "dataset_manifest_sha256": dataset_manifest_sha,
                "model": model_info,
                "environment": {
                    "torch": str(torch.__version__),
                    "device": device_info.selected,
                    "accelerator_name": device_info.accelerator_name,
                    "determinism": determinism,
                    "amp_enabled": amp_enabled,
                },
                "selection": {
                    "metric": "ordinary_validation.world_model_utility",
                    "mode": "min",
                    "tie_break": "earliest_epoch",
                    "predicted_profile_target_used_for_selection": False,
                    "calibration_or_test_used": False,
                },
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)

        best_path = output_dir / "checkpoint_best.pt"
        last_path = output_dir / "checkpoint_last.pt"
        history: list[dict[str, object]] = []
        best_metric = math.inf
        best_epoch = -1
        best_validation: dict[str, float | int] | None = None
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
                utility_config,
                split="ordinary_validation",
                collect_records=False,
            )
            validation_metrics = _with_world_model_utility(
                validation_metrics, utility_config
            )
            row: dict[str, object] = {
                "epoch": epoch,
                "train": train_metrics,
                "ordinary_validation": validation_metrics,
            }
            history.append(row)
            if epoch == 1 or epoch % resolved.run.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            selection_value = float(validation_metrics["world_model_utility"])
            checkpoint = {
                "schema_version": 1,
                "experiment": resolved.experiment,
                "semantic_arm": PREDICTED_PROFILE_ARM,
                "epoch": epoch,
                "config": resolved.to_dict(),
                "teacher_plan_sha256": teacher_plan_sha256,
                "orchestration_plan_sha256": orchestration_plan_sha256,
                "label_manifest_sha256": label_index.manifest_sha256,
                "coverage_manifest_sha256": coverage["verified_manifest_sha256"],
                "dataset_manifest_sha256": dataset_manifest_sha,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ordinary_validation": validation_metrics,
            }
            _save_checkpoint_atomic(torch, last_path, checkpoint)
            if selection_value < best_metric:
                best_metric = selection_value
                best_epoch = epoch
                best_validation = validation_metrics
                _save_checkpoint_atomic(torch, best_path, checkpoint)
        if best_epoch < 1 or best_validation is None:
            raise RuntimeError("predicted-profile arm produced no selectable checkpoint")
        history_path = output_dir / "history.jsonl"
        _write_jsonl_atomic(history_path, history)
        selected = torch.load(best_path, map_location=selected_device, weights_only=False)
        model.load_state_dict(selected["model_state"])
        selected_checkpoint_sha = _sha256(best_path)
        postfit_validation, postfit_validation_path = run_validation_postfit_audit(
            modules,
            model,
            validation_audit_dataset,
            validation_trajectories,
            selected_device,
            resolved,
            output_dir=output_dir,
            selected_checkpoint_sha256=selected_checkpoint_sha,
            semantic_arm=PREDICTED_PROFILE_ARM,
            ordinary_validation_metrics=best_validation,
            include_profile_prediction_diagnostics=True,
        )
        checkpoint_manifest = {
            "schema_version": 1,
            "status": "success",
            "semantic_arm": PREDICTED_PROFILE_ARM,
            "evidence_eligible": evidence_eligible,
            "teacher_plan_sha256": teacher_plan_sha256,
            "orchestration_plan_sha256": orchestration_plan_sha256,
            "resolved_config_sha256": resolved_sha,
            "dataset_manifest_sha256": dataset_manifest_sha,
            "label_manifest_sha256": label_index.manifest_sha256,
            "coverage_manifest_sha256": coverage["verified_manifest_sha256"],
            "selection": {
                "metric": "ordinary_validation.world_model_utility",
                "selected_epoch": best_epoch,
                "selected_value": best_metric,
                "predicted_profile_target_used_for_selection": False,
                "calibration_or_test_used": False,
            },
            "postfit_validation_audit": {
                "path": postfit_validation_path.name,
                "sha256": file_sha256(postfit_validation_path),
                "manifest_sha256": postfit_validation["manifest_sha256"],
                "used_for_fitting_or_checkpoint_selection": False,
                "calibration_or_test_used": False,
            },
            "checkpoints": {
                "best": {"path": best_path.name, "sha256": selected_checkpoint_sha},
                "last": {"path": last_path.name, "sha256": _sha256(last_path)},
            },
            "state_hashes": {
                "selected_model_state": _state_sha256(torch, selected["model_state"]),
                "selected_optimizer_state": _state_sha256(
                    torch, selected["optimizer_state"]
                ),
            },
        }
        checkpoint_manifest_path = output_dir / "checkpoint_manifest.json"
        write_json_atomic(checkpoint_manifest_path, checkpoint_manifest)
        run_manifest.update(
            {
                "status": "success",
                "wall_time_seconds": time.monotonic() - started,
                "selected_epoch": best_epoch,
                "best_validation_world_model_utility": best_metric,
                "checkpoint_manifest": {
                    "path": checkpoint_manifest_path.name,
                    "sha256": file_sha256(checkpoint_manifest_path),
                },
                "history": {
                    "path": history_path.name,
                    "sha256": file_sha256(history_path),
                },
                "postfit_validation_audit": {
                    "path": postfit_validation_path.name,
                    "sha256": file_sha256(postfit_validation_path),
                    "manifest_sha256": postfit_validation["manifest_sha256"],
                },
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        return {
            "status": "success",
            "semantic_arm": PREDICTED_PROFILE_ARM,
            "selected_epoch": best_epoch,
            "training_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
            "validation_postfit_audit_count": postfit_validation["sample_count"],
            "coverage_gate_passed": scientific_gate_passed,
            "evidence_eligible": evidence_eligible,
            "output_dir": str(output_dir),
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
