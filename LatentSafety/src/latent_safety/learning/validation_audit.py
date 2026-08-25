"""Reusable frozen-checkpoint validation audit with explicit physical rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_safety.learning.config import LearningConfig
from latent_safety.learning.data import (
    ControlledVideoDataset,
    observation_feature_vector,
    observed_constant_action_profile,
    state_feature_vector,
)
from latent_safety.manifest import write_json_atomic
from latent_safety.records import AuditRecord, write_jsonl

VALIDATION_POSTFIT_MANIFEST = "validation_postfit_manifest.json"
VALIDATION_AUDIT_RECORDS = "audit_validation.jsonl"


class ValidationAuditError(ValueError):
    """Raised when a validation-only post-fit audit violates its access boundary."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(values: Sequence[str]) -> str:
    return _canonical_sha256(sorted(values))


def _select_rollout_refs(
    dataset: ControlledVideoDataset,
    config: LearningConfig,
) -> tuple[Any, ...]:
    candidates = dataset.rollout_refs(max(config.evaluation.rollout_horizons))
    maximum = config.evaluation.max_rollout_cases
    if len(candidates) <= maximum:
        return candidates
    return tuple(
        candidates[(index * len(candidates)) // maximum] for index in range(maximum)
    )


def _evaluate_validation_rollouts(
    modules: Any,
    model: Any,
    dataset: ControlledVideoDataset,
    device: Any,
    config: LearningConfig,
) -> dict[str, dict[str, float | int]]:
    horizons = config.evaluation.rollout_horizons
    refs = _select_rollout_refs(dataset, config)
    sums = {
        horizon: {"latent_mse": 0.0, "pixel_mse": 0.0, "cases": 0}
        for horizon in horizons
    }
    model.eval()
    with modules.torch.no_grad():
        for ref in refs:
            initial, actions, targets = dataset.rollout_case(ref, horizons)
            latent = model.encode(initial.unsqueeze(0).to(device), sample=False)["z"]
            actions = actions.to(device)
            for step in range(1, max(horizons) + 1):
                latent = model.predict_next(latent, actions[step - 1].unsqueeze(0))
                if step not in sums:
                    continue
                target_history = targets[step].unsqueeze(0).to(device)
                target_latent = model.encode(target_history, sample=False)["z"]
                decoded = model.decoder(latent)
                latent_mse = (latent - target_latent).square().mean()
                pixel_mse = (decoded - target_history[:, -1]).square().mean()
                sums[step]["latent_mse"] += float(latent_mse.item())
                sums[step]["pixel_mse"] += float(pixel_mse.item())
                sums[step]["cases"] += 1
    if any(int(values["cases"]) <= 0 for values in sums.values()):
        raise ValidationAuditError("validation rollout audit has an empty horizon")
    return {
        str(horizon): {
            "latent_mse": values["latent_mse"] / int(values["cases"]),
            "pixel_mse": values["pixel_mse"] / int(values["cases"]),
            "cases": int(values["cases"]),
        }
        for horizon, values in sums.items()
    }


def run_validation_postfit_audit(
    modules: Any,
    model: Any,
    validation_dataset: ControlledVideoDataset,
    validation_trajectories: Sequence[Any],
    device: Any,
    config: LearningConfig,
    *,
    output_dir: Path,
    selected_checkpoint_sha256: str,
    semantic_arm: str,
    ordinary_validation_metrics: Mapping[str, float | int],
    include_profile_prediction_diagnostics: bool = False,
) -> tuple[dict[str, object], Path]:
    """Audit a selected checkpoint without materializing calibration or final test.

    Known-dynamics profile-label construction remains disabled.  Each validation target is instead
    produced after checkpoint selection by explicitly executing every constant-action physical
    branch and recording its observed margins at ``t=0,...,H``.
    """

    if len(selected_checkpoint_sha256) != 64:
        raise ValidationAuditError("selected checkpoint SHA-256 is malformed")
    if not semantic_arm:
        raise ValidationAuditError("semantic_arm must be non-empty")
    audit_path = output_dir / VALIDATION_AUDIT_RECORDS
    manifest_path = output_dir / VALIDATION_POSTFIT_MANIFEST
    if audit_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite validation post-fit audit artifacts")
    if any(trajectory.split != "validation" for trajectory in validation_trajectories):
        raise ValidationAuditError("post-fit audit received a non-validation trajectory")
    if any(trajectory.action_safety_margins for trajectory in validation_trajectories):
        raise ValidationAuditError("post-fit audit materialized oracle action profiles")
    trajectory_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in validation_trajectories
    }
    if len(trajectory_by_id) != len(validation_trajectories):
        raise ValidationAuditError("post-fit validation trajectories contain duplicate IDs")
    loader = modules.torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.run.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    records: list[AuditRecord] = []
    component_error_sum = 0.0
    component_count = 0
    sign_disagreements = 0
    false_safe_count = 0
    model.eval()
    with modules.torch.no_grad():
        for raw_batch in loader:
            if "action_safety_margins" in raw_batch:
                raise ValidationAuditError(
                    "post-fit validation loader exposed a precomputed oracle profile"
                )
            histories = raw_batch["history"].to(device)
            actions = raw_batch["action"].to(device)
            outputs = model(histories, actions)
            latents = outputs["mean"].detach().cpu().tolist()
            predictions = (
                outputs["predicted_action_profile"].detach().cpu().tolist()
                if include_profile_prediction_diagnostics
                else [None] * len(latents)
            )
            for (
                sample_id,
                trajectory_id,
                timestep,
                margin,
                latent,
                prediction,
            ) in zip(
                raw_batch["sample_id"],
                raw_batch["trajectory_id"],
                raw_batch["timestep"].tolist(),
                raw_batch["safety_margin"].tolist(),
                latents,
                predictions,
                strict=True,
            ):
                trajectory = trajectory_by_id.get(str(trajectory_id))
                if trajectory is None:
                    raise ValidationAuditError(
                        "post-fit validation sample has an unknown trajectory"
                    )
                target = observed_constant_action_profile(
                    trajectory.states[int(timestep)],
                    config.data,
                )
                if prediction is not None:
                    if len(target) != len(prediction):
                        raise ValidationAuditError(
                            "post-fit prediction and observed target dimensions disagree"
                        )
                    for predicted, observed in zip(prediction, target, strict=True):
                        component_error_sum += abs(float(predicted) - observed)
                        component_count += 1
                        sign_disagreements += (float(predicted) >= 0.0) != (
                            observed >= 0.0
                        )
                        false_safe_count += float(predicted) >= 0.0 and observed < 0.0
                records.append(
                    AuditRecord(
                        sample_id=str(sample_id),
                        trajectory_id=str(trajectory_id),
                        split="validation",
                        safety_margin=float(margin),
                        latent=tuple(float(value) for value in latent),
                        observation_latent=observation_feature_vector(
                            trajectory.states,
                            int(timestep),
                            config.data,
                        ),
                        state_latent=state_feature_vector(
                            trajectory.states[int(timestep)],
                            config.data,
                        ),
                        action_safety_margins=target,
                    )
                )
    if len(records) != len(validation_dataset) or not records:
        raise ValidationAuditError(
            "post-fit validation audit did not cover every validation sample"
        )
    write_jsonl(audit_path, records)
    rollout_metrics = _evaluate_validation_rollouts(
        modules,
        model,
        validation_dataset,
        device,
        config,
    )
    maximum_horizon = str(max(config.evaluation.rollout_horizons))
    profile_diagnostics: dict[str, float | int] | None = None
    if include_profile_prediction_diagnostics:
        if component_count == 0:
            raise ValidationAuditError("profile diagnostics contain no components")
        profile_diagnostics = {
            "componentwise_mae": component_error_sum / component_count,
            "sign_disagreement_rate": sign_disagreements / component_count,
            "false_safe_rate": false_safe_count / component_count,
            "component_count": component_count,
        }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "validation_postfit_audit_success",
        "semantic_arm": semantic_arm,
        "split_access": {
            "materialized": ["validation"],
            "not_materialized": ["calibration", "test"],
            "checkpoint_selection_complete_before_profile_targets": True,
        },
        "checkpoint_sha256": selected_checkpoint_sha256,
        "ordinary_validation_metrics": dict(ordinary_validation_metrics),
        "e2_utility_summary": {
            "reconstruction_mse": float(ordinary_validation_metrics["reconstruction"]),
            "maximum_horizon": int(maximum_horizon),
            "maximum_horizon_rollout_pixel_mse": float(
                rollout_metrics[maximum_horizon]["pixel_mse"]
            ),
        },
        "trajectory_count": len(validation_trajectories),
        "trajectory_ids_sha256": _ids_sha256(
            [trajectory.trajectory_id for trajectory in validation_trajectories]
        ),
        "sample_count": len(records),
        "sample_ids_sha256": _ids_sha256([record.sample_id for record in records]),
        "physical_profile_target": {
            "source": "explicit_observed_physical_rollouts",
            "constant_action_grid": list(config.data.actions),
            "profile_rollout_horizon": config.data.action_profile_horizon,
            "observed_timesteps": "t_0_through_H_inclusive",
            "branch_process_noise": 0.0,
            "action_safety_profile_helper_called": False,
            "used_for_fitting_or_checkpoint_selection": False,
        },
        "profile_prediction_diagnostics": profile_diagnostics,
        "rollout_metrics": rollout_metrics,
        "audit_records": {
            "path": audit_path.name,
            "sha256": _file_sha256(audit_path),
            "record_count": len(records),
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    return manifest, manifest_path
