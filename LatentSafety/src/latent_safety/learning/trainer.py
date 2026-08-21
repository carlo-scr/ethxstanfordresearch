"""Training, checkpointing, and evaluation for the optional PyTorch pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from latent_safety.learning.config import LearningConfig, canonical_config_json
from latent_safety.learning.data import ControlledCartDataset, DatasetBundle, build_datasets
from latent_safety.learning.losses import compute_losses
from latent_safety.learning.models import build_world_model, model_metadata
from latent_safety.learning.runtime import (
    TorchModules,
    require_torch,
    seed_data_loader_worker,
    seed_everything,
    select_device,
)
from latent_safety.manifest import base_manifest, write_json_atomic
from latent_safety.records import AuditRecord, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(torch: Any, state: Any) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer, _use_new_zipfile_serialization=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _save_checkpoint_atomic(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _make_loaders(
    modules: TorchModules,
    bundle: DatasetBundle,
    config: LearningConfig,
    *,
    pin_memory: bool,
) -> dict[str, Any]:
    torch = modules.torch
    generator = torch.Generator()
    generator.manual_seed(config.run.seed)
    loaders: dict[str, Any] = {}
    for split, dataset in bundle.datasets.items():
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.run.batch_size,
            shuffle=split == "train",
            num_workers=config.run.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.run.num_workers > 0,
            worker_init_fn=seed_data_loader_worker,
            generator=generator if split == "train" else None,
            drop_last=False,
        )
    return loaders


def _empty_accumulator() -> dict[str, float]:
    return {
        "total": 0.0,
        "reconstruction": 0.0,
        "transition": 0.0,
        "kl": 0.0,
        "safety": 0.0,
        "margin_mae": 0.0,
        "action_profile_mae": 0.0,
        "margin_sign_accuracy": 0.0,
        "examples": 0.0,
    }


def _update_accumulator(
    accumulator: dict[str, float],
    losses: dict[str, Any],
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> None:
    batch_size = int(batch["history"].shape[0])
    for name in ("total", "reconstruction", "transition", "kl", "safety"):
        accumulator[name] += float(losses[name].detach().item()) * batch_size
    margin_error = (
        outputs["predicted_margin"] - batch["safety_margin"]
    ).abs().mean()
    action_error = (
        outputs["predicted_action_profile"] - batch["action_safety_margins"]
    ).abs().mean()
    sign_accuracy = (
        (outputs["predicted_margin"] >= 0.0)
        == (batch["safety_margin"] >= 0.0)
    ).float().mean()
    accumulator["margin_mae"] += float(margin_error.detach().item()) * batch_size
    accumulator["action_profile_mae"] += float(action_error.detach().item()) * batch_size
    accumulator["margin_sign_accuracy"] += float(sign_accuracy.detach().item()) * batch_size
    accumulator["examples"] += batch_size


def _finalize_accumulator(accumulator: dict[str, float]) -> dict[str, float | int]:
    examples = int(accumulator["examples"])
    if examples == 0:
        raise RuntimeError("evaluation received an empty split")
    return {
        name: value / examples
        for name, value in accumulator.items()
        if name != "examples"
    } | {"examples": examples}


def _with_world_model_utility(
    metrics: dict[str, float | int], config: LearningConfig
) -> dict[str, float | int]:
    utility = (
        config.objective.reconstruction_weight * float(metrics["reconstruction"])
        + config.objective.transition_weight * float(metrics["transition"])
        + config.objective.kl_weight * float(metrics["kl"])
    )
    reported = {**metrics, "world_model_utility": utility}
    if config.objective.safety_arm != "h_prediction":
        reported.pop("margin_mae", None)
        reported.pop("margin_sign_accuracy", None)
    if config.objective.safety_arm != "safe_action_profile":
        reported.pop("action_profile_mae", None)
    return reported


def _train_epoch(
    modules: TorchModules,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    config: LearningConfig,
) -> dict[str, float | int]:
    torch = modules.torch
    model.train()
    accumulator = _empty_accumulator()
    amp_enabled = config.run.amp and device.type == "cuda"
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast:
            outputs = model(batch["history"], batch["action"])
            with torch.no_grad():
                target_latent = model.encode(batch["next_history"], sample=False)["z"]
            losses = compute_losses(
                modules,
                outputs,
                target_latent,
                batch,
                config.objective,
            )
        if not bool(torch.isfinite(losses["total"]).item()):
            raise FloatingPointError("encountered a non-finite training objective")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.run.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        _update_accumulator(accumulator, losses, outputs, batch)
    return _finalize_accumulator(accumulator)


def _evaluate_loader(
    modules: TorchModules,
    model: Any,
    loader: Any,
    device: Any,
    config: LearningConfig,
    *,
    split: str,
    collect_records: bool,
) -> tuple[dict[str, float | int], tuple[AuditRecord, ...]]:
    torch = modules.torch
    model.eval()
    accumulator = _empty_accumulator()
    records: list[AuditRecord] = []
    record_limit = config.evaluation.max_audit_records_per_split
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch["history"], batch["action"])
            target_latent = model.encode(batch["next_history"], sample=False)["z"]
            losses = compute_losses(
                modules,
                outputs,
                target_latent,
                batch,
                config.objective,
            )
            _update_accumulator(accumulator, losses, outputs, batch)
            if collect_records and len(records) < record_limit:
                means = outputs["mean"].detach().cpu().tolist()
                margins = batch["safety_margin"].detach().cpu().tolist()
                profiles = batch["action_safety_margins"].detach().cpu().tolist()
                observation_features = raw_batch["observation_features"].tolist()
                state_features = raw_batch["state_features"].tolist()
                remaining = record_limit - len(records)
                for sample_id, trajectory_id, latent, observation, state, margin, profile in zip(
                    raw_batch["sample_id"][:remaining],
                    raw_batch["trajectory_id"][:remaining],
                    means[:remaining],
                    observation_features[:remaining],
                    state_features[:remaining],
                    margins[:remaining],
                    profiles[:remaining],
                    strict=True,
                ):
                    records.append(
                        AuditRecord(
                            sample_id=str(sample_id),
                            trajectory_id=str(trajectory_id),
                            split=split,
                            safety_margin=float(margin),
                            latent=tuple(float(value) for value in latent),
                            observation_latent=tuple(
                                float(value) for value in observation
                            ),
                            state_latent=tuple(float(value) for value in state),
                            action_safety_margins=tuple(
                                float(value) for value in profile
                            ),
                        )
                    )
    return _finalize_accumulator(accumulator), tuple(records)


def _select_rollout_refs(dataset: ControlledCartDataset, config: LearningConfig) -> tuple[Any, ...]:
    candidates = dataset.rollout_refs(max(config.evaluation.rollout_horizons))
    maximum = config.evaluation.max_rollout_cases
    if len(candidates) <= maximum:
        return candidates
    # Even coverage avoids silently selecting only early trajectory IDs.
    return tuple(candidates[(index * len(candidates)) // maximum] for index in range(maximum))


def _evaluate_rollouts(
    modules: TorchModules,
    model: Any,
    dataset: ControlledCartDataset,
    device: Any,
    config: LearningConfig,
) -> dict[str, dict[str, float | int]]:
    torch = modules.torch
    horizons = config.evaluation.rollout_horizons
    refs = _select_rollout_refs(dataset, config)
    sums = {
        horizon: {"latent_mse": 0.0, "pixel_mse": 0.0, "cases": 0}
        for horizon in horizons
    }
    model.eval()
    with torch.no_grad():
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
    return {
        str(horizon): {
            "latent_mse": values["latent_mse"] / max(1, int(values["cases"])),
            "pixel_mse": values["pixel_mse"] / max(1, int(values["cases"])),
            "cases": int(values["cases"]),
        }
        for horizon, values in sums.items()
    }


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def run_experiment(
    config: LearningConfig,
    *,
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute one training arm and emit self-contained provenance artifacts."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty run directory: {output_dir}; "
            "pass a new --output path"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    started = time.monotonic()
    resolved_config_json = canonical_config_json(config)
    resolved_config_sha256 = hashlib.sha256(
        resolved_config_json.encode("utf-8")
    ).hexdigest()
    run_manifest = base_manifest(repo_root=repo_root, config_path=config_path)
    run_manifest.update(
        {
            "experiment": config.experiment,
            "status": "initializing",
            "seed": config.run.seed,
            "training_arm": config.objective.safety_arm,
            "resolved_config": {
                "sha256": resolved_config_sha256,
                "values": config.to_dict(),
            },
            "failure": None,
        }
    )
    write_json_atomic(run_manifest_path, run_manifest)
    try:
        modules = require_torch()
        torch = modules.torch
        device, device_info = select_device(torch, config.run.device)
        determinism = seed_everything(
            torch,
            seed=config.run.seed,
            deterministic=config.run.deterministic,
            warn_only=config.run.deterministic_warn_only,
        )
        bundle = build_datasets(config.data, seed=config.run.seed, torch=torch)
        dataset_manifest_path = output_dir / "dataset_manifest.json"
        write_json_atomic(dataset_manifest_path, bundle.manifest_payload)
        loaders = _make_loaders(
            modules,
            bundle,
            config,
            pin_memory=device.type == "cuda",
        )
        model = build_world_model(modules, config.model, config.data).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.run.learning_rate,
            weight_decay=config.run.weight_decay,
        )
        amp_enabled = config.run.amp and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        model_info = model_metadata(model, config.model, config.data)
        run_manifest.update(
            {
                "status": "running",
                "dataset": {
                    "manifest_sha256": bundle.manifest_sha256,
                    "manifest_path": str(dataset_manifest_path),
                    "split_trajectory_ids": bundle.split_trajectory_ids,
                    "split_sample_counts": {
                        split: len(dataset) for split, dataset in bundle.datasets.items()
                    },
                },
                "model": model_info,
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": config.run.learning_rate,
                    "weight_decay": config.run.weight_decay,
                },
                "environment": {
                    **run_manifest["environment"],
                    "torch": str(torch.__version__),
                    "cuda_runtime": str(torch.version.cuda),
                    "device": device_info.selected,
                    "accelerator_name": device_info.accelerator_name,
                    "requested_device": device_info.requested,
                    "amp_enabled": amp_enabled,
                    "determinism": determinism,
                },
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)

        best_path = output_dir / "checkpoint_best.pt"
        last_path = output_dir / "checkpoint_last.pt"
        history_path = output_dir / "history.jsonl"
        history_path.write_text("", encoding="utf-8")
        best_metric = math.inf
        best_epoch = -1
        best_validation: dict[str, float | int] | None = None
        for epoch in range(1, config.run.epochs + 1):
            train_metrics = _with_world_model_utility(
                _train_epoch(
                    modules,
                    model,
                    loaders["train"],
                    optimizer,
                    scaler,
                    device,
                    config,
                ),
                config,
            )
            validation_metrics, _ = _evaluate_loader(
                modules,
                model,
                loaders["validation"],
                device,
                config,
                split="validation",
                collect_records=False,
            )
            validation_metrics = _with_world_model_utility(
                validation_metrics,
                config,
            )
            epoch_payload = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            _append_history(history_path, epoch_payload)
            if epoch == 1 or epoch % config.run.log_every == 0 or epoch == config.run.epochs:
                print(json.dumps(epoch_payload, sort_keys=True), flush=True)
            selection_value = float(validation_metrics["world_model_utility"])
            checkpoint_payload = {
                "schema_version": 1,
                "experiment": config.experiment,
                "epoch": epoch,
                "config": config.to_dict(),
                "dataset_manifest_sha256": bundle.manifest_sha256,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "validation": validation_metrics,
            }
            _save_checkpoint_atomic(torch, last_path, checkpoint_payload)
            if selection_value < best_metric:
                best_metric = selection_value
                best_epoch = epoch
                best_validation = validation_metrics
                _save_checkpoint_atomic(torch, best_path, checkpoint_payload)

        if best_epoch < 0 or best_validation is None:
            raise RuntimeError("training completed without a selectable checkpoint")
        selected_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(selected_checkpoint["model_state"])

        split_metrics: dict[str, Any] = {}
        audit_paths: dict[str, str] = {}
        for split in ("train", "validation", "calibration", "test"):
            collect = config.evaluation.emit_audit_records and split in {
                "validation",
                "calibration",
                "test",
            }
            metrics, records = _evaluate_loader(
                modules,
                model,
                loaders[split],
                device,
                config,
                split=split,
                collect_records=collect,
            )
            split_metrics[split] = _with_world_model_utility(metrics, config)
            if records:
                audit_path = output_dir / f"audit_{split}.jsonl"
                write_jsonl(audit_path, records)
                audit_paths[split] = str(audit_path)
        rollout_metrics = {
            split: _evaluate_rollouts(
                modules,
                model,
                bundle.datasets[split],
                device,
                config,
            )
            for split in ("validation", "test")
        }

        checkpoint_manifest = base_manifest(repo_root=repo_root, config_path=config_path)
        checkpoint_manifest.update(
            {
                "experiment": config.experiment,
                "status": "success",
                "selection": {
                    "metric": "validation.world_model_utility",
                    "mode": "min",
                    "selected_epoch": best_epoch,
                    "selected_value": best_metric,
                    "test_labels_used": False,
                },
                "dataset_manifest_sha256": bundle.manifest_sha256,
                "resolved_config_sha256": resolved_config_sha256,
                "checkpoints": {
                    "best": {"path": str(best_path), "sha256": _sha256(best_path)},
                    "last": {"path": str(last_path), "sha256": _sha256(last_path)},
                },
                "state_hashes": {
                    "selected_model_state": _state_sha256(
                        torch, selected_checkpoint["model_state"]
                    ),
                    "selected_optimizer_state": _state_sha256(
                        torch, selected_checkpoint["optimizer_state"]
                    ),
                },
                "parent_checkpoint": None,
            }
        )
        checkpoint_manifest_path = output_dir / "checkpoint_manifest.json"
        write_json_atomic(checkpoint_manifest_path, checkpoint_manifest)

        evaluation_manifest = base_manifest(repo_root=repo_root, config_path=config_path)
        evaluation_manifest.update(
            {
                "experiment": config.experiment,
                "status": "success",
                "checkpoint_manifest": str(checkpoint_manifest_path),
                "checkpoint_sha256": _sha256(best_path),
                "dataset_manifest_sha256": bundle.manifest_sha256,
                "resolved_config_sha256": resolved_config_sha256,
                "split_metrics": split_metrics,
                "rollout_metrics": rollout_metrics,
                "audit_records": audit_paths,
            }
        )
        evaluation_manifest_path = output_dir / "evaluation_manifest.json"
        write_json_atomic(evaluation_manifest_path, evaluation_manifest)

        run_manifest.update(
            {
                "status": "success",
                "wall_time_seconds": time.monotonic() - started,
                "selected_epoch": best_epoch,
                "checkpoint_manifest": str(checkpoint_manifest_path),
                "evaluation_manifest": str(evaluation_manifest_path),
                "history": str(history_path),
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "selected_epoch": best_epoch,
            "best_validation_world_model_utility": best_metric,
            "test_metrics": split_metrics["test"],
        }
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "wall_time_seconds": time.monotonic() - started,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        raise
