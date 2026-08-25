"""Training, checkpointing, and evaluation for the optional PyTorch pipeline."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from latent_safety.learning.config import LearningConfig, canonical_config_json
from latent_safety.learning.data import ControlledCartDataset, DatasetBundle, build_datasets
from latent_safety.learning.fcsrl_protocol import (
    FCSRL_ATOM_COUNT,
    FCSRL_DISCOUNT,
    FCSRL_ENCODER_EMA_SOURCE_MIX,
    FCSRL_RETURN_LENGTH,
    FCSRL_UNROLL_LENGTH,
)
from latent_safety.learning.fcsrl_training import (
    ControlledFCSRLWindowDataset,
    build_categorical_feasibility_head,
    build_ema_target_encoder,
    compute_fcsrl_feasibility_loss,
    iter_fcsrl_batches,
    update_ema_target_encoder,
)
from latent_safety.learning.losses import compute_losses
from latent_safety.learning.models import build_world_model, model_metadata
from latent_safety.learning.runtime import (
    TorchModules,
    require_torch,
    seed_data_loader_worker,
    seed_everything,
    select_device,
)
from latent_safety.learning.validation_audit import (
    VALIDATION_AUDIT_RECORDS,
    run_validation_postfit_audit,
)
from latent_safety.manifest import base_manifest, write_json_atomic
from latent_safety.records import AuditRecord, write_jsonl

FULL_EVALUATION_ACCESS = "full_evaluation"
TRAIN_VALIDATION_ONLY_ACCESS = "train_validation_only"
EXPERIMENT_ACCESS_SCOPES = (
    FULL_EVALUATION_ACCESS,
    TRAIN_VALIDATION_ONLY_ACCESS,
)


def experiment_access_scope_payload(
    access_scope: str,
    config: LearningConfig,
) -> dict[str, Any]:
    """Resolve a frozen data-access boundary before any trajectory is generated."""

    if access_scope not in EXPERIMENT_ACCESS_SCOPES:
        raise ValueError(
            f"access_scope must be one of {EXPERIMENT_ACCESS_SCOPES}, got {access_scope!r}"
        )
    if access_scope == FULL_EVALUATION_ACCESS:
        return {
            "name": access_scope,
            "materialized_splits": ["train", "validation", "calibration", "test"],
            "evaluated_splits": ["train", "validation", "calibration", "test"],
            "rollout_splits": ["validation", "test"],
            "oracle_action_profiles_materialized": True,
            "calibration_access": True,
            "final_test_access": True,
        }
    allowed_arms = {
        "none",
        "h_prediction",
        "boundary_contrastive",
        "fcsrl_feasibility_loss_adaptation",
    }
    if config.objective.safety_arm not in allowed_arms:
        raise ValueError(
            "train_validation_only access forbids objectives that require an oracle or "
            "externally ingested action-profile target"
        )
    return {
        "name": access_scope,
        "materialized_splits": ["train", "validation"],
        "evaluated_splits": ["train", "validation"],
        "rollout_splits": ["validation"],
        "oracle_action_profiles_materialized": False,
        "calibration_access": False,
        "final_test_access": False,
    }


def _validate_optional_sha256(value: str | None, *, label: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


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


def _canonical_tensor_state_sha256(torch: Any, state: dict[str, Any]) -> str:
    """Hash named tensors independently of PyTorch serialization storage identifiers."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"canonical tensor state contains non-tensor field {name!r}")
        contiguous = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _save_checkpoint_atomic(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _capture_training_state(torch: Any, scaler: Any, train_generator: Any) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_loader_generator_state": train_generator.get_state(),
        "scaler_state": scaler.state_dict(),
    }


def _restore_training_state(
    torch: Any,
    scaler: Any,
    train_generator: Any,
    state: Any,
) -> None:
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint is missing training_state")
    required = {
        "python_random_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "train_loader_generator_state",
        "scaler_state",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"resume training_state is missing fields: {missing}")
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_states = state["cuda_rng_state_all"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    train_generator.set_state(state["train_loader_generator_state"].cpu())
    scaler.load_state_dict(state["scaler_state"])


def _resume_configs_compatible(previous: Any, current: LearningConfig) -> bool:
    """Allow only a larger total epoch budget when continuing an exact run."""

    if not isinstance(previous, dict):
        return False
    candidate = copy.deepcopy(previous)
    run = candidate.get("run")
    if not isinstance(run, dict):
        return False
    previous_epochs = run.get("epochs")
    if isinstance(previous_epochs, bool) or not isinstance(previous_epochs, int):
        return False
    if previous_epochs > current.run.epochs:
        return False
    run["epochs"] = current.run.epochs
    return candidate == current.to_dict()


def _load_resume_pair(
    torch: Any,
    resume_from: Path,
    config: LearningConfig,
    *,
    dataset_manifest_sha256: str,
    data_access_scope: dict[str, Any],
    orchestration_plan_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not resume_from.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from}")
    if resume_from.name != "checkpoint_last.pt":
        raise ValueError("resume requires a checkpoint_last.pt continuation state")
    checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("resume checkpoint has an unsupported schema")
    if checkpoint.get("experiment") != config.experiment:
        raise ValueError("resume checkpoint experiment does not match the current config")
    if not _resume_configs_compatible(checkpoint.get("config"), config):
        raise ValueError(
            "resume checkpoint config differs from the current config beyond run.epochs"
        )
    if checkpoint.get("data_access_scope") != data_access_scope:
        raise ValueError("resume checkpoint data-access scope does not match")
    if checkpoint.get("orchestration_plan_sha256") != orchestration_plan_sha256:
        raise ValueError("resume checkpoint orchestration-plan identity does not match")
    if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("resume checkpoint dataset manifest does not match regenerated data")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("resume checkpoint epoch must be a positive integer")
    if epoch >= config.run.epochs:
        raise ValueError("resume checkpoint already reaches the configured epoch budget")
    for key in ("model_state", "optimizer_state", "training_state", "validation"):
        if key not in checkpoint:
            raise ValueError(f"resume checkpoint is missing {key}")

    best_path = resume_from.with_name("checkpoint_best.pt")
    if not best_path.is_file():
        raise FileNotFoundError(
            f"resume requires the sibling best checkpoint: {best_path}"
        )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(best, dict) or best.get("schema_version") != 1:
        raise ValueError("sibling best checkpoint has an unsupported schema")
    if best.get("experiment") != config.experiment:
        raise ValueError("sibling best checkpoint experiment does not match")
    if not _resume_configs_compatible(best.get("config"), config):
        raise ValueError("sibling best checkpoint config does not match")
    if best.get("data_access_scope") != data_access_scope:
        raise ValueError("sibling best checkpoint data-access scope does not match")
    if best.get("orchestration_plan_sha256") != orchestration_plan_sha256:
        raise ValueError("sibling best checkpoint orchestration-plan identity does not match")
    if best.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("sibling best checkpoint dataset manifest does not match")
    best_epoch = best.get("epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch < 1
        or best_epoch > epoch
    ):
        raise ValueError("sibling best checkpoint has an invalid selected epoch")
    if not isinstance(best.get("validation"), dict):
        raise ValueError("sibling best checkpoint is missing validation metrics")
    return checkpoint, best


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
) -> tuple[dict[str, Any], Any]:
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
    return loaders, generator


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
    action_error = None
    if "action_safety_margins" in batch and "predicted_action_profile_target" in batch:
        raise ValueError("batch cannot mix oracle and predicted action-profile targets")
    if "action_safety_margins" in batch:
        action_error = (
            outputs["predicted_action_profile"] - batch["action_safety_margins"]
        ).abs().mean()
    elif "predicted_action_profile_target" in batch:
        action_error = (
            outputs["predicted_action_profile"]
            - batch["predicted_action_profile_target"]
        ).abs().mean()
    sign_accuracy = (
        (outputs["predicted_margin"] >= 0.0)
        == (batch["safety_margin"] >= 0.0)
    ).float().mean()
    accumulator["margin_mae"] += float(margin_error.detach().item()) * batch_size
    if action_error is not None:
        accumulator["action_profile_mae"] += (
            float(action_error.detach().item()) * batch_size
        )
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
    if config.objective.safety_arm not in {
        "safe_action_profile",
        "nonprivileged_predicted_action_profile",
    }:
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


def _fcsrl_base_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Expose the first transition of a sequence window to the unchanged base objective."""

    return {
        "history": batch["histories"][:, 0],
        "next_history": batch["histories"][:, 1],
        "action": batch["actions"][:, 0],
        "safety_margin": batch["safety_margins"][:, 0],
    }


def _train_fcsrl_epoch(
    modules: TorchModules,
    model: Any,
    target_encoder: Any,
    feasibility_head: Any,
    window_dataset: ControlledFCSRLWindowDataset,
    optimizer: Any,
    scaler: Any,
    device: Any,
    config: LearningConfig,
    *,
    epoch: int,
) -> dict[str, float | int]:
    """Train one deterministic sequence epoch without changing the ordinary-arm path."""

    torch = modules.torch
    model.train()
    feasibility_head.train()
    target_encoder.eval()
    accumulator = _empty_accumulator()
    amp_enabled = config.run.amp and device.type == "cuda"
    active_positions = 0
    window_count = 0
    parameters = [
        parameter
        for module in (model, feasibility_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    for raw_batch in iter_fcsrl_batches(
        window_dataset,
        torch=torch,
        batch_size=config.run.batch_size,
        seed=config.run.seed,
        epoch=epoch - 1,
        shuffle=True,
        drop_last=False,
    ):
        batch = _move_batch(raw_batch, device)
        base_batch = _fcsrl_base_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast:
            outputs = model(base_batch["history"], base_batch["action"])
            with torch.no_grad():
                target_latent = model.encode(
                    base_batch["next_history"], sample=False
                )["z"]
            base_losses = compute_losses(
                modules,
                outputs,
                target_latent,
                base_batch,
                config.objective,
            )
        # The categorical recursion stays in float32 even when the unchanged base path uses AMP.
        feasibility = compute_fcsrl_feasibility_loss(
            modules,
            model,
            target_encoder,
            feasibility_head,
            batch,
        )
        total_loss = (
            base_losses["total"]
            + config.objective.safety_weight * feasibility.loss
        )
        if not bool(torch.isfinite(total_loss).item()):
            raise FloatingPointError("encountered a non-finite FCSRL training objective")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        for parameter in parameters:
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all().item()
            ):
                raise FloatingPointError("encountered a non-finite FCSRL gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.run.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("encountered a non-finite FCSRL gradient norm")
        scaler.step(optimizer)
        scaler.update()
        update_ema_target_encoder(torch, target_encoder, model)
        losses = {
            **base_losses,
            "total": total_loss,
            "safety": feasibility.loss,
        }
        _update_accumulator(accumulator, losses, outputs, base_batch)
        active_positions += feasibility.active_positions
        window_count += int(batch["histories"].shape[0])
    metrics = _finalize_accumulator(accumulator)
    metrics["fcsrl_active_positions"] = active_positions
    metrics["fcsrl_windows"] = window_count
    return metrics


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


def _encoder_state(model: Any) -> dict[str, Any]:
    prefixes = ("frame_encoder.", "history_encoder.", "posterior_mean.")
    return {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def _write_fcsrl_regression_artifact(
    modules: TorchModules,
    model: Any,
    target_encoder: Any,
    feasibility_head: Any,
    window_dataset: ControlledFCSRLWindowDataset,
    device: Any,
    config: LearningConfig,
    path: Path,
    *,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    """Persist one deterministic numerical fixture without promoting learned evidence."""

    torch = modules.torch
    raw_batch = next(
        iter_fcsrl_batches(
            window_dataset,
            torch=torch,
            batch_size=1,
            seed=config.run.seed,
            epoch=0,
            shuffle=False,
            drop_last=False,
        )
    )
    batch = _move_batch(raw_batch, device)
    model_was_training = model.training
    head_was_training = feasibility_head.training
    model.eval()
    feasibility_head.eval()
    with torch.no_grad():
        output = compute_fcsrl_feasibility_loss(
            modules,
            model,
            target_encoder,
            feasibility_head,
            batch,
        )
        target_histories = batch["histories"][:, 1:].reshape(
            FCSRL_RETURN_LENGTH, *batch["histories"].shape[2:]
        )
        ema_latents = target_encoder(target_histories)
    model.train(model_was_training)
    feasibility_head.train(head_was_training)

    online_encoder_state = _encoder_state(model)
    ema_encoder_state = target_encoder.state_dict()
    state_names_match = online_encoder_state.keys() == ema_encoder_state.keys()
    ema_matches_online = state_names_match and all(
        torch.equal(online_encoder_state[name], ema_encoder_state[name])
        for name in online_encoder_state
    )
    payload = {
        "schema_version": 1,
        "artifact": "fcsrl_component_regression_fixture_v1",
        "status": "engineering_regression_not_learned_model_evidence",
        "seed": config.run.seed,
        "checkpoint_epoch": checkpoint_epoch,
        "protocol": {
            "atom_count": FCSRL_ATOM_COUNT,
            "discount": FCSRL_DISCOUNT,
            "return_length": FCSRL_RETURN_LENGTH,
            "unroll_length": FCSRL_UNROLL_LENGTH,
            "encoder_ema_source_mix": FCSRL_ENCODER_EMA_SOURCE_MIX,
            "head_hidden_dim": config.objective.fcsrl_head_hidden_dim,
        },
        "window": {
            "trajectory_id": raw_batch["trajectory_id"][0],
            "start_timestep": raw_batch["start_timestep"][0],
            "sample_ids": list(raw_batch["sample_ids"][0]),
            "violations": batch["violations"][0].detach().cpu().tolist(),
            "terminations": batch["terminations"][0].detach().cpu().tolist(),
            "truncations": batch["truncations"][0].detach().cpu().tolist(),
        },
        "ema": {
            "online_encoder_state_sha256": _canonical_tensor_state_sha256(
                torch, online_encoder_state
            ),
            "target_encoder_state_sha256": _canonical_tensor_state_sha256(
                torch, ema_encoder_state
            ),
            "matches_online_at_fixture_time": ema_matches_online,
            "latents": ema_latents.detach().cpu().tolist(),
        },
        "fixture": {
            "bootstrap_values": output.bootstrap_values[0].detach().cpu().tolist(),
            "targets": output.target_trace.targets[0].detach().cpu().tolist(),
            "mask": output.target_trace.mask[0].detach().cpu().tolist(),
            "train_mask": output.target_trace.train_mask[0].detach().cpu().tolist(),
            "projected_targets": (
                output.target_trace.projected_targets[0].detach().cpu().tolist()
            ),
            "loss": float(output.loss.detach().cpu().item()),
        },
    }
    write_json_atomic(path, payload)
    return payload


def run_experiment(
    config: LearningConfig,
    *,
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
    resume_from: Path | None = None,
    access_scope: str = FULL_EVALUATION_ACCESS,
    orchestration_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute one training arm and emit self-contained provenance artifacts."""

    data_access_scope = experiment_access_scope_payload(access_scope, config)
    _validate_optional_sha256(
        orchestration_plan_sha256,
        label="orchestration_plan_sha256",
    )
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
            "data_access_scope": data_access_scope,
            "orchestration_plan_sha256": orchestration_plan_sha256,
            "resolved_config": {
                "sha256": resolved_config_sha256,
                "values": config.to_dict(),
            },
            "resume": (
                {
                    "path": str(resume_from),
                    "sha256": _sha256(resume_from) if resume_from.is_file() else None,
                }
                if resume_from is not None
                else None
            ),
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
        materialized_splits = tuple(data_access_scope["materialized_splits"])
        bundle = build_datasets(
            config.data,
            seed=config.run.seed,
            torch=torch,
            include_splits=materialized_splits,
            include_action_profiles=bool(
                data_access_scope["oracle_action_profiles_materialized"]
            ),
        )
        dataset_manifest_path = output_dir / "dataset_manifest.json"
        write_json_atomic(dataset_manifest_path, bundle.manifest_payload)
        loaders, train_generator = _make_loaders(
            modules,
            bundle,
            config,
            pin_memory=device.type == "cuda",
        )
        model = build_world_model(modules, config.model, config.data).to(device)
        is_fcsrl = (
            config.objective.safety_arm == "fcsrl_feasibility_loss_adaptation"
        )
        fcsrl_windows: ControlledFCSRLWindowDataset | None = None
        feasibility_head: Any | None = None
        target_encoder: Any | None = None
        if is_fcsrl:
            fcsrl_windows = ControlledFCSRLWindowDataset(
                bundle.datasets["train"],
                torch=torch,
            )
            feasibility_head = build_categorical_feasibility_head(
                modules,
                latent_dim=config.model.latent_dim,
                hidden_dim=config.objective.fcsrl_head_hidden_dim,
            ).to(device)
            target_encoder = build_ema_target_encoder(
                modules,
                model,
                config.model,
                config.data,
            ).to(device)
        optimizer_parameters = list(model.parameters())
        if feasibility_head is not None:
            optimizer_parameters.extend(feasibility_head.parameters())
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=config.run.learning_rate,
            weight_decay=config.run.weight_decay,
        )
        amp_enabled = config.run.amp and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        resumed_checkpoint: dict[str, Any] | None = None
        resumed_best: dict[str, Any] | None = None
        if resume_from is not None:
            resumed_checkpoint, resumed_best = _load_resume_pair(
                torch,
                resume_from,
                config,
                dataset_manifest_sha256=bundle.manifest_sha256,
                data_access_scope=data_access_scope,
                orchestration_plan_sha256=orchestration_plan_sha256,
            )
            for label, candidate in (
                ("resume", resumed_checkpoint),
                ("sibling best", resumed_best),
            ):
                candidate_fcsrl = candidate.get("fcsrl_state")
                if is_fcsrl:
                    if not isinstance(candidate_fcsrl, dict):
                        raise ValueError(f"{label} FCSRL checkpoint is missing fcsrl_state")
                    expected_protocol = {
                        "atom_count": FCSRL_ATOM_COUNT,
                        "discount": FCSRL_DISCOUNT,
                        "return_length": FCSRL_RETURN_LENGTH,
                        "unroll_length": FCSRL_UNROLL_LENGTH,
                        "encoder_ema_source_mix": FCSRL_ENCODER_EMA_SOURCE_MIX,
                        "head_hidden_dim": config.objective.fcsrl_head_hidden_dim,
                    }
                    if candidate_fcsrl.get("protocol") != expected_protocol:
                        raise ValueError(f"{label} FCSRL protocol metadata does not match")
                    if not {
                        "head_state",
                        "target_encoder_state",
                    }.issubset(candidate_fcsrl):
                        raise ValueError(f"{label} FCSRL checkpoint state is incomplete")
                elif candidate_fcsrl is not None:
                    raise ValueError(
                        f"{label} ordinary-arm checkpoint unexpectedly contains FCSRL state"
                    )
            model.load_state_dict(resumed_checkpoint["model_state"])
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state"])
            fcsrl_state = resumed_checkpoint.get("fcsrl_state")
            if is_fcsrl:
                if not isinstance(fcsrl_state, dict):
                    raise ValueError("FCSRL resume checkpoint is missing fcsrl_state")
                assert feasibility_head is not None and target_encoder is not None
                feasibility_head.load_state_dict(fcsrl_state["head_state"])
                target_encoder.load_state_dict(fcsrl_state["target_encoder_state"])
                target_encoder.eval()
            elif fcsrl_state is not None:
                raise ValueError("ordinary-arm resume checkpoint unexpectedly contains FCSRL state")
            _restore_training_state(
                torch,
                scaler,
                train_generator,
                resumed_checkpoint["training_state"],
            )
        model_info = model_metadata(model, config.model, config.data)
        fcsrl_artifact_path: Path | None = None
        if is_fcsrl:
            assert (
                fcsrl_windows is not None
                and feasibility_head is not None
                and target_encoder is not None
            )
            fcsrl_artifact_path = output_dir / "fcsrl_regression_fixture.json"
            _write_fcsrl_regression_artifact(
                modules,
                model,
                target_encoder,
                feasibility_head,
                fcsrl_windows,
                device,
                config,
                fcsrl_artifact_path,
                checkpoint_epoch=(
                    int(resumed_checkpoint["epoch"])
                    if resumed_checkpoint is not None
                    else 0
                ),
            )
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
                "data_access_scope": data_access_scope,
                "model": model_info,
                "fcsrl": (
                    {
                        "protocol": "offline_same_backbone_feasibility_loss_adaptation",
                        "head_hidden_dim": config.objective.fcsrl_head_hidden_dim,
                        "head_parameter_count": sum(
                            parameter.numel()
                            for parameter in feasibility_head.parameters()
                        ),
                        "ema_source_mix": FCSRL_ENCODER_EMA_SOURCE_MIX,
                        "return_length": FCSRL_RETURN_LENGTH,
                        "unroll_length": FCSRL_UNROLL_LENGTH,
                        "train_window_count": len(fcsrl_windows),
                        "regression_fixture": str(fcsrl_artifact_path),
                        "scope": "component_integration_not_confirmatory_evidence",
                    }
                    if is_fcsrl
                    else None
                ),
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
        start_epoch = 1
        if resumed_checkpoint is not None and resumed_best is not None:
            start_epoch = int(resumed_checkpoint["epoch"]) + 1
            best_epoch = int(resumed_best["epoch"])
            best_validation = resumed_best["validation"]
            best_metric = float(best_validation["world_model_utility"])
            carried_best = {
                **resumed_best,
                "config": config.to_dict(),
                "resumed_from": {
                    "path": str(resume_from),
                    "sha256": _sha256(resume_from),
                },
            }
            _save_checkpoint_atomic(torch, best_path, carried_best)
        for epoch in range(start_epoch, config.run.epochs + 1):
            if is_fcsrl:
                assert (
                    fcsrl_windows is not None
                    and feasibility_head is not None
                    and target_encoder is not None
                )
                raw_train_metrics = _train_fcsrl_epoch(
                    modules,
                    model,
                    target_encoder,
                    feasibility_head,
                    fcsrl_windows,
                    optimizer,
                    scaler,
                    device,
                    config,
                    epoch=epoch,
                )
            else:
                raw_train_metrics = _train_epoch(
                    modules,
                    model,
                    loaders["train"],
                    optimizer,
                    scaler,
                    device,
                    config,
                )
            train_metrics = _with_world_model_utility(
                raw_train_metrics,
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
                "data_access_scope": data_access_scope,
                "orchestration_plan_sha256": orchestration_plan_sha256,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "training_state": _capture_training_state(
                    torch,
                    scaler,
                    train_generator,
                ),
                "fcsrl_state": (
                    {
                        "head_state": feasibility_head.state_dict(),
                        "target_encoder_state": target_encoder.state_dict(),
                        "protocol": {
                            "atom_count": FCSRL_ATOM_COUNT,
                            "discount": FCSRL_DISCOUNT,
                            "return_length": FCSRL_RETURN_LENGTH,
                            "unroll_length": FCSRL_UNROLL_LENGTH,
                            "encoder_ema_source_mix": FCSRL_ENCODER_EMA_SOURCE_MIX,
                            "head_hidden_dim": config.objective.fcsrl_head_hidden_dim,
                        },
                    }
                    if is_fcsrl
                    else None
                ),
                "validation": validation_metrics,
                "resumed_from": (
                    {
                        "path": str(resume_from),
                        "sha256": _sha256(resume_from),
                    }
                    if resume_from is not None
                    else None
                ),
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
        if is_fcsrl:
            selected_fcsrl_state = selected_checkpoint.get("fcsrl_state")
            if not isinstance(selected_fcsrl_state, dict):
                raise ValueError("selected FCSRL checkpoint is missing fcsrl_state")
            assert feasibility_head is not None and target_encoder is not None
            feasibility_head.load_state_dict(selected_fcsrl_state["head_state"])
            target_encoder.load_state_dict(
                selected_fcsrl_state["target_encoder_state"]
            )
            target_encoder.eval()

        postfit_validation: dict[str, object] | None = None
        postfit_validation_path: Path | None = None
        if access_scope == TRAIN_VALIDATION_ONLY_ACCESS:
            validation_trajectories = tuple(
                trajectory
                for trajectory in bundle.trajectories
                if trajectory.split == "validation"
            )
            postfit_validation, postfit_validation_path = run_validation_postfit_audit(
                modules,
                model,
                bundle.datasets["validation"],
                validation_trajectories,
                device,
                config,
                output_dir=output_dir,
                selected_checkpoint_sha256=_sha256(best_path),
                semantic_arm=config.objective.safety_arm,
                ordinary_validation_metrics=best_validation,
                include_profile_prediction_diagnostics=False,
            )

        split_metrics: dict[str, Any] = {}
        audit_paths: dict[str, str] = (
            {"validation": str(output_dir / VALIDATION_AUDIT_RECORDS)}
            if postfit_validation is not None
            else {}
        )
        for split in data_access_scope["evaluated_splits"]:
            collect = (
                bool(data_access_scope["oracle_action_profiles_materialized"])
                and config.evaluation.emit_audit_records
                and split in {"validation", "calibration", "test"}
            )
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
        rollout_metrics = (
            {"validation": postfit_validation["rollout_metrics"]}
            if postfit_validation is not None
            else {
                split: _evaluate_rollouts(
                    modules,
                    model,
                    bundle.datasets[split],
                    device,
                    config,
                )
                for split in data_access_scope["rollout_splits"]
            }
        )

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
                    "calibration_or_test_materialized": bool(
                        data_access_scope["calibration_access"]
                        or data_access_scope["final_test_access"]
                    ),
                },
                "data_access_scope": data_access_scope,
                "orchestration_plan_sha256": orchestration_plan_sha256,
                "postfit_validation_audit": (
                    {
                        "path": str(postfit_validation_path),
                        "sha256": _sha256(postfit_validation_path),
                        "manifest_sha256": postfit_validation["manifest_sha256"],
                        "used_for_fitting_or_checkpoint_selection": False,
                    }
                    if postfit_validation is not None
                    and postfit_validation_path is not None
                    else None
                ),
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
                    "selected_fcsrl_head_state": (
                        _state_sha256(
                            torch,
                            selected_checkpoint["fcsrl_state"]["head_state"],
                        )
                        if is_fcsrl
                        else None
                    ),
                    "selected_fcsrl_target_encoder_state": (
                        _state_sha256(
                            torch,
                            selected_checkpoint["fcsrl_state"][
                                "target_encoder_state"
                            ],
                        )
                        if is_fcsrl
                        else None
                    ),
                },
                "fcsrl_regression_fixture": (
                    {
                        "path": str(fcsrl_artifact_path),
                        "sha256": _sha256(fcsrl_artifact_path),
                    }
                    if fcsrl_artifact_path is not None
                    else None
                ),
                "parent_checkpoint": (
                    {
                        "path": str(resume_from),
                        "sha256": _sha256(resume_from),
                        "resumed_epoch": int(resumed_checkpoint["epoch"]),
                    }
                    if resume_from is not None and resumed_checkpoint is not None
                    else None
                ),
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
                "data_access_scope": data_access_scope,
                "orchestration_plan_sha256": orchestration_plan_sha256,
                "postfit_validation_audit": (
                    {
                        "path": str(postfit_validation_path),
                        "sha256": _sha256(postfit_validation_path),
                        "manifest_sha256": postfit_validation["manifest_sha256"],
                    }
                    if postfit_validation is not None
                    and postfit_validation_path is not None
                    else None
                ),
                "split_metrics": split_metrics,
                "rollout_metrics": rollout_metrics,
                "audit_records": audit_paths,
                "fcsrl_regression_fixture": (
                    str(fcsrl_artifact_path)
                    if fcsrl_artifact_path is not None
                    else None
                ),
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
                "completed_epoch_range": [start_epoch, config.run.epochs],
                "postfit_validation_audit": (
                    str(postfit_validation_path)
                    if postfit_validation_path is not None
                    else None
                ),
            }
        )
        write_json_atomic(run_manifest_path, run_manifest)
        result = {
            "status": "success",
            "output_dir": str(output_dir),
            "selected_epoch": best_epoch,
            "best_validation_world_model_utility": best_metric,
            "fcsrl_regression_fixture": (
                str(fcsrl_artifact_path)
                if fcsrl_artifact_path is not None
                else None
            ),
        }
        if data_access_scope["final_test_access"]:
            result["test_metrics"] = split_metrics["test"]
        else:
            result["validation_metrics"] = split_metrics["validation"]
            result["postfit_validation_audit"] = str(postfit_validation_path)
        return result
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
