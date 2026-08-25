"""World-model objectives and preregistered safety-supervision arms."""

from __future__ import annotations

from typing import Any

from latent_safety.learning.config import ObjectiveConfig
from latent_safety.learning.runtime import TorchModules


def _zero(latent: Any) -> Any:
    return latent.sum() * 0.0


def boundary_contrastive_loss(
    modules: TorchModules,
    latent: Any,
    safety_margin: Any,
    config: ObjectiveConfig,
) -> Any:
    """Separate opposite-side boundary pairs while locally aligning like-margin pairs.

    Only near-boundary negative pairs are pushed apart. Positive pairs have the same safety sign
    and similar physical margin, limiting the incentive to erase task-relevant variation globally.
    """

    torch = modules.torch
    # Unit-normalized codes remove the trivial escape in which the encoder multiplies every code
    # by a large scalar.  The operational audit still calibrates its own radius separately.
    metric_latent = modules.functional.normalize(latent, p=2.0, dim=-1, eps=1e-8)
    batch_size = int(latent.shape[0])
    if batch_size < 2:
        return _zero(latent)
    left, right = torch.triu_indices(batch_size, batch_size, offset=1, device=latent.device)
    left_margin = safety_margin[left]
    right_margin = safety_margin[right]
    near_boundary = (left_margin.abs() <= config.boundary_band) & (
        right_margin.abs() <= config.boundary_band
    )
    opposite = near_boundary & (left_margin * right_margin < 0.0)
    same_local = (
        (left_margin * right_margin > 0.0)
        & ((left_margin - right_margin).abs() <= config.positive_margin_band)
    )
    terms: list[Any] = []
    per_class_limit = max(1, config.max_contrastive_pairs // 2)
    negative_indices = torch.nonzero(opposite, as_tuple=False).flatten()[:per_class_limit]
    positive_indices = torch.nonzero(same_local, as_tuple=False).flatten()[:per_class_limit]
    if negative_indices.numel() > 0:
        negative_distances = torch.linalg.vector_norm(
            metric_latent[left[negative_indices]] - metric_latent[right[negative_indices]],
            dim=-1,
        )
        terms.append(
            torch.relu(config.contrastive_margin - negative_distances).square().mean()
        )
    if positive_indices.numel() > 0:
        positive_distances = torch.linalg.vector_norm(
            metric_latent[left[positive_indices]] - metric_latent[right[positive_indices]],
            dim=-1,
        )
        terms.append(positive_distances.square().mean())
    return torch.stack(terms).mean() if terms else _zero(latent)


def compute_losses(
    modules: TorchModules,
    outputs: dict[str, Any],
    target_latent: Any,
    batch: dict[str, Any],
    config: ObjectiveConfig,
) -> dict[str, Any]:
    """Compute named, independently logged loss terms and their weighted total."""

    functional = modules.functional
    current_frame = batch["history"][:, -1]
    reconstruction = functional.mse_loss(outputs["reconstruction"], current_frame)
    transition = functional.mse_loss(outputs["predicted_next_z"], target_latent.detach())
    if outputs["is_variational"]:
        kl = -0.5 * (
            1.0
            + outputs["logvar"]
            - outputs["mean"].square()
            - outputs["logvar"].exp()
        ).sum(dim=-1).mean()
    else:
        kl = _zero(outputs["z"])

    arm = config.safety_arm
    if arm in {"none", "fcsrl_feasibility_loss_adaptation"}:
        # The sequence trainer adds the FCSRL categorical term after computing this unchanged
        # world-model objective.  Keeping it external prevents relabeling the transition MSE.
        safety = _zero(outputs["z"])
    elif arm == "h_prediction":
        safety = functional.smooth_l1_loss(
            outputs["predicted_margin"] / config.margin_scale,
            batch["safety_margin"] / config.margin_scale,
        )
    elif arm == "boundary_contrastive":
        safety = boundary_contrastive_loss(
            modules,
            outputs["z"],
            batch["safety_margin"],
            config,
        )
    elif arm == "safe_action_profile":
        safety = functional.smooth_l1_loss(
            outputs["predicted_action_profile"] / config.margin_scale,
            batch["action_safety_margins"] / config.margin_scale,
        )
    elif arm == "nonprivileged_predicted_action_profile":
        if "action_safety_margins" in batch:
            raise ValueError(
                "nonprivileged predicted-profile training forbids oracle action_safety_margins"
            )
        target = batch.get("predicted_action_profile_target")
        if target is None:
            raise ValueError(
                "nonprivileged predicted-profile training requires "
                "predicted_action_profile_target"
            )
        safety = functional.smooth_l1_loss(
            outputs["predicted_action_profile"] / config.margin_scale,
            target / config.margin_scale,
        )
    else:  # guarded by configuration validation
        raise AssertionError(f"unhandled safety arm: {arm}")
    total = (
        config.reconstruction_weight * reconstruction
        + config.transition_weight * transition
        + config.kl_weight * kl
        + config.safety_weight * safety
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "transition": transition,
        "kl": kl,
        "safety": safety,
    }
