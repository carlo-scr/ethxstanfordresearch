"""Common-Action Geometry Repair losses and head-removal helpers.

The pure-Python reference objective is executable in the base environment.  Differentiable
functions accept the repository's lazily imported ``TorchModules`` object and therefore do not
import PyTorch when this module is imported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from latent_safety.learning.runtime import TorchModules

Vector = Sequence[float]


@dataclass(frozen=True)
class CommonActionRepairConfig:
    """Named weights and geometry needed by the first repair implementation."""

    gamma: float
    audit_radius: float
    train_radius: float
    softmin_temperature: float
    margin_scale: float = 1.0
    boundary_band: float = 0.25
    boundary_boost: float = 1.0
    profile_tail_fraction: float = 0.1
    profile_tail_weight: float = 1.0
    profile_weight: float = 1.0
    common_action_weight: float = 1.0
    feature_retention_weight: float = 1.0
    geometry_retention_weight: float = 1.0

    def validate(self) -> None:
        finite_fields = {
            "gamma": self.gamma,
            "audit_radius": self.audit_radius,
            "train_radius": self.train_radius,
            "softmin_temperature": self.softmin_temperature,
            "margin_scale": self.margin_scale,
            "boundary_band": self.boundary_band,
            "boundary_boost": self.boundary_boost,
            "profile_tail_fraction": self.profile_tail_fraction,
            "profile_tail_weight": self.profile_tail_weight,
            "profile_weight": self.profile_weight,
            "common_action_weight": self.common_action_weight,
            "feature_retention_weight": self.feature_retention_weight,
            "geometry_retention_weight": self.geometry_retention_weight,
        }
        if any(not math.isfinite(value) for value in finite_fields.values()):
            raise ValueError("repair configuration values must be finite")
        if self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative")
        if self.audit_radius < 0.0:
            raise ValueError("audit_radius must be nonnegative")
        if self.train_radius <= self.audit_radius:
            raise ValueError("train_radius must be strictly greater than audit_radius")
        if self.softmin_temperature <= 0.0:
            raise ValueError("softmin_temperature must be positive")
        if self.margin_scale <= 0.0 or self.boundary_band <= 0.0:
            raise ValueError("margin_scale and boundary_band must be positive")
        if not 0.0 < self.profile_tail_fraction <= 1.0:
            raise ValueError("profile_tail_fraction must lie in (0, 1]")
        nonnegative = (
            self.boundary_boost,
            self.profile_tail_weight,
            self.profile_weight,
            self.common_action_weight,
            self.feature_retention_weight,
            self.geometry_retention_weight,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("repair weights must be nonnegative")


@dataclass(frozen=True)
class ReferenceCommonActionLoss:
    """Dependency-free hard set-wise objective for regression tests and audit parity."""

    loss: float
    viable_center_count: int
    eligible_center_count: int
    eligible_center_coverage: float | None
    action_count: int
    active_nonself_counts: tuple[int, ...]
    per_center_costs: tuple[tuple[float, ...] | None, ...]
    per_center_minimum: tuple[float | None, ...]
    chosen_actions: tuple[int | None, ...]


@dataclass(frozen=True)
class TorchCommonActionLoss:
    """Differentiable surrogate plus the hard diagnostic with the theorem's zero set."""

    loss: Any
    hard_loss: Any
    costs: Any
    hard_per_center: Any
    chosen_actions: Any
    viable_mask: Any
    active_member_counts: Any
    active_nonself_counts: Any
    eligible_center_mask: Any
    eligible_center_coverage: float | None


@dataclass(frozen=True)
class ProfileLossOutput:
    loss: Any
    weighted_mean: Any
    tail_mean: Any
    mean_boundary_weight: Any


@dataclass(frozen=True)
class RepairLosses:
    total: Any
    native: Any
    profile: Any
    common_action: Any
    common_action_hard: Any
    common_action_active_nonself_counts: Any
    common_action_eligible_center_coverage: float | None
    retention_feature: Any
    retention_geometry: Any


def _validate_reference_inputs(
    latents: Sequence[Vector],
    profiles: Sequence[Sequence[float]],
    *,
    gamma: float,
    train_radius: float,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be nonnegative and finite")
    if not math.isfinite(train_radius) or train_radius <= 0.0:
        raise ValueError("train_radius must be positive and finite")
    if len(latents) != len(profiles) or not latents:
        raise ValueError("latents and profiles must have the same positive length")
    vectors = tuple(tuple(float(value) for value in row) for row in latents)
    dimension = len(vectors[0])
    if dimension < 1 or any(len(row) != dimension for row in vectors):
        raise ValueError("latents must be non-empty and share one dimension")
    rows = tuple(tuple(float(value) for value in row) for row in profiles)
    action_count = len(rows[0])
    if action_count < 1 or any(len(row) != action_count for row in rows):
        raise ValueError("profiles must be non-empty and share one action dimension")
    if any(not math.isfinite(value) for row in vectors + rows for value in row):
        raise ValueError("latents and profiles must be finite")
    return vectors, rows


def common_action_geometry_reference_loss(
    latents: Sequence[Vector],
    action_safety_margins: Sequence[Sequence[float]],
    *,
    gamma: float,
    train_radius: float,
) -> ReferenceCommonActionLoss:
    """Compute the hard finite-reference Common-Action Geometry objective.

    Targets are treated as fixed physical margins.  Members and centers are restricted to
    ``I_gamma`` before distances are used.  The compact-support hinge is zero at exactly
    ``train_radius``, so the corresponding zero-loss statement concerns strict neighborhoods.
    """

    vectors, profiles = _validate_reference_inputs(
        latents,
        action_safety_margins,
        gamma=gamma,
        train_radius=train_radius,
    )
    viable = tuple(index for index, row in enumerate(profiles) if max(row) >= gamma)
    action_count = len(profiles[0])
    all_costs: list[tuple[float, ...] | None] = [None] * len(vectors)
    minima: list[float | None] = [None] * len(vectors)
    actions: list[int | None] = [None] * len(vectors)
    active_nonself_counts = [0] * len(vectors)
    for center in viable:
        weighted_members = []
        for member in viable:
            distance = math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(vectors[center], vectors[member], strict=True)
                )
            )
            radius_weight = max(0.0, train_radius - distance) ** 2
            if radius_weight > 0.0:
                weighted_members.append((member, radius_weight))
        active_nonself_counts[center] = sum(
            member != center for member, _ in weighted_members
        )
        costs = []
        for action in range(action_count):
            cost = sum(
                max(0.0, gamma - profiles[member][action]) * radius_weight
                for member, radius_weight in weighted_members
            )
            costs.append(cost)
        chosen = min(range(action_count), key=lambda action: (costs[action], action))
        all_costs[center] = tuple(costs)
        minima[center] = costs[chosen]
        actions[center] = chosen
    viable_minima = [value for value in minima if value is not None]
    eligible_count = sum(active_nonself_counts[index] > 0 for index in viable)
    return ReferenceCommonActionLoss(
        loss=(sum(viable_minima) / len(viable_minima) if viable_minima else 0.0),
        viable_center_count=len(viable),
        eligible_center_count=eligible_count,
        eligible_center_coverage=(eligible_count / len(viable) if viable else None),
        action_count=action_count,
        active_nonself_counts=tuple(active_nonself_counts),
        per_center_costs=tuple(all_costs),
        per_center_minimum=tuple(minima),
        chosen_actions=tuple(actions),
    )


def common_action_geometry_loss(
    modules: TorchModules,
    metric_latents: Any,
    action_safety_margins: Any,
    *,
    gamma: float,
    train_radius: float,
    temperature: float,
    comparison_mask: Any | None = None,
) -> TorchCommonActionLoss:
    """Return the differentiable normalized-softmin repair loss.

    ``metric_latents`` must already use the calibration-frozen normalization declared by the run.
    The function never fits batch statistics.  Physical profile targets are detached, so the
    encoder cannot reduce this term by changing the labels.  A normalized log-mean-exp soft
    minimum keeps this nonnegative; the hard minimum is logged separately because only it has the
    finite-reference zero-loss semantics stated in the paper.  If supplied, ``comparison_mask``
    must be a symmetric boolean relation containing every viable self-edge; diagnostics and any
    zero-loss interpretation are then scoped to that declared relation, not the full batch.
    """

    torch = modules.torch
    functional = modules.functional
    if metric_latents.ndim != 2 or action_safety_margins.ndim != 2:
        raise ValueError("metric_latents and action_safety_margins must be rank-two tensors")
    if metric_latents.shape[0] != action_safety_margins.shape[0]:
        raise ValueError("metric_latents and action profiles must share the batch dimension")
    if metric_latents.shape[0] < 1 or metric_latents.shape[1] < 1:
        raise ValueError("metric_latents must have positive dimensions")
    if action_safety_margins.shape[1] < 1:
        raise ValueError("action profiles must contain at least one action")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be nonnegative and finite")
    if not math.isfinite(train_radius) or train_radius <= 0.0:
        raise ValueError("train_radius must be positive and finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    if not bool(torch.isfinite(metric_latents).all().item()):
        raise ValueError("metric_latents must be finite")
    if not bool(torch.isfinite(action_safety_margins).all().item()):
        raise ValueError("action_safety_margins must be finite")
    if not bool(metric_latents.is_floating_point()) or not bool(
        action_safety_margins.is_floating_point()
    ):
        raise ValueError("metric_latents and action profiles must be floating-point tensors")
    if metric_latents.device != action_safety_margins.device:
        raise ValueError("metric_latents and action profiles must share one device")

    targets = action_safety_margins.detach().to(dtype=metric_latents.dtype)
    viable = targets.amax(dim=-1) >= gamma
    distances = torch.cdist(metric_latents, metric_latents, p=2.0)
    active = (distances < train_radius) & viable.unsqueeze(0) & viable.unsqueeze(1)
    if comparison_mask is not None:
        if comparison_mask.shape != distances.shape:
            raise ValueError("comparison_mask must have shape [B,B]")
        if comparison_mask.dtype != torch.bool:
            raise ValueError("comparison_mask must have boolean dtype")
        if comparison_mask.device != metric_latents.device:
            raise ValueError("comparison_mask must be on the metric_latents device")
        if not torch.equal(comparison_mask, comparison_mask.transpose(0, 1)):
            raise ValueError("comparison_mask must be symmetric")
        if bool(viable.any().item()) and not bool(
            torch.diagonal(comparison_mask)[viable].all().item()
        ):
            raise ValueError("comparison_mask diagonal must include every viable center")
        active = active & comparison_mask
    weights = functional.relu(train_radius - distances).square()
    weights = weights * active.to(dtype=metric_latents.dtype)
    active_member_counts = active.sum(dim=-1)
    active_nonself_counts = active_member_counts - viable.to(
        dtype=active_member_counts.dtype
    )
    eligible_center_mask = viable & (active_nonself_counts > 0)
    deficits = functional.relu(gamma - targets)
    costs = weights @ deficits
    raw_hard_per_center, raw_chosen_actions = costs.min(dim=-1)
    hard_per_center = torch.where(
        viable,
        raw_hard_per_center,
        torch.zeros_like(raw_hard_per_center),
    )
    chosen_actions = torch.where(
        viable,
        raw_chosen_actions,
        torch.full_like(raw_chosen_actions, -1),
    )
    viable_count = int(viable.sum().item())
    eligible_count = int(eligible_center_mask.sum().item())
    eligible_coverage = eligible_count / viable_count if viable_count else None
    if viable_count == 0:
        zero = metric_latents.sum() * 0.0
        return TorchCommonActionLoss(
            loss=zero,
            hard_loss=zero,
            costs=costs,
            hard_per_center=hard_per_center,
            chosen_actions=chosen_actions,
            viable_mask=viable,
            active_member_counts=active_member_counts,
            active_nonself_counts=active_nonself_counts,
            eligible_center_mask=eligible_center_mask,
            eligible_center_coverage=eligible_coverage,
        )
    viable_costs = costs[viable]
    normalized_softmin = -temperature * (
        torch.logsumexp(-viable_costs / temperature, dim=-1)
        - math.log(int(costs.shape[1]))
    )
    return TorchCommonActionLoss(
        loss=normalized_softmin.mean(),
        hard_loss=hard_per_center[viable].mean(),
        costs=costs,
        hard_per_center=hard_per_center,
        chosen_actions=chosen_actions,
        viable_mask=viable,
        active_member_counts=active_member_counts,
        active_nonself_counts=active_nonself_counts,
        eligible_center_mask=eligible_center_mask,
        eligible_center_coverage=eligible_coverage,
    )


def boundary_tail_profile_loss(
    modules: TorchModules,
    predicted_profiles: Any,
    target_profiles: Any,
    *,
    gamma: float,
    margin_scale: float,
    boundary_band: float,
    boundary_boost: float,
    tail_fraction: float,
    tail_weight: float,
) -> ProfileLossOutput:
    """Boundary-weighted SmoothL1 plus a top-tail per-state error term."""

    torch = modules.torch
    functional = modules.functional
    if predicted_profiles.shape != target_profiles.shape or predicted_profiles.ndim != 2:
        raise ValueError("predicted and target profiles must share shape [B,K]")
    if predicted_profiles.shape[0] < 1 or predicted_profiles.shape[1] < 1:
        raise ValueError("profile tensors must have positive dimensions")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be nonnegative and finite")
    if not bool(torch.isfinite(predicted_profiles).all().item()) or not bool(
        torch.isfinite(target_profiles).all().item()
    ):
        raise ValueError("predicted and target profiles must be finite")
    if not bool(predicted_profiles.is_floating_point()) or not bool(
        target_profiles.is_floating_point()
    ):
        raise ValueError("predicted and target profiles must be floating-point tensors")
    if (
        predicted_profiles.device != target_profiles.device
        or predicted_profiles.dtype != target_profiles.dtype
    ):
        raise ValueError("predicted and target profiles must share dtype and device")
    if margin_scale <= 0.0 or boundary_band <= 0.0:
        raise ValueError("margin_scale and boundary_band must be positive")
    if boundary_boost < 0.0 or tail_weight < 0.0:
        raise ValueError("boundary_boost and tail_weight must be nonnegative")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    targets = target_profiles.detach()
    component = functional.smooth_l1_loss(
        predicted_profiles / margin_scale,
        targets / margin_scale,
        reduction="none",
    )
    boundary_weight = 1.0 + boundary_boost * torch.exp(
        -(targets - gamma).abs() / boundary_band
    )
    per_state = (component * boundary_weight).mean(dim=-1)
    tail_count = max(1, math.ceil(int(per_state.shape[0]) * tail_fraction))
    tail_mean = torch.topk(per_state, k=tail_count, largest=True).values.mean()
    weighted_mean = per_state.mean()
    return ProfileLossOutput(
        loss=weighted_mean + tail_weight * tail_mean,
        weighted_mean=weighted_mean,
        tail_mean=tail_mean,
        mean_boundary_weight=boundary_weight.mean(),
    )


def feature_retention_loss(modules: TorchModules, repaired: Any, frozen: Any) -> Any:
    """Anchor repaired features to a detached frozen-teacher feature tensor."""

    if repaired.shape != frozen.shape:
        raise ValueError("repaired and frozen features must have identical shapes")
    torch = modules.torch
    if not bool(repaired.is_floating_point()) or not bool(frozen.is_floating_point()):
        raise ValueError("retention features must be floating-point tensors")
    if repaired.device != frozen.device or repaired.dtype != frozen.dtype:
        raise ValueError("retention features must share dtype and device")
    if not bool(torch.isfinite(repaired).all().item()) or not bool(
        torch.isfinite(frozen).all().item()
    ):
        raise ValueError("retention features must be finite")
    return modules.functional.smooth_l1_loss(repaired, frozen.detach())


def geometry_retention_loss(modules: TorchModules, repaired: Any, frozen: Any) -> Any:
    """Retain upper-triangle pairwise geometry without diagonal self-distances."""

    torch = modules.torch
    if repaired.shape != frozen.shape or repaired.ndim != 2:
        raise ValueError("repaired and frozen decision features must share shape [B,D]")
    if not bool(repaired.is_floating_point()) or not bool(frozen.is_floating_point()):
        raise ValueError("geometry features must be floating-point tensors")
    if repaired.device != frozen.device or repaired.dtype != frozen.dtype:
        raise ValueError("geometry features must share dtype and device")
    if not bool(torch.isfinite(repaired).all().item()) or not bool(
        torch.isfinite(frozen).all().item()
    ):
        raise ValueError("geometry features must be finite")
    if repaired.shape[0] < 2:
        return repaired.sum() * 0.0
    repaired_distances = torch.cdist(repaired, repaired, p=2.0)
    frozen_distances = torch.cdist(frozen.detach(), frozen.detach(), p=2.0)
    left, right = torch.triu_indices(
        int(repaired.shape[0]),
        int(repaired.shape[0]),
        offset=1,
        device=repaired.device,
    )
    return modules.functional.smooth_l1_loss(
        repaired_distances[left, right],
        frozen_distances[left, right],
    )


def compute_repair_losses(
    modules: TorchModules,
    *,
    native_loss: Any,
    repaired_metric_latents: Any,
    frozen_metric_latents: Any,
    predicted_profiles: Any,
    target_profiles: Any,
    config: CommonActionRepairConfig,
    comparison_mask: Any | None = None,
) -> RepairLosses:
    """Compose family-native, profile, set-wise, and retention terms without conflating them."""

    config.validate()
    torch = modules.torch
    if not torch.is_tensor(native_loss) or native_loss.ndim != 0:
        raise ValueError("native_loss must be a scalar tensor")
    if not bool(torch.isfinite(native_loss).item()):
        raise ValueError("native_loss must be finite")
    profile = boundary_tail_profile_loss(
        modules,
        predicted_profiles,
        target_profiles,
        gamma=config.gamma,
        margin_scale=config.margin_scale,
        boundary_band=config.boundary_band,
        boundary_boost=config.boundary_boost,
        tail_fraction=config.profile_tail_fraction,
        tail_weight=config.profile_tail_weight,
    )
    common_action = common_action_geometry_loss(
        modules,
        repaired_metric_latents,
        target_profiles,
        gamma=config.gamma,
        train_radius=config.train_radius,
        temperature=config.softmin_temperature,
        comparison_mask=comparison_mask,
    )
    feature = feature_retention_loss(
        modules,
        repaired_metric_latents,
        frozen_metric_latents,
    )
    geometry = geometry_retention_loss(
        modules,
        repaired_metric_latents,
        frozen_metric_latents,
    )
    total = (
        native_loss
        + config.profile_weight * profile.loss
        + config.common_action_weight * common_action.loss
        + config.feature_retention_weight * feature
        + config.geometry_retention_weight * geometry
    )
    return RepairLosses(
        total=total,
        native=native_loss,
        profile=profile.loss,
        common_action=common_action.loss,
        common_action_hard=common_action.hard_loss,
        common_action_active_nonself_counts=common_action.active_nonself_counts,
        common_action_eligible_center_coverage=common_action.eligible_center_coverage,
        retention_feature=feature,
        retention_geometry=geometry,
    )


def encoder_only_state_dict(
    state_dict: Mapping[str, Any],
    *,
    training_head_prefix: str = "profile_head.",
    retained_prefixes: tuple[str, ...] = ("encoder.", "repair_adapter."),
) -> dict[str, Any]:
    """Return a whitelist-filtered encoder state and reject undeclared side modules.

    The explicit failure prevents an evaluation artifact from being mislabeled as head-removed
    when its producer used a different module name.  The returned mapping preserves values without
    copying tensor storage; the checkpoint writer owns serialization and hashing.
    """

    if not training_head_prefix:
        raise ValueError("training_head_prefix must be non-empty")
    if (
        isinstance(retained_prefixes, str)
        or not retained_prefixes
        or any(not prefix for prefix in retained_prefixes)
        or len(set(retained_prefixes)) != len(retained_prefixes)
    ):
        raise ValueError("retained_prefixes must contain unique non-empty prefixes")
    if any(
        training_head_prefix.startswith(prefix) or prefix.startswith(training_head_prefix)
        for prefix in retained_prefixes
    ):
        raise ValueError("training-head and retained prefixes must not overlap")
    if any(not isinstance(key, str) or not key for key in state_dict):
        raise ValueError("state_dict keys must be non-empty strings")
    removed = tuple(key for key in state_dict if key.startswith(training_head_prefix))
    if not removed:
        raise ValueError("state_dict contains no keys under the declared training head prefix")
    unexpected = tuple(
        key
        for key in state_dict
        if not key.startswith(training_head_prefix)
        and not any(key.startswith(prefix) for prefix in retained_prefixes)
    )
    if unexpected:
        raise ValueError(
            "state_dict contains undeclared non-encoder keys: " + ", ".join(unexpected)
        )
    encoder = {
        key: value
        for key, value in state_dict.items()
        if any(key.startswith(prefix) for prefix in retained_prefixes)
    }
    if not encoder:
        raise ValueError("head removal produced an empty encoder state")
    return encoder
