"""Dependency-lazy modules and parameter guards for encoder-internal repair.

The builders in this module do not import PyTorch at module import time.  A checkpoint-specific
wrapper is still responsible for inserting the returned intervention *before* its audited
bottleneck and recording that module path in the adapter manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from latent_safety.learning.runtime import TorchModules


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prefixes(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of prefixes, not one string")
    resolved = tuple(values)
    if not resolved or any(not value for value in resolved):
        raise ValueError(f"{name} must contain non-empty prefixes")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    return resolved


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        name == prefix.removesuffix(".")
        or name.startswith(prefix if prefix.endswith(".") else prefix + ".")
        for prefix in prefixes
    )


def _prefixes_overlap(left: str, right: str) -> bool:
    left_root = left.removesuffix(".")
    right_root = right.removesuffix(".")
    return (
        left_root == right_root
        or left_root.startswith(right_root + ".")
        or right_root.startswith(left_root + ".")
    )


def build_residual_bottleneck_intervention(
    modules: TorchModules,
    *,
    feature_dimension: int,
    bottleneck_dimension: int,
) -> Any:
    """Build an identity-initialized residual adapter for internal token features.

    The module accepts ``[..., D]`` tensors.  Its final projection is initialized to zero, so
    insertion does not perturb the frozen checkpoint before optimization.  This component alone
    does not prove pre-bottleneck placement; the model-specific wrapper and manifest must name and
    verify the intercepted internal module.
    """

    feature_dimension = _positive_integer(feature_dimension, name="feature_dimension")
    bottleneck_dimension = _positive_integer(
        bottleneck_dimension,
        name="bottleneck_dimension",
    )
    if bottleneck_dimension >= feature_dimension:
        raise ValueError("bottleneck_dimension must be smaller than feature_dimension")
    nn = modules.nn

    class ResidualBottleneckIntervention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(feature_dimension)
            self.down = nn.Linear(feature_dimension, bottleneck_dimension)
            self.activation = nn.GELU()
            self.up = nn.Linear(bottleneck_dimension, feature_dimension)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, features: Any) -> Any:
            if features.ndim < 2 or int(features.shape[-1]) != feature_dimension:
                raise ValueError(
                    f"intervention expects [...,{feature_dimension}] token features"
                )
            residual = self.up(self.activation(self.down(self.normalization(features))))
            return features + residual

    return ResidualBottleneckIntervention()


def build_action_conditioned_profile_head(
    modules: TorchModules,
    *,
    representation_dimension: int,
    action_count: int,
    action_embedding_dimension: int,
    hidden_dimension: int,
) -> Any:
    """Build a training-only head that evaluates every registered discrete action.

    The output is ``[B,K]``.  Action embeddings make the conditioning explicit rather than
    treating columns of one affine layer as an undocumented action convention.  This module must
    live under the exported checkpoint's declared training-head prefix and is removed before the
    fresh-head evaluation.
    """

    representation_dimension = _positive_integer(
        representation_dimension,
        name="representation_dimension",
    )
    action_count = _positive_integer(action_count, name="action_count")
    action_embedding_dimension = _positive_integer(
        action_embedding_dimension,
        name="action_embedding_dimension",
    )
    hidden_dimension = _positive_integer(hidden_dimension, name="hidden_dimension")
    nn = modules.nn
    torch = modules.torch

    class ActionConditionedProfileHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_embeddings = nn.Embedding(action_count, action_embedding_dimension)
            self.predictor = nn.Sequential(
                nn.Linear(
                    representation_dimension + action_embedding_dimension,
                    hidden_dimension,
                ),
                nn.GELU(),
                nn.Linear(hidden_dimension, 1),
            )

        def forward(self, representations: Any) -> Any:
            if representations.ndim != 2 or int(representations.shape[-1]) != (
                representation_dimension
            ):
                raise ValueError(
                    "profile head expects [B,representation_dimension] features"
                )
            actions = torch.arange(action_count, device=representations.device)
            embeddings = self.action_embeddings(actions)
            batch_size = int(representations.shape[0])
            tiled_features = representations.unsqueeze(1).expand(
                batch_size,
                action_count,
                representation_dimension,
            )
            tiled_actions = embeddings.unsqueeze(0).expand(
                batch_size,
                action_count,
                action_embedding_dimension,
            )
            return self.predictor(torch.cat((tiled_features, tiled_actions), dim=-1)).squeeze(-1)

    return ActionConditionedProfileHead()


def configure_trainable_parameter_policy(
    model: Any,
    *,
    trainable_prefixes: Sequence[str],
) -> tuple[str, ...]:
    """Freeze every parameter except explicitly named intervention/head subtrees."""

    prefixes = _prefixes(trainable_prefixes, name="trainable_prefixes")
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = _matches_prefix(name, prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise ValueError("no model parameter matches the declared trainable prefixes")
    return tuple(selected)


def validate_trainable_parameter_policy(
    model: Any,
    *,
    trainable_prefixes: Sequence[str],
) -> tuple[str, ...]:
    """Fail if the observed trainable set differs from the declared prefix policy."""

    prefixes = _prefixes(trainable_prefixes, name="trainable_prefixes")
    observed = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not observed:
        raise ValueError("model has no trainable parameters")
    unexpected = tuple(name for name in observed if not _matches_prefix(name, prefixes))
    if unexpected:
        raise ValueError("undeclared trainable parameters: " + ", ".join(unexpected))
    missing_prefixes = tuple(
        prefix
        for prefix in prefixes
        if not any(_matches_prefix(name, (prefix,)) for name in observed)
    )
    if missing_prefixes:
        raise ValueError("trainable prefixes matched no parameters: " + ", ".join(missing_prefixes))
    return observed


def snapshot_frozen_parameters(
    modules: TorchModules,
    model: Any,
    *,
    mutable_prefixes: Sequence[str],
) -> dict[str, Any]:
    """Clone non-mutable parameters for a post-step mutation audit."""

    del modules
    prefixes = _prefixes(mutable_prefixes, name="mutable_prefixes")
    snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not _matches_prefix(name, prefixes)
    }
    if not snapshot:
        raise ValueError("mutation audit has no frozen parameters to monitor")
    return snapshot


def assert_frozen_parameters_unchanged(
    modules: TorchModules,
    model: Any,
    snapshot: Mapping[str, Any],
) -> None:
    """Verify bitwise equality of every snapshotted non-repair parameter."""

    torch = modules.torch
    current = dict(model.named_parameters())
    missing = tuple(sorted(set(snapshot) - set(current)))
    if missing:
        raise ValueError("snapshotted parameters are missing from model: " + ", ".join(missing))
    changed = []
    for name, expected in snapshot.items():
        observed = current[name].detach().cpu()
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            changed.append(name)
            continue
        if not bool(torch.equal(observed, expected)):
            changed.append(name)
    if changed:
        raise ValueError("frozen parameters changed: " + ", ".join(changed))


def validate_gradient_route(
    modules: TorchModules,
    model: Any,
    *,
    required_gradient_prefixes: Sequence[str],
    forbidden_gradient_prefixes: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Audit one backward pass for required and forbidden parameter-gradient routes."""

    torch = modules.torch
    required = _prefixes(required_gradient_prefixes, name="required_gradient_prefixes")
    forbidden = _prefixes(forbidden_gradient_prefixes, name="forbidden_gradient_prefixes")
    if any(_prefixes_overlap(left, right) for left in required for right in forbidden):
        raise ValueError("required and forbidden gradient prefixes must not overlap")
    with_gradient = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        and bool((parameter.grad != 0).any().item())
    )
    missing = tuple(
        prefix
        for prefix in required
        if not any(_matches_prefix(name, (prefix,)) for name in with_gradient)
    )
    forbidden_hits = tuple(
        name for name in with_gradient if _matches_prefix(name, forbidden)
    )
    if missing:
        raise ValueError("required gradient routes were inactive: " + ", ".join(missing))
    if forbidden_hits:
        raise ValueError("forbidden parameters received gradients: " + ", ".join(forbidden_hits))
    return {
        "parameters_with_nonzero_finite_gradient": with_gradient,
        "required_prefixes": required,
        "forbidden_prefixes": forbidden,
    }
