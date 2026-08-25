"""Frozen numerical semantics for the offline FCSRL feasibility-loss adaptation.

This module implements the categorical distribution, target projection, ten-transition recursion,
and termination mask from the pinned protocol.  It does not implement the original online agent or
silently replace this project's existing latent-transition objective.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

FCSRL_ATOM_COUNT = 63
FCSRL_SYMLOG_LOW = -2.0
FCSRL_SYMLOG_HIGH = 4.0
FCSRL_DISCOUNT = 0.9
FCSRL_RETURN_LENGTH = 10
FCSRL_UNROLL_LENGTH = 4
FCSRL_ENCODER_EMA_SOURCE_MIX = 0.01


class FCSRLProtocolError(ValueError):
    """Raised when a feasibility-loss input violates the frozen adaptation."""


@dataclass(frozen=True)
class FCSRLTargetTrace:
    targets: tuple[float, ...]
    mask: tuple[int, ...]
    projected_targets: tuple[tuple[float, ...], ...]
    train_targets: tuple[float, ...]
    train_mask: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def symlog(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise FCSRLProtocolError("symlog input must be finite")
    return math.copysign(math.log1p(abs(converted)), converted)


def symexp(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise FCSRLProtocolError("symexp input must be finite")
    return math.copysign(math.expm1(abs(converted)), converted)


def categorical_atoms() -> tuple[float, ...]:
    spacing = (FCSRL_SYMLOG_HIGH - FCSRL_SYMLOG_LOW) / (FCSRL_ATOM_COUNT - 1)
    return tuple(FCSRL_SYMLOG_LOW + spacing * index for index in range(FCSRL_ATOM_COUNT))


FCSRL_ATOMS = categorical_atoms()


def project_categorical_target(target: float) -> tuple[float, ...]:
    """Linearly project a scalar target onto adjacent atoms in symlog space."""

    encoded = min(FCSRL_SYMLOG_HIGH, max(FCSRL_SYMLOG_LOW, symlog(target)))
    spacing = FCSRL_ATOMS[1] - FCSRL_ATOMS[0]
    position = (encoded - FCSRL_SYMLOG_LOW) / spacing
    lower = max(0, min(FCSRL_ATOM_COUNT - 1, math.floor(position)))
    upper = max(0, min(FCSRL_ATOM_COUNT - 1, math.ceil(position)))
    probabilities = [0.0] * FCSRL_ATOM_COUNT
    if lower == upper:
        probabilities[lower] = 1.0
    else:
        upper_weight = position - lower
        probabilities[lower] = 1.0 - upper_weight
        probabilities[upper] = upper_weight
    return tuple(probabilities)


def categorical_mean(probabilities: Sequence[float]) -> float:
    """Decode the mean atom in symlog coordinates using ``symexp``."""

    converted = tuple(float(value) for value in probabilities)
    if len(converted) != FCSRL_ATOM_COUNT:
        raise FCSRLProtocolError(
            f"categorical distribution must have {FCSRL_ATOM_COUNT} atoms"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise FCSRLProtocolError("categorical probabilities must be finite and non-negative")
    total = sum(converted)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise FCSRLProtocolError("categorical probabilities must sum to one")
    return symexp(sum(probability * atom for probability, atom in zip(converted, FCSRL_ATOMS)))


def projected_cross_entropy(
    logits: Sequence[float], target_probabilities: Sequence[float]
) -> float:
    """Stable scalar cross-entropy used by deterministic regression fixtures."""

    converted_logits = tuple(float(value) for value in logits)
    target = tuple(float(value) for value in target_probabilities)
    if len(converted_logits) != FCSRL_ATOM_COUNT:
        raise FCSRLProtocolError(f"logits must have {FCSRL_ATOM_COUNT} atoms")
    if any(not math.isfinite(value) for value in converted_logits):
        raise FCSRLProtocolError("logits must be finite")
    # Reuse the probability validation without changing the target's semantics.
    categorical_mean(target)
    maximum = max(converted_logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in converted_logits)
    )
    return -sum(
        weight * (logit - log_normalizer)
        for weight, logit in zip(target, converted_logits, strict=True)
    )


def sequence_mask(
    terminations: Sequence[bool], truncations: Sequence[bool]
) -> tuple[int, ...]:
    """Keep the first ending transition and mask every later position."""

    if len(terminations) != FCSRL_RETURN_LENGTH or len(truncations) != FCSRL_RETURN_LENGTH:
        raise FCSRLProtocolError(
            f"termination arrays must have length {FCSRL_RETURN_LENGTH}"
        )
    if any(not isinstance(value, bool) for value in (*terminations, *truncations)):
        raise FCSRLProtocolError("termination and truncation values must be booleans")
    if any(done and timeout for done, timeout in zip(terminations, truncations, strict=True)):
        raise FCSRLProtocolError("a transition cannot be both terminated and truncated")
    active = True
    mask: list[int] = []
    for done, timeout in zip(terminations, truncations, strict=True):
        mask.append(int(active))
        if active and (done or timeout):
            active = False
    return tuple(mask)


def build_target_trace(
    violations: Sequence[int | bool],
    terminations: Sequence[bool],
    truncations: Sequence[bool],
    bootstrap_values: Sequence[float],
    *,
    discount: float = FCSRL_DISCOUNT,
) -> FCSRLTargetTrace:
    """Compute the frozen ten-transition backward targets and first-four train slice."""

    sequences = (violations, terminations, truncations, bootstrap_values)
    if any(len(values) != FCSRL_RETURN_LENGTH for values in sequences):
        raise FCSRLProtocolError(
            f"all target-recursion inputs must have length {FCSRL_RETURN_LENGTH}"
        )
    converted_violations: list[int] = []
    for value in violations:
        if isinstance(value, bool):
            converted_violations.append(int(value))
        elif isinstance(value, int) and value in {0, 1}:
            converted_violations.append(value)
        else:
            raise FCSRLProtocolError("violation labels must be binary")
    converted_bootstrap = tuple(float(value) for value in bootstrap_values)
    if any(not math.isfinite(value) for value in converted_bootstrap):
        raise FCSRLProtocolError("bootstrap values must be finite")
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise FCSRLProtocolError("discount must lie in [0, 1]")
    mask = sequence_mask(terminations, truncations)
    targets = [0.0] * FCSRL_RETURN_LENGTH
    targets[-1] = converted_bootstrap[-1]
    for timestep in range(FCSRL_RETURN_LENGTH - 2, -1, -1):
        continuation = (
            converted_bootstrap[timestep]
            if truncations[timestep]
            else targets[timestep + 1]
        )
        targets[timestep] = max(
            float(converted_violations[timestep]),
            discount * (1.0 - float(terminations[timestep])) * continuation,
        )
    projected = tuple(project_categorical_target(target) for target in targets)
    return FCSRLTargetTrace(
        targets=tuple(targets),
        mask=mask,
        projected_targets=projected,
        train_targets=tuple(targets[:FCSRL_UNROLL_LENGTH]),
        train_mask=tuple(mask[:FCSRL_UNROLL_LENGTH]),
    )
