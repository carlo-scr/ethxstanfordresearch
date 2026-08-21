"""Exact finite-state oracles for the static safety-sufficiency theory.

These routines are deliberately combinatorial.  They are useful for theorem unit tests and
synthetic controls where every state in the audited region is enumerated.  They are not estimators
of a continuous population from an i.i.d. sample.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Vector = Sequence[float]
Code = tuple[float, ...]


@dataclass(frozen=True)
class FiniteStaticAudit:
    """Exact static-fiber summary on an enumerated state space."""

    state_count: int
    fiber_count: int
    mixed_fiber_count: int
    collided_safe_count: int
    exact_defect: float
    collision_margins: tuple[float, ...]
    maximal_sound_codes: tuple[Code, ...]


@dataclass(frozen=True)
class CompletenessCheck:
    """Existence and construction check for a sound, margin-complete certificate."""

    margin: float
    exists: bool
    required_codes: tuple[Code, ...]
    conflicting_required_codes: tuple[Code, ...]


@dataclass(frozen=True)
class DataProcessingCheck:
    """Finite verification of exact-defect monotonicity under a proposed coarsening."""

    is_deterministic_postprocessing: bool
    fine_defect: float
    coarse_defect: float
    monotone: bool


def _validated_codes(
    latents: Sequence[Vector], margins: Sequence[float], *, rounding_digits: int | None
) -> tuple[tuple[Code, ...], tuple[float, ...]]:
    if len(latents) != len(margins):
        raise ValueError("latents and margins must have the same length")
    if not latents:
        raise ValueError("at least one state is required")
    dimension = len(latents[0])
    if dimension == 0 or any(len(latent) != dimension for latent in latents):
        raise ValueError("latent codes must be non-empty and share one dimension")

    codes: list[Code] = []
    for latent in latents:
        code = tuple(float(value) for value in latent)
        if any(not math.isfinite(value) for value in code):
            raise ValueError("latent codes must be finite")
        if rounding_digits is not None:
            code = tuple(round(value, rounding_digits) for value in code)
        codes.append(code)

    numeric_margins = tuple(float(margin) for margin in margins)
    if any(not math.isfinite(margin) for margin in numeric_margins):
        raise ValueError("margins must be finite")
    return tuple(codes), numeric_margins


def _fiber_indices(codes: Sequence[Code]) -> dict[Code, list[int]]:
    fibers: dict[Code, list[int]] = {}
    for index, code in enumerate(codes):
        fibers.setdefault(code, []).append(index)
    return fibers


def audit_finite_static_fibers(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    rounding_digits: int | None = None,
) -> FiniteStaticAudit:
    """Enumerate exact mixed fibers and the maximal sound latent certificate.

    A state is safe at ``h >= 0`` and unsafe at ``h < 0``.  A latent code is accepted by the
    maximal sound certificate exactly when every enumerated state in that fiber is safe.  The
    returned defect is ``max({h(x): x safe and its fiber contains an unsafe state} union {0})``.

    ``rounding_digits`` intentionally changes the audited representation by quantizing it.  It must
    not be described as proof of exact collisions for the unrounded encoder.
    """

    codes, numeric_margins = _validated_codes(
        latents, margins, rounding_digits=rounding_digits
    )
    fibers = _fiber_indices(codes)
    collision_margins: list[float] = []
    mixed_fiber_count = 0
    maximal_sound_codes: list[Code] = []

    for code, indices in fibers.items():
        has_unsafe = any(numeric_margins[index] < 0.0 for index in indices)
        safe_margins = [
            numeric_margins[index] for index in indices if numeric_margins[index] >= 0.0
        ]
        if has_unsafe and safe_margins:
            mixed_fiber_count += 1
            collision_margins.extend(safe_margins)
        if not has_unsafe:
            maximal_sound_codes.append(code)

    ordered_collision_margins = tuple(sorted(collision_margins))
    return FiniteStaticAudit(
        state_count=len(codes),
        fiber_count=len(fibers),
        mixed_fiber_count=mixed_fiber_count,
        collided_safe_count=len(collision_margins),
        exact_defect=max(ordered_collision_margins, default=0.0),
        collision_margins=ordered_collision_margins,
        maximal_sound_codes=tuple(sorted(maximal_sound_codes)),
    )


def check_sound_completeness(
    latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    completeness_margin: float,
    rounding_digits: int | None = None,
) -> CompletenessCheck:
    """Check whether any sound certificate can include all states with ``h >= margin``.

    On a finite enumerated space the answer is exact: such a certificate exists iff no required
    code also contains an unsafe state.  This explicitly resolves the endpoint case that a bare
    supremum value cannot distinguish when the defect equals zero.
    """

    margin = float(completeness_margin)
    if not math.isfinite(margin):
        raise ValueError("completeness_margin must be finite")
    codes, numeric_margins = _validated_codes(
        latents, margins, rounding_digits=rounding_digits
    )
    fibers = _fiber_indices(codes)
    required_codes = {
        code for code, value in zip(codes, numeric_margins, strict=True) if value >= margin
    }
    conflicting = {
        code
        for code in required_codes
        if any(numeric_margins[index] < 0.0 for index in fibers[code])
    }
    return CompletenessCheck(
        margin=margin,
        exists=not conflicting,
        required_codes=tuple(sorted(required_codes)),
        conflicting_required_codes=tuple(sorted(conflicting)),
    )


def check_finite_data_processing(
    fine_latents: Sequence[Vector],
    coarse_latents: Sequence[Vector],
    margins: Sequence[float],
    *,
    rounding_digits: int | None = None,
) -> DataProcessingCheck:
    """Check exact-defect monotonicity when coarse codes are a function of fine codes.

    ``is_deterministic_postprocessing`` is false when one fine code maps to multiple coarse codes on
    the enumerated states.  In that case the data-processing premise is absent and ``monotone`` is
    reported as false, even if the two numerical defects happen to be ordered.
    """

    fine_codes, numeric_margins = _validated_codes(
        fine_latents, margins, rounding_digits=rounding_digits
    )
    coarse_codes, _ = _validated_codes(
        coarse_latents, margins, rounding_digits=rounding_digits
    )
    mapping: dict[Code, Code] = {}
    deterministic = True
    for fine, coarse in zip(fine_codes, coarse_codes, strict=True):
        previous = mapping.setdefault(fine, coarse)
        if previous != coarse:
            deterministic = False

    fine_audit = audit_finite_static_fibers(
        fine_codes, numeric_margins, rounding_digits=None
    )
    coarse_audit = audit_finite_static_fibers(
        coarse_codes, numeric_margins, rounding_digits=None
    )
    numerically_monotone = coarse_audit.exact_defect + 1e-12 >= fine_audit.exact_defect
    return DataProcessingCheck(
        is_deterministic_postprocessing=deterministic,
        fine_defect=fine_audit.exact_defect,
        coarse_defect=coarse_audit.exact_defect,
        monotone=deterministic and numerically_monotone,
    )


def enumerate_binary_certificates(codes: Iterable[Code]) -> tuple[frozenset[Code], ...]:
    """Enumerate all latent subsets for tiny theorem checks.

    This helper is exponential and intentionally refuses more than 20 codes.
    """

    unique_codes = tuple(sorted(set(codes)))
    if len(unique_codes) > 20:
        raise ValueError("certificate enumeration is restricted to at most 20 unique codes")
    return tuple(
        frozenset(
            unique_codes[index]
            for index in range(len(unique_codes))
            if mask & (1 << index)
        )
        for mask in range(1 << len(unique_codes))
    )
