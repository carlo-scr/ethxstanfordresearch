"""Dependency-light contracts for pretrained encoders and branchable physical domains."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _shape(value: Any, *, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{field_name} must expose a tensor-like shape")
    try:
        normalized = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} has an invalid shape") from error
    if any(dimension < 0 for dimension in normalized):
        raise ValueError(f"{field_name} shape must be nonnegative")
    return normalized


def _sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _git_revision(value: str, *, field_name: str) -> None:
    if len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a full lowercase commit digest")


def _dtype_text(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype).lower()


def _require_dtype(value: Any, *, field_name: str, kind: str) -> None:
    dtype = _dtype_text(value)
    if dtype is None:
        return
    markers = {
        "bool": ("bool",),
        "float": ("float", "double", "half", "bfloat"),
        "integer": ("int", "long", "short", "byte", "uint"),
    }[kind]
    if not any(marker in dtype for marker in markers):
        raise ValueError(f"{field_name} must use a {kind} dtype when dtype is exposed")


def _inspected_values(value: Any) -> list[Any] | None:
    """Return flattened Python values when a tensor-like object explicitly supports inspection."""

    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        return None
    try:
        nested = tolist()
    except (RuntimeError, TypeError, ValueError):
        return None
    flattened: list[Any] = []
    stack = [nested]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
        else:
            flattened.append(current)
    return flattened


def _require_finite(values: list[Any], *, field_name: str) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} values must be numeric") from error
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field_name} values must be finite")
    return normalized


@dataclass(frozen=True)
class VisualBatch:
    """Canonical application-side visual input.

    ``frames`` uses ``[B,T,C,H,W]`` float values in ``[0,1]``.  Every family adapter owns its
    permutation and normalization.  ``valid_frames`` uses ``[B,T]`` so padding cannot silently
    change a representation's causal information span. ``validate()`` always checks structure and
    any exposed dtypes. Set ``inspect_values=True`` to scan pixels and masks; concrete loaders must
    do this (or provide equivalent observed validation) before production extraction. The default
    does not pretend to inspect opaque tensor values.
    """

    frames: Any
    valid_frames: Any
    proprioception: Any | None = None

    def validate(self, *, inspect_values: bool = False) -> None:
        frames = _shape(self.frames, field_name="frames")
        mask = _shape(self.valid_frames, field_name="valid_frames")
        if len(frames) != 5 or frames[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if frames[0] < 1 or frames[1] < 1 or frames[3] < 1 or frames[4] < 1:
            raise ValueError("frames must have positive batch, time, height, and width")
        if mask != frames[:2]:
            raise ValueError("valid_frames must have shape [B,T]")
        _require_dtype(self.frames, field_name="frames", kind="float")
        _require_dtype(self.valid_frames, field_name="valid_frames", kind="bool")
        if self.proprioception is not None:
            proprioception = _shape(self.proprioception, field_name="proprioception")
            if len(proprioception) != 3 or proprioception[:2] != frames[:2]:
                raise ValueError("proprioception must have shape [B,T,P]")
            _require_dtype(self.proprioception, field_name="proprioception", kind="float")
        if inspect_values:
            frame_values = _inspected_values(self.frames)
            mask_values = _inspected_values(self.valid_frames)
            if frame_values is None or mask_values is None:
                raise ValueError("strict visual value validation requires inspectable tensors")
            normalized_frames = _require_finite(frame_values, field_name="frames")
            if any(value < 0.0 or value > 1.0 for value in normalized_frames):
                raise ValueError("frames values must lie in [0,1]")
            if len(mask_values) != frames[0] * frames[1] or any(
                value not in (False, True, 0, 1) for value in mask_values
            ):
                raise ValueError("valid_frames values must be boolean")
            for batch_index in range(frames[0]):
                start = batch_index * frames[1]
                row = tuple(bool(value) for value in mask_values[start : start + frames[1]])
                if not any(row):
                    raise ValueError("every sample must contain at least one valid frame")
                first_invalid = next((index for index, value in enumerate(row) if not value), None)
                if first_invalid is not None and any(row[first_invalid:]):
                    raise ValueError("valid_frames must be a true-prefix mask")


@dataclass(frozen=True)
class AdapterCapabilities:
    """Executable representation capabilities, not general upstream-model claims.

    ``accepts_video_batches`` describes the complete adapter, whereas
    ``native_temporal_mode`` describes the upstream encoder.  This distinction lets a framewise
    image encoder accept canonical video batches without pretending to be a native video model.
    Reconstruction and prediction are intentionally absent: those operations need separate,
    executable decoder/predictor protocols rather than opaque booleans.
    """

    accepts_video_batches: bool
    native_temporal_mode: str
    representation_kind: str
    repair_scope: str
    returns_intermediates: bool

    def validate(self) -> None:
        if self.native_temporal_mode not in {
            "causal_video",
            "bidirectional_video",
            "framewise_image",
        }:
            raise ValueError("native_temporal_mode is not registered")
        if self.representation_kind not in {
            "continuous_dense",
            "discrete_ids_with_continuous_codes",
        }:
            raise ValueError("representation_kind is not registered")
        if self.repair_scope not in {
            "none",
            "post_representation_control",
            "encoder_internal",
        }:
            raise ValueError("repair_scope is not registered")


@dataclass(frozen=True)
class ArtifactProvenance:
    """Declared identity of one local artifact used by a family adapter.

    A digest in this structure is a declaration.  Whether bytes were checked before loading is
    recorded separately in :class:`LoaderDeclaration` and is never inferred from this string.
    """

    role: str
    artifact_id: str
    sha256: str
    size_bytes: int | None = None

    def validate(self) -> None:
        if not self.role.strip() or not self.artifact_id.strip():
            raise ValueError("artifact role and identifier must be non-empty")
        _sha256(self.sha256, field_name=f"artifact[{self.role}].sha256")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be nonnegative when provided")


@dataclass(frozen=True)
class LoaderDeclaration:
    """Caller-declared loader behavior that this dependency-light scaffold cannot observe.

    These fields make provenance claims reviewable without implying that a callable wrapper has
    sandboxed arbitrary functions, inspected network traffic, or verified a checkout itself.
    """

    artifact_source: str
    network_access: str
    artifact_hash_check: str
    source_revision_check: str
    upstream_code_execution: str
    hub_remote_code_trust: str

    def validate(self) -> None:
        choices = {
            "artifact_source": {"local_files", "url_or_hub", "unknown"},
            "network_access": {"disabled", "enabled", "unknown"},
            "artifact_hash_check": {
                "before_load",
                "after_load",
                "not_checked",
                "unknown",
            },
            "source_revision_check": {
                "full_commit_checked_out",
                "declared_only",
                "not_checked",
                "unknown",
            },
            "upstream_code_execution": {
                "none",
                "installed_package",
                "pinned_local_repository",
                "floating_or_remote_repository",
                "unknown",
            },
            "hub_remote_code_trust": {"disabled", "enabled", "not_applicable", "unknown"},
        }
        for field_name, allowed in choices.items():
            if getattr(self, field_name) not in allowed:
                raise ValueError(f"{field_name} is not a registered loader declaration")


@dataclass(frozen=True)
class AdapterProvenance:
    """Declared representation identity recorded with every extracted feature artifact."""

    contract_key: str
    checkpoint_id: str
    artifacts: tuple[ArtifactProvenance, ...]
    source_repository: str
    source_revision: str
    preprocessing_id: str
    preprocessing_sha256: str
    feature_view: str
    pooling_id: str
    pooling_sha256: str
    normalization_id: str
    normalization_sha256: str
    intervention_point: str
    license_id: str

    def validate(self) -> None:
        required = (
            self.contract_key,
            self.checkpoint_id,
            self.source_repository,
            self.source_revision,
            self.preprocessing_id,
            self.feature_view,
            self.pooling_id,
            self.normalization_id,
            self.intervention_point,
            self.license_id,
        )
        if any(not value.strip() for value in required):
            raise ValueError("adapter provenance strings must be non-empty")
        _git_revision(self.source_revision, field_name="source_revision")
        _sha256(self.preprocessing_sha256, field_name="preprocessing_sha256")
        _sha256(self.pooling_sha256, field_name="pooling_sha256")
        _sha256(self.normalization_sha256, field_name="normalization_sha256")
        if not self.artifacts:
            raise ValueError("adapter provenance must declare at least one artifact")
        roles = tuple(artifact.role for artifact in self.artifacts)
        if len(set(roles)) != len(roles):
            raise ValueError("artifact roles must be unique")
        for artifact in self.artifacts:
            artifact.validate()


@dataclass(frozen=True)
class RepairParameterGroup:
    """Named trainable parameters tied to a declared intervention point."""

    name: str
    intervention_point: str
    parameters: tuple[Any, ...]

    def validate(self) -> None:
        if not self.name.strip() or not self.intervention_point.strip():
            raise ValueError("repair parameter group name and intervention point must be non-empty")
        if not self.parameters:
            raise ValueError("repair parameter groups must contain at least one parameter")


@dataclass(frozen=True)
class RepresentationBatch:
    """Uniform dense-token boundary exposed by every pretrained-family adapter.

    Validation is structural and dtype-aware by default. ``inspect_values=True`` additionally scans
    the token mask. Concrete loaders remain responsible for deriving that mask from their actual
    patch/tubelet/causal-compression geometry.
    """

    tokens: Any
    pooled: Any
    token_mask: Any
    grid: tuple[int, int, int]
    provenance: AdapterProvenance
    discrete_ids: Any | None = None
    class_tokens: Any | None = None
    register_tokens: Any | None = None
    intermediates: Mapping[str, Any] = field(default_factory=dict)
    native: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, inspect_values: bool = False) -> None:
        tokens = _shape(self.tokens, field_name="tokens")
        pooled = _shape(self.pooled, field_name="pooled")
        mask = _shape(self.token_mask, field_name="token_mask")
        if len(tokens) != 3 or min(tokens) < 1:
            raise ValueError("tokens must have shape [B,N,D] with positive dimensions")
        if pooled != (tokens[0], tokens[2]):
            raise ValueError("pooled must have shape [B,D] matching tokens")
        if mask != tokens[:2]:
            raise ValueError("token_mask must have shape [B,N]")
        _require_dtype(self.tokens, field_name="tokens", kind="float")
        _require_dtype(self.pooled, field_name="pooled", kind="float")
        _require_dtype(self.token_mask, field_name="token_mask", kind="bool")
        if len(self.grid) != 3 or any(dimension < 1 for dimension in self.grid):
            raise ValueError("grid must contain positive (T_lat,H_lat,W_lat) dimensions")
        if self.grid[0] * self.grid[1] * self.grid[2] != tokens[1]:
            raise ValueError("grid product must equal the dense token count")
        if self.discrete_ids is not None:
            discrete = _shape(self.discrete_ids, field_name="discrete_ids")
            if discrete != tokens[:2]:
                raise ValueError("discrete_ids must have shape [B,N] matching tokens")
            _require_dtype(self.discrete_ids, field_name="discrete_ids", kind="integer")
        if self.class_tokens is not None and _shape(
            self.class_tokens, field_name="class_tokens"
        ) != (tokens[0], tokens[2]):
            raise ValueError("class_tokens must have shape [B,D] matching tokens")
        if self.register_tokens is not None:
            registers = _shape(self.register_tokens, field_name="register_tokens")
            if len(registers) != 3 or registers[0] != tokens[0] or registers[2] != tokens[2]:
                raise ValueError("register_tokens must have shape [B,R,D] matching tokens")
        if not isinstance(self.intermediates, Mapping) or not isinstance(self.native, Mapping):
            raise ValueError("intermediates and native must be mappings")
        if inspect_values:
            mask_values = _inspected_values(self.token_mask)
            if mask_values is None:
                raise ValueError("strict token-mask validation requires an inspectable tensor")
            if len(mask_values) != tokens[0] * tokens[1] or any(
                value not in (False, True, 0, 1) for value in mask_values
            ):
                raise ValueError("token_mask values must be boolean")
            for batch_index in range(tokens[0]):
                start = batch_index * tokens[1]
                if not any(bool(value) for value in mask_values[start : start + tokens[1]]):
                    raise ValueError("every sample must contain at least one valid token")
        self.provenance.validate()


@dataclass(frozen=True)
class ActionProfileBatch:
    """Counterfactual action profiles from saved-state physical branching."""

    joint_margins: Any
    normalized_constraint_margins: Any
    skill_ids: tuple[str, ...]
    constraint_names: tuple[str, ...]
    constraint_scales: tuple[float, ...]
    profile_source: str

    def validate(self) -> None:
        joint = _shape(self.joint_margins, field_name="joint_margins")
        raw = _shape(
            self.normalized_constraint_margins,
            field_name="normalized_constraint_margins",
        )
        if len(joint) != 2 or min(joint) < 1:
            raise ValueError("joint_margins must have shape [B,K]")
        if len(raw) != 3 or raw[:2] != joint:
            raise ValueError("normalized_constraint_margins must have shape [B,K,C]")
        if (
            len(self.skill_ids) != joint[1]
            or any(not skill_id for skill_id in self.skill_ids)
            or len(set(self.skill_ids)) != len(self.skill_ids)
        ):
            raise ValueError("skill_ids must uniquely identify every action-profile column")
        if (
            len(self.constraint_names) != raw[2]
            or any(not name for name in self.constraint_names)
            or len(set(self.constraint_names)) != raw[2]
        ):
            raise ValueError("constraint_names must uniquely identify every constraint channel")
        if len(self.constraint_scales) != raw[2] or any(
            not math.isfinite(scale) or scale <= 0.0 for scale in self.constraint_scales
        ):
            raise ValueError("constraint_scales must be positive and match constraint channels")
        if not self.profile_source.strip():
            raise ValueError("profile_source must be non-empty")
        _require_dtype(self.joint_margins, field_name="joint_margins", kind="float")
        _require_dtype(
            self.normalized_constraint_margins,
            field_name="normalized_constraint_margins",
            kind="float",
        )
        joint_values = _inspected_values(self.joint_margins)
        constraint_values = _inspected_values(self.normalized_constraint_margins)
        if joint_values is not None and constraint_values is not None:
            expected_joint_size = joint[0] * joint[1]
            expected_constraint_size = expected_joint_size * raw[2]
            if len(joint_values) != expected_joint_size or len(constraint_values) != (
                expected_constraint_size
            ):
                raise ValueError("inspectable action-profile values disagree with their shapes")
            normalized_joint = _require_finite(joint_values, field_name="joint_margins")
            normalized_constraints = _require_finite(
                constraint_values,
                field_name="normalized_constraint_margins",
            )
            for profile_index, observed_joint in enumerate(normalized_joint):
                start = profile_index * raw[2]
                expected_joint = min(normalized_constraints[start : start + raw[2]])
                if not math.isclose(observed_joint, expected_joint, rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(
                        "joint_margins must equal the minimum normalized constraint margin"
                    )


@runtime_checkable
class RepresentationAdapter(Protocol):
    """Adapter boundary for frozen extraction and declared pre-bottleneck repair."""

    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def preprocess(self, batch: VisualBatch) -> Any: ...

    def encode(
        self,
        batch: VisualBatch,
        *,
        return_intermediates: bool = False,
    ) -> RepresentationBatch: ...

    def repair_parameters(self) -> tuple[RepairParameterGroup, ...]: ...

    def checkpoint_manifest(self) -> Mapping[str, Any]: ...


@runtime_checkable
class BranchableDomainAdapter(Protocol):
    """Saved-state simulator boundary used to generate physical action profiles."""

    def observe(self, snapshot_id: str) -> VisualBatch: ...

    def branch_action_profiles(self, snapshot_ids: tuple[str, ...]) -> ActionProfileBatch: ...

    def manifest(self) -> Mapping[str, Any]: ...
