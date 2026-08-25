"""CPU-testable training components for the frozen offline FCSRL adaptation.

The main experiment trainer does not import this module yet.  It provides the narrow pieces needed
to do so without changing the world-model backbone: deterministic trajectory-local windows, a
stop-gradient EMA encoder, the categorical feasibility loss, and one optimizer-step primitive.
PyTorch remains an optional dependency and is passed in through :class:`TorchModules`.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from latent_safety.learning.config import DataConfig, ModelConfig
from latent_safety.learning.fcsrl_protocol import (
    FCSRL_ATOM_COUNT,
    FCSRL_DISCOUNT,
    FCSRL_ENCODER_EMA_SOURCE_MIX,
    FCSRL_RETURN_LENGTH,
    FCSRL_SYMLOG_HIGH,
    FCSRL_SYMLOG_LOW,
    FCSRL_UNROLL_LENGTH,
    FCSRLProtocolError,
)
from latent_safety.learning.runtime import TorchModules


_EMA_COMPONENTS = ("frame_encoder", "history_encoder", "posterior_mean")


@dataclass(frozen=True, order=True)
class FCSRLWindowRef:
    """Identity of one trajectory-local ten-transition window."""

    trajectory_id: str
    start_timestep: int
    trajectory_index: int


@dataclass(frozen=True)
class FCSRLTorchTargetTrace:
    """Tensor counterpart of the dependency-free :class:`FCSRLTargetTrace`."""

    targets: Any
    mask: Any
    projected_targets: Any
    train_targets: Any
    train_mask: Any


@dataclass(frozen=True)
class FCSRLFeasibilityOutput:
    """Differentiable feasibility loss and detached target diagnostics."""

    loss: Any
    per_position_loss: Any
    logits: Any
    rollout_latents: Any
    bootstrap_values: Any
    target_trace: FCSRLTorchTargetTrace
    active_positions: int


@dataclass(frozen=True)
class FCSRLTrainingStepResult:
    """Scalar record returned after one successful optimizer and EMA update."""

    total_loss: float
    base_loss: float
    feasibility_loss: float
    feasibility_weight: float
    active_positions: int
    gradient_norm: float | None


def _require_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FCSRLProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _require_tensor(torch: Any, value: Any, *, name: str, ndim: int) -> Any:
    if not torch.is_tensor(value):
        raise FCSRLProtocolError(f"{name} must be a PyTorch tensor")
    if value.ndim != ndim:
        raise FCSRLProtocolError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _require_finite(torch: Any, value: Any, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FCSRLProtocolError(f"{name} must be finite")


class ControlledFCSRLWindowDataset:
    """Deterministic contiguous windows over one existing controlled-video split.

    Windows are sorted by trajectory ID and then timestep, never cross trajectories, and include
    histories ``H_0, ..., H_10`` for ten actions. There is one window per original transition, so
    base-objective exposure matches the ordinary loader. Positions beyond a finite trajectory repeat
    its terminal history and are masked strictly after the retained time-limit truncation. These
    controlled fixtures do not have true environment terminations.
    """

    def __init__(self, dataset: Any, *, torch: Any) -> None:
        self.dataset = dataset
        self.torch = torch
        horizon = _require_integer(dataset.config.horizon, name="data horizon", minimum=1)
        if horizon < FCSRL_RETURN_LENGTH:
            raise FCSRLProtocolError(
                f"data horizon must be at least {FCSRL_RETURN_LENGTH} transitions"
            )
        indexed: list[tuple[str, int]] = []
        seen_ids: set[str] = set()
        for trajectory_index in dataset.trajectory_indices:
            trajectory = dataset.trajectories[trajectory_index]
            trajectory_id = str(trajectory.trajectory_id)
            if not trajectory_id:
                raise FCSRLProtocolError("trajectory IDs must be non-empty")
            if trajectory_id in seen_ids:
                raise FCSRLProtocolError(f"duplicate trajectory ID: {trajectory_id}")
            seen_ids.add(trajectory_id)
            if len(trajectory.actions) != horizon:
                raise FCSRLProtocolError(
                    f"trajectory {trajectory_id} has {len(trajectory.actions)} actions, "
                    f"expected {horizon}"
                )
            if len(trajectory.safety_margins) != horizon:
                raise FCSRLProtocolError(
                    f"trajectory {trajectory_id} must have one safety margin per transition"
                )
            reference = dataset.rendered_frames[trajectory_index]
            if not self.torch.is_tensor(reference) or reference.shape[0] != horizon + 1:
                raise FCSRLProtocolError(
                    f"trajectory {trajectory_id} must have horizon + 1 rendered frames"
                )
            indexed.append((trajectory_id, trajectory_index))
        if not indexed:
            raise FCSRLProtocolError("FCSRL window dataset cannot be empty")
        indexed.sort()
        self.refs = tuple(
            FCSRLWindowRef(
                trajectory_id=trajectory_id,
                start_timestep=start,
                trajectory_index=trajectory_index,
            )
            for trajectory_id, trajectory_index in indexed
            for start in range(horizon)
        )

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        trajectory = self.dataset.trajectories[ref.trajectory_index]
        start = ref.start_timestep
        stop = start + FCSRL_RETURN_LENGTH
        horizon = self.dataset.config.horizon
        histories = self.torch.stack(
            [
                self.dataset._history(  # noqa: SLF001
                    ref.trajectory_index,
                    min(timestep, horizon),
                )
                for timestep in range(start, stop + 1)
            ],
            dim=0,
        )
        reference = histories
        transition_timesteps = tuple(range(start, stop))
        padding_action = float(self.dataset.config.actions[0])
        actions = reference.new_tensor(
            [
                trajectory.actions[timestep]
                if timestep < horizon
                else padding_action
                for timestep in transition_timesteps
            ]
        ).unsqueeze(-1)
        safety_margins = reference.new_tensor(
            [
                trajectory.safety_margins[timestep]
                if timestep < horizon
                else trajectory.safety_margins[-1]
                for timestep in transition_timesteps
            ]
        )
        violations = self.torch.tensor(
            [
                trajectory.safety_margins[timestep] < 0.0
                if timestep < horizon
                else False
                for timestep in transition_timesteps
            ],
            dtype=self.torch.bool,
            device=reference.device,
        )
        terminations = self.torch.zeros(
            FCSRL_RETURN_LENGTH,
            dtype=self.torch.bool,
            device=reference.device,
        )
        truncations = self.torch.tensor(
            [timestep == horizon - 1 for timestep in transition_timesteps],
            dtype=self.torch.bool,
            device=reference.device,
        )
        mask = torch_sequence_mask(
            self.torch,
            terminations.unsqueeze(0),
            truncations.unsqueeze(0),
        ).squeeze(0)
        _require_finite(self.torch, histories, name="window histories")
        _require_finite(self.torch, actions, name="window actions")
        _require_finite(self.torch, safety_margins, name="window safety margins")
        return {
            "trajectory_id": ref.trajectory_id,
            "start_timestep": start,
            "sample_ids": tuple(
                (
                    f"{ref.trajectory_id}:{timestep:04d}"
                    if timestep < horizon
                    else f"{ref.trajectory_id}:pad:{timestep - horizon:04d}"
                )
                for timestep in transition_timesteps
            ),
            "histories": histories,
            "actions": actions,
            "safety_margins": safety_margins,
            "violations": violations,
            "terminations": terminations,
            "truncations": truncations,
            "mask": mask,
        }


def deterministic_batch_indices(
    length: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return a platform-stable batch plan without ambient random state."""

    length = _require_integer(length, name="length", minimum=1)
    batch_size = _require_integer(batch_size, name="batch_size", minimum=1)
    seed = _require_integer(seed, name="seed", minimum=0)
    epoch = _require_integer(epoch, name="epoch", minimum=0)
    if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
        raise FCSRLProtocolError("shuffle and drop_last must be booleans")
    indices = list(range(length))
    if shuffle:
        indices.sort(
            key=lambda index: (
                hashlib.sha256(
                    f"fcsrl-window-order-v1|{seed}|{epoch}|{index}".encode()
                ).digest(),
                index,
            )
        )
    batches = [
        tuple(indices[offset : offset + batch_size])
        for offset in range(0, length, batch_size)
    ]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches.pop()
    if not batches:
        raise FCSRLProtocolError("batch plan would contain no batches")
    return tuple(batches)


def collate_fcsrl_windows(torch: Any, windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Strictly collate windows and revalidate their declared masks."""

    if not windows:
        raise FCSRLProtocolError("cannot collate an empty window list")
    tensor_keys = (
        "histories",
        "actions",
        "safety_margins",
        "violations",
        "terminations",
        "truncations",
        "mask",
    )
    for window in windows:
        missing = [key for key in tensor_keys if key not in window]
        if missing:
            raise FCSRLProtocolError(f"window is missing fields: {missing}")
    try:
        batch = {key: torch.stack([window[key] for window in windows]) for key in tensor_keys}
    except (RuntimeError, TypeError) as error:
        raise FCSRLProtocolError(f"windows do not have stack-compatible shapes: {error}") from error
    exact_mask = torch_sequence_mask(torch, batch["terminations"], batch["truncations"])
    if batch["mask"].dtype != torch.bool or not bool(torch.equal(batch["mask"], exact_mask)):
        raise FCSRLProtocolError("window mask does not match exact ending-transition semantics")
    batch["trajectory_id"] = tuple(str(window["trajectory_id"]) for window in windows)
    batch["start_timestep"] = tuple(int(window["start_timestep"]) for window in windows)
    batch["sample_ids"] = tuple(tuple(window["sample_ids"]) for window in windows)
    return batch


def iter_fcsrl_batches(
    dataset: Any,
    *,
    torch: Any,
    batch_size: int,
    seed: int,
    epoch: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield batches following :func:`deterministic_batch_indices`."""

    plan = deterministic_batch_indices(
        len(dataset),
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    for indices in plan:
        yield collate_fcsrl_windows(torch, [dataset[index] for index in indices])


def build_ema_target_encoder(
    modules: TorchModules,
    online_model: Any,
    model_config: ModelConfig,
    data_config: DataConfig,
) -> Any:
    """Clone only the posterior-mean encoder path and freeze it permanently."""

    torch = modules.torch
    nn = modules.nn
    for name in _EMA_COMPONENTS:
        if not hasattr(online_model, name):
            raise FCSRLProtocolError(f"online model is missing encoder component {name!r}")
    if model_config.history_encoder not in {"stack", "gru"}:
        raise FCSRLProtocolError("unsupported history encoder for EMA target")

    class EMATargetEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frame_encoder = copy.deepcopy(online_model.frame_encoder)
            self.history_encoder = copy.deepcopy(online_model.history_encoder)
            self.posterior_mean = copy.deepcopy(online_model.posterior_mean)
            self.history_encoder_name = model_config.history_encoder
            self.history_length = data_config.history_length
            self.channels = data_config.channels
            self.image_size = data_config.image_size
            self.hidden_dim = model_config.hidden_dim
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            super().train(False)

        def train(self, mode: bool = True) -> Any:
            del mode
            return super().train(False)

        def forward(self, history: Any) -> Any:
            if not torch.is_tensor(history) or history.ndim != 5:
                raise FCSRLProtocolError(
                    "EMA history must have shape [batch, history, channel, height, width]"
                )
            expected = (
                self.history_length,
                self.channels,
                self.image_size,
                self.image_size,
            )
            if tuple(history.shape[1:]) != expected:
                raise FCSRLProtocolError(
                    f"EMA history tail must be {expected}, got {tuple(history.shape[1:])}"
                )
            _require_finite(torch, history, name="EMA history")
            batch_size = history.shape[0]
            with torch.no_grad():
                frames = history.reshape(
                    batch_size * self.history_length,
                    self.channels,
                    self.image_size,
                    self.image_size,
                )
                features = self.frame_encoder(frames).view(
                    batch_size, self.history_length, self.hidden_dim
                )
                if self.history_encoder_name == "gru":
                    encoded, _ = self.history_encoder(features)
                    context = encoded[:, -1]
                else:
                    context = self.history_encoder(
                        torch.cat(
                            (features[:, -1], features[:, -1] - features[:, 0]),
                            dim=-1,
                        )
                    )
                latent = self.posterior_mean(context)
            _require_finite(torch, latent, name="EMA latent")
            return latent.detach()

    return EMATargetEncoder()


def _named_encoder_parameters(model: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for component_name in _EMA_COMPONENTS:
        component = getattr(model, component_name, None)
        if component is None:
            raise FCSRLProtocolError(f"model is missing encoder component {component_name!r}")
        for name, parameter in component.named_parameters():
            parameters[f"{component_name}.{name}"] = parameter
    return parameters


def _named_encoder_buffers(model: Any) -> dict[str, Any]:
    buffers: dict[str, Any] = {}
    for component_name in _EMA_COMPONENTS:
        component = getattr(model, component_name, None)
        if component is None:
            raise FCSRLProtocolError(f"model is missing encoder component {component_name!r}")
        for name, buffer in component.named_buffers():
            buffers[f"{component_name}.{name}"] = buffer
    return buffers


def update_ema_target_encoder(
    torch: Any,
    target_encoder: Any,
    online_model: Any,
    *,
    source_mix: float = FCSRL_ENCODER_EMA_SOURCE_MIX,
) -> None:
    """Apply ``target <- (1-source_mix) target + source_mix online`` exactly once."""

    source_mix = float(source_mix)
    if not math.isfinite(source_mix) or not 0.0 < source_mix <= 1.0:
        raise FCSRLProtocolError("EMA source_mix must lie in (0, 1]")
    target_parameters = _named_encoder_parameters(target_encoder)
    source_parameters = _named_encoder_parameters(online_model)
    if target_parameters.keys() != source_parameters.keys():
        raise FCSRLProtocolError("EMA target and online encoder parameter names differ")
    target_buffers = _named_encoder_buffers(target_encoder)
    source_buffers = _named_encoder_buffers(online_model)
    if target_buffers.keys() != source_buffers.keys():
        raise FCSRLProtocolError("EMA target and online encoder buffer names differ")
    with torch.no_grad():
        for name in target_parameters:
            target = target_parameters[name]
            source = source_parameters[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise FCSRLProtocolError(f"EMA parameter mismatch for {name}")
            target.mul_(1.0 - source_mix).add_(source, alpha=source_mix)
        # Non-floating state (and normalization statistics, if introduced later) is copied rather
        # than averaged.  Current encoders have no buffers, but this keeps the contract fail-closed.
        for name in target_buffers:
            target = target_buffers[name]
            source = source_buffers[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise FCSRLProtocolError(f"EMA buffer mismatch for {name}")
            target.copy_(source)
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)
    target_encoder.eval()


def build_categorical_feasibility_head(
    modules: TorchModules,
    *,
    latent_dim: int,
    hidden_dim: int,
) -> Any:
    """Construct the only trainable module added by the adaptation."""

    latent_dim = _require_integer(latent_dim, name="latent_dim", minimum=1)
    hidden_dim = _require_integer(hidden_dim, name="hidden_dim", minimum=1)
    nn = modules.nn
    return nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, FCSRL_ATOM_COUNT),
    )


def torch_sequence_mask(torch: Any, terminations: Any, truncations: Any) -> Any:
    """Vectorized inclusive ending mask for tensors shaped ``[batch, 10]``."""

    terminations = _require_tensor(
        torch, terminations, name="terminations", ndim=2
    )
    truncations = _require_tensor(torch, truncations, name="truncations", ndim=2)
    if terminations.shape != truncations.shape or terminations.shape[1] != FCSRL_RETURN_LENGTH:
        raise FCSRLProtocolError(
            f"termination tensors must both have shape [batch, {FCSRL_RETURN_LENGTH}]"
        )
    if terminations.dtype != torch.bool or truncations.dtype != torch.bool:
        raise FCSRLProtocolError("termination and truncation tensors must have boolean dtype")
    if terminations.device != truncations.device:
        raise FCSRLProtocolError("termination and truncation tensors must share a device")
    if bool((terminations & truncations).any().item()):
        raise FCSRLProtocolError("a transition cannot be both terminated and truncated")
    endings = terminations | truncations
    endings_before = endings.to(dtype=torch.int64).cumsum(dim=1) - endings.to(
        dtype=torch.int64
    )
    return endings_before == 0


def _validate_binary_tensor(torch: Any, values: Any, *, name: str) -> Any:
    values = _require_tensor(torch, values, name=name, ndim=2)
    if values.shape[1] != FCSRL_RETURN_LENGTH:
        raise FCSRLProtocolError(
            f"{name} must have shape [batch, {FCSRL_RETURN_LENGTH}]"
        )
    if values.dtype == torch.bool:
        return values
    _require_finite(torch, values, name=name)
    if not bool(((values == 0) | (values == 1)).all().item()):
        raise FCSRLProtocolError(f"{name} must be binary")
    return values.to(dtype=torch.bool)


def torch_project_categorical_targets(torch: Any, targets: Any) -> Any:
    """Differentiation-free adjacent-atom projection for an arbitrary target tensor."""

    if not torch.is_tensor(targets) or not targets.is_floating_point():
        raise FCSRLProtocolError("categorical targets must be a floating-point tensor")
    _require_finite(torch, targets, name="categorical targets")
    detached = targets.detach()
    encoded = torch.sign(detached) * torch.log1p(detached.abs())
    encoded = encoded.clamp(FCSRL_SYMLOG_LOW, FCSRL_SYMLOG_HIGH)
    spacing = (FCSRL_SYMLOG_HIGH - FCSRL_SYMLOG_LOW) / (FCSRL_ATOM_COUNT - 1)
    position = (encoded - FCSRL_SYMLOG_LOW) / spacing
    lower = position.floor().to(dtype=torch.int64).clamp(0, FCSRL_ATOM_COUNT - 1)
    upper = position.ceil().to(dtype=torch.int64).clamp(0, FCSRL_ATOM_COUNT - 1)
    upper_weight = position - lower.to(dtype=position.dtype)
    upper_weight = torch.where(lower == upper, torch.zeros_like(upper_weight), upper_weight)
    projected = torch.zeros(
        (*targets.shape, FCSRL_ATOM_COUNT),
        dtype=targets.dtype,
        device=targets.device,
    )
    projected.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
    projected.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return projected.detach()


def torch_categorical_mean_from_logits(torch: Any, logits: Any) -> Any:
    """Decode categorical logits using the frozen mean-in-symlog-space rule."""

    if not torch.is_tensor(logits) or not logits.is_floating_point():
        raise FCSRLProtocolError("categorical logits must be a floating-point tensor")
    if logits.ndim < 2 or logits.shape[-1] != FCSRL_ATOM_COUNT:
        raise FCSRLProtocolError(
            f"categorical logits must end in {FCSRL_ATOM_COUNT} atoms"
        )
    _require_finite(torch, logits, name="categorical logits")
    atoms = torch.linspace(
        FCSRL_SYMLOG_LOW,
        FCSRL_SYMLOG_HIGH,
        FCSRL_ATOM_COUNT,
        dtype=logits.dtype,
        device=logits.device,
    )
    encoded_mean = (torch.softmax(logits, dim=-1) * atoms).sum(dim=-1)
    decoded = torch.sign(encoded_mean) * torch.expm1(encoded_mean.abs())
    _require_finite(torch, decoded, name="decoded categorical mean")
    return decoded


def build_torch_target_trace(
    torch: Any,
    violations: Any,
    terminations: Any,
    truncations: Any,
    bootstrap_values: Any,
    *,
    discount: float = FCSRL_DISCOUNT,
) -> FCSRLTorchTargetTrace:
    """Vectorized frozen recursion; all returned targets are stop-gradient tensors."""

    violations = _validate_binary_tensor(torch, violations, name="violations")
    terminations = _require_tensor(torch, terminations, name="terminations", ndim=2)
    truncations = _require_tensor(torch, truncations, name="truncations", ndim=2)
    bootstrap_values = _require_tensor(
        torch, bootstrap_values, name="bootstrap_values", ndim=2
    )
    expected_shape = violations.shape
    if any(
        tensor.shape != expected_shape
        for tensor in (terminations, truncations, bootstrap_values)
    ):
        raise FCSRLProtocolError("all target-recursion tensors must have the same shape")
    if not bootstrap_values.is_floating_point():
        raise FCSRLProtocolError("bootstrap_values must be floating point")
    if any(
        tensor.device != bootstrap_values.device
        for tensor in (violations, terminations, truncations)
    ):
        raise FCSRLProtocolError("all target-recursion tensors must share a device")
    _require_finite(torch, bootstrap_values, name="bootstrap_values")
    discount = float(discount)
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise FCSRLProtocolError("discount must lie in [0, 1]")
    mask = torch_sequence_mask(torch, terminations, truncations)
    bootstrap = bootstrap_values.detach()
    violation_values = violations.to(dtype=bootstrap.dtype)
    targets = torch.empty_like(bootstrap)
    targets[:, -1] = bootstrap[:, -1]
    for timestep in range(FCSRL_RETURN_LENGTH - 2, -1, -1):
        continuation = torch.where(
            truncations[:, timestep],
            bootstrap[:, timestep],
            targets[:, timestep + 1],
        )
        discounted = (
            discount
            * (~terminations[:, timestep]).to(dtype=bootstrap.dtype)
            * continuation
        )
        targets[:, timestep] = torch.maximum(violation_values[:, timestep], discounted)
    targets = targets.detach()
    projected = torch_project_categorical_targets(torch, targets)
    return FCSRLTorchTargetTrace(
        targets=targets,
        mask=mask,
        projected_targets=projected,
        train_targets=targets[:, :FCSRL_UNROLL_LENGTH],
        train_mask=mask[:, :FCSRL_UNROLL_LENGTH],
    )


def _validate_training_batch(torch: Any, batch: Mapping[str, Any]) -> tuple[int, Any, Any]:
    required = (
        "histories",
        "actions",
        "violations",
        "terminations",
        "truncations",
        "mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise FCSRLProtocolError(f"FCSRL batch is missing fields: {missing}")
    histories = _require_tensor(torch, batch["histories"], name="histories", ndim=6)
    actions = _require_tensor(torch, batch["actions"], name="actions", ndim=3)
    batch_size = histories.shape[0]
    if batch_size < 1 or histories.shape[1] != FCSRL_RETURN_LENGTH + 1:
        raise FCSRLProtocolError(
            f"histories must have shape [batch, {FCSRL_RETURN_LENGTH + 1}, H, C, Y, X]"
        )
    if actions.shape != (batch_size, FCSRL_RETURN_LENGTH, 1):
        raise FCSRLProtocolError(
            f"actions must have shape [batch, {FCSRL_RETURN_LENGTH}, 1]"
        )
    if not histories.is_floating_point() or not actions.is_floating_point():
        raise FCSRLProtocolError("histories and actions must be floating point")
    _require_finite(torch, histories, name="histories")
    _require_finite(torch, actions, name="actions")
    for name in ("violations", "terminations", "truncations", "mask"):
        tensor = _require_tensor(torch, batch[name], name=name, ndim=2)
        if tensor.shape != (batch_size, FCSRL_RETURN_LENGTH):
            raise FCSRLProtocolError(
                f"{name} must have shape [batch, {FCSRL_RETURN_LENGTH}]"
            )
    exact_mask = torch_sequence_mask(torch, batch["terminations"], batch["truncations"])
    if batch["mask"].dtype != torch.bool or not bool(torch.equal(batch["mask"], exact_mask)):
        raise FCSRLProtocolError("batch mask does not match exact ending-transition semantics")
    _validate_binary_tensor(torch, batch["violations"], name="violations")
    return batch_size, histories, actions


def compute_fcsrl_feasibility_loss(
    modules: TorchModules,
    online_model: Any,
    target_encoder: Any,
    feasibility_head: Any,
    batch: Mapping[str, Any],
) -> FCSRLFeasibilityOutput:
    """Compute the first-four categorical loss on online transition rollouts.

    ``z_0`` is the online posterior mean for ``H_0``.  The existing world-model transition produces
    ``z_1, z_2, z_3`` from the first three behavior actions.  Targets use the same head under
    stop-gradient on EMA encodings of ``H_1, ..., H_10``.
    """

    torch = modules.torch
    functional = modules.functional
    batch_size, histories, actions = _validate_training_batch(torch, batch)
    if any(parameter.requires_grad for parameter in target_encoder.parameters()):
        raise FCSRLProtocolError("EMA target encoder parameters must be frozen")
    target_encoder.eval()
    with torch.no_grad():
        target_histories = histories[:, 1:].reshape(
            batch_size * FCSRL_RETURN_LENGTH, *histories.shape[2:]
        )
        target_latents = target_encoder(target_histories)
        target_logits = feasibility_head(target_latents)
        if target_logits.shape != (batch_size * FCSRL_RETURN_LENGTH, FCSRL_ATOM_COUNT):
            raise FCSRLProtocolError(
                "feasibility head must emit one 63-atom distribution per latent"
            )
        bootstrap_values = torch_categorical_mean_from_logits(
            torch, target_logits
        ).reshape(batch_size, FCSRL_RETURN_LENGTH)
    target_trace = build_torch_target_trace(
        torch,
        batch["violations"],
        batch["terminations"],
        batch["truncations"],
        bootstrap_values,
    )

    encoded = online_model.encode(histories[:, 0], sample=False)
    if not isinstance(encoded, Mapping) or "z" not in encoded:
        raise FCSRLProtocolError("online model encode() must return a mapping containing 'z'")
    latent = encoded["z"]
    if not torch.is_tensor(latent) or latent.ndim != 2 or latent.shape[0] != batch_size:
        raise FCSRLProtocolError("online encoder must return latents shaped [batch, latent_dim]")
    _require_finite(torch, latent, name="online latent")
    rollout_latents = [latent]
    for timestep in range(FCSRL_UNROLL_LENGTH - 1):
        latent = online_model.predict_next(latent, actions[:, timestep])
        if not torch.is_tensor(latent) or latent.shape != rollout_latents[0].shape:
            raise FCSRLProtocolError("online transition changed the latent shape")
        _require_finite(torch, latent, name="online rollout latent")
        rollout_latents.append(latent)
    stacked_latents = torch.stack(rollout_latents, dim=1)
    logits = feasibility_head(stacked_latents)
    expected_logits = (batch_size, FCSRL_UNROLL_LENGTH, FCSRL_ATOM_COUNT)
    if logits.shape != expected_logits:
        raise FCSRLProtocolError(
            f"feasibility logits must have shape {expected_logits}, got {tuple(logits.shape)}"
        )
    _require_finite(torch, logits, name="feasibility logits")
    projected = target_trace.projected_targets[:, :FCSRL_UNROLL_LENGTH]
    per_position = -(projected * functional.log_softmax(logits, dim=-1)).sum(dim=-1)
    mask_weights = target_trace.train_mask.to(dtype=per_position.dtype)
    active_positions = int(target_trace.train_mask.sum().item())
    if active_positions < 1:
        raise FCSRLProtocolError("FCSRL batch has no active train positions")
    loss = (per_position * mask_weights).sum() / mask_weights.sum()
    _require_finite(torch, per_position, name="per-position feasibility loss")
    _require_finite(torch, loss, name="feasibility loss")
    return FCSRLFeasibilityOutput(
        loss=loss,
        per_position_loss=per_position,
        logits=logits,
        rollout_latents=stacked_latents,
        bootstrap_values=bootstrap_values.detach(),
        target_trace=target_trace,
        active_positions=active_positions,
    )


def _optimizer_parameter_ids(optimizer: Any) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }


def run_fcsrl_training_step(
    modules: TorchModules,
    online_model: Any,
    target_encoder: Any,
    feasibility_head: Any,
    optimizer: Any,
    batch: Mapping[str, Any],
    *,
    base_loss: Any,
    feasibility_weight: float,
    gradient_clip_norm: float | None = None,
) -> FCSRLTrainingStepResult:
    """Apply one combined same-backbone optimizer step, then exactly one EMA update.

    The caller constructs ``base_loss`` from the unchanged world-model objective.  This function
    deliberately does not choose a loss weight, checkpoint, validation rule, or training budget.
    """

    torch = modules.torch
    if not torch.is_tensor(base_loss) or base_loss.ndim != 0:
        raise FCSRLProtocolError("base_loss must be a scalar PyTorch tensor")
    _require_finite(torch, base_loss, name="base_loss")
    feasibility_weight = float(feasibility_weight)
    if not math.isfinite(feasibility_weight) or feasibility_weight <= 0.0:
        raise FCSRLProtocolError("feasibility_weight must be finite and positive")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
            raise FCSRLProtocolError("gradient_clip_norm must be finite and positive")
    required_parameters = [
        parameter
        for module in (online_model, feasibility_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not required_parameters:
        raise FCSRLProtocolError("training step has no trainable parameters")
    optimizer_ids = _optimizer_parameter_ids(optimizer)
    missing_count = sum(id(parameter) not in optimizer_ids for parameter in required_parameters)
    if missing_count:
        raise FCSRLProtocolError(
            f"optimizer is missing {missing_count} trainable world-model/head parameters"
        )

    optimizer.zero_grad(set_to_none=True)
    feasibility = compute_fcsrl_feasibility_loss(
        modules,
        online_model,
        target_encoder,
        feasibility_head,
        batch,
    )
    total_loss = base_loss + feasibility_weight * feasibility.loss
    _require_finite(torch, total_loss, name="combined FCSRL training loss")
    total_loss.backward()
    for parameter in required_parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise FCSRLProtocolError("training step produced a non-finite gradient")
    gradient_norm: float | None = None
    if gradient_clip_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(required_parameters, gradient_clip_norm)
        if not bool(torch.isfinite(norm).item()):
            raise FCSRLProtocolError("training step produced a non-finite gradient norm")
        gradient_norm = float(norm.detach().item())
    optimizer.step()
    update_ema_target_encoder(torch, target_encoder, online_model)
    return FCSRLTrainingStepResult(
        total_loss=float(total_loss.detach().item()),
        base_loss=float(base_loss.detach().item()),
        feasibility_loss=float(feasibility.loss.detach().item()),
        feasibility_weight=feasibility_weight,
        active_positions=feasibility.active_positions,
        gradient_norm=gradient_norm,
    )
