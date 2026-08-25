"""Dependency-free configuration schema for the optional PyTorch experiments.

Keeping this module in the Python standard library is deliberate: experiment plans can be
validated on login nodes and in CI before a heavyweight learning environment is installed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latent_safety.learning.fcsrl_protocol import FCSRL_RETURN_LENGTH


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the declared protocol."""


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    calibration: float
    test: float

    def items(self) -> tuple[tuple[str, float], ...]:
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("calibration", self.calibration),
            ("test", self.test),
        )


@dataclass(frozen=True)
class RunConfig:
    seed: int
    output_dir: str
    device: str
    deterministic: bool
    deterministic_warn_only: bool
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    amp: bool
    log_every: int


@dataclass(frozen=True)
class DataConfig:
    task: str
    trajectories: int
    horizon: int
    image_size: int
    channels: int
    history_length: int
    actions: tuple[float, ...]
    action_profile_horizon: int
    splits: SplitConfig
    position_limit: float
    render_extent: float
    dt: float
    damping: float
    acceleration: float
    process_noise: float
    nuisance_strength: float
    corrective_policy_probability: float
    challenge_fraction: float
    challenge_corrective_probability: float


@dataclass(frozen=True)
class ModelConfig:
    family: str
    history_encoder: str
    latent_dim: int
    hidden_dim: int
    transition_hidden_dim: int
    posterior_logvar_min: float
    posterior_logvar_max: float


@dataclass(frozen=True)
class ObjectiveConfig:
    safety_arm: str
    reconstruction_weight: float
    transition_weight: float
    kl_weight: float
    safety_weight: float
    margin_scale: float
    boundary_band: float
    positive_margin_band: float
    contrastive_margin: float
    max_contrastive_pairs: int
    fcsrl_head_hidden_dim: int


@dataclass(frozen=True)
class EvaluationConfig:
    rollout_horizons: tuple[int, ...]
    max_rollout_cases: int
    emit_audit_records: bool
    max_audit_records_per_split: int


@dataclass(frozen=True)
class LearningConfig:
    schema_version: int
    experiment: str
    status: str
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    objective: ObjectiveConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/TOML-compatible representation with stable tuple handling."""

        return dataclasses.asdict(self)


_DEVICES = {"auto", "cpu", "cuda", "mps"}
_MODEL_FAMILIES = {"ae", "beta_vae"}
_HISTORY_ENCODERS = {"stack", "gru"}
_SAFETY_ARMS = {
    "none",
    "h_prediction",
    "boundary_contrastive",
    "safe_action_profile",
    "nonprivileged_predicted_action_profile",
    "fcsrl_feasibility_loss_adaptation",
}
_TASKS = {
    "controlled_cart_video",
    "controlled_pendulum_video",
    "controlled_dubins_navigation_pixels",
}


def _table(parent: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{context}.{key} must be a TOML table")
    return value


def _value(table: dict[str, Any], key: str, expected: type, *, context: str) -> Any:
    if key not in table:
        raise ConfigError(f"missing required field {context}.{key}")
    value = table[key]
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{context}.{key} must be an integer")
    if expected is float and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ConfigError(f"{context}.{key} must be numeric")
    if expected not in {int, float} and not isinstance(value, expected):
        raise ConfigError(f"{context}.{key} must be {expected.__name__}")
    return value


def _positive(value: float | int, name: str, *, allow_zero: bool = False) -> None:
    valid = value >= 0 if allow_zero else value > 0
    if not valid or not math.isfinite(float(value)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be finite and {qualifier}")


def _fraction(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ConfigError(f"{name} must lie strictly between zero and one")


def _float_tuple(table: dict[str, Any], key: str, *, context: str) -> tuple[float, ...]:
    if key not in table or not isinstance(table[key], (list, tuple)):
        raise ConfigError(f"{context}.{key} must be an array")
    values = table[key]
    if not values:
        raise ConfigError(f"{context}.{key} must be non-empty")
    converted: list[float] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{context}.{key}[{index}] must be numeric")
        converted.append(float(value))
    if any(not math.isfinite(value) for value in converted):
        raise ConfigError(f"{context}.{key} values must be finite")
    return tuple(converted)


def _int_tuple(table: dict[str, Any], key: str, *, context: str) -> tuple[int, ...]:
    if key not in table or not isinstance(table[key], (list, tuple)):
        raise ConfigError(f"{context}.{key} must be an array")
    values = table[key]
    if not values:
        raise ConfigError(f"{context}.{key} must be non-empty")
    converted: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{context}.{key}[{index}] must be an integer")
        converted.append(value)
    return tuple(converted)


def parse_config(payload: dict[str, Any]) -> LearningConfig:
    """Parse and validate a decoded TOML mapping."""

    schema_version = int(_value(payload, "schema_version", int, context="config"))
    experiment = str(_value(payload, "experiment", str, context="config"))
    status = str(_value(payload, "status", str, context="config"))
    run_raw = _table(payload, "run", context="config")
    data_raw = _table(payload, "data", context="config")
    split_raw = _table(data_raw, "splits", context="data")
    model_raw = _table(payload, "model", context="config")
    objective_raw = _table(payload, "objective", context="config")
    evaluation_raw = _table(payload, "evaluation", context="config")

    splits = SplitConfig(
        train=float(_value(split_raw, "train", float, context="data.splits")),
        validation=float(
            _value(split_raw, "validation", float, context="data.splits")
        ),
        calibration=float(
            _value(split_raw, "calibration", float, context="data.splits")
        ),
        test=float(_value(split_raw, "test", float, context="data.splits")),
    )
    run = RunConfig(
        seed=int(_value(run_raw, "seed", int, context="run")),
        output_dir=str(_value(run_raw, "output_dir", str, context="run")),
        device=str(_value(run_raw, "device", str, context="run")),
        deterministic=bool(_value(run_raw, "deterministic", bool, context="run")),
        deterministic_warn_only=bool(
            _value(run_raw, "deterministic_warn_only", bool, context="run")
        ),
        epochs=int(_value(run_raw, "epochs", int, context="run")),
        batch_size=int(_value(run_raw, "batch_size", int, context="run")),
        num_workers=int(_value(run_raw, "num_workers", int, context="run")),
        learning_rate=float(
            _value(run_raw, "learning_rate", float, context="run")
        ),
        weight_decay=float(_value(run_raw, "weight_decay", float, context="run")),
        gradient_clip_norm=float(
            _value(run_raw, "gradient_clip_norm", float, context="run")
        ),
        amp=bool(_value(run_raw, "amp", bool, context="run")),
        log_every=int(_value(run_raw, "log_every", int, context="run")),
    )
    data = DataConfig(
        task=str(_value(data_raw, "task", str, context="data")),
        trajectories=int(_value(data_raw, "trajectories", int, context="data")),
        horizon=int(_value(data_raw, "horizon", int, context="data")),
        image_size=int(_value(data_raw, "image_size", int, context="data")),
        channels=int(_value(data_raw, "channels", int, context="data")),
        history_length=int(
            _value(data_raw, "history_length", int, context="data")
        ),
        actions=_float_tuple(data_raw, "actions", context="data"),
        action_profile_horizon=int(
            _value(data_raw, "action_profile_horizon", int, context="data")
        ),
        splits=splits,
        position_limit=float(
            _value(data_raw, "position_limit", float, context="data")
        ),
        render_extent=float(
            _value(data_raw, "render_extent", float, context="data")
        ),
        dt=float(_value(data_raw, "dt", float, context="data")),
        damping=float(_value(data_raw, "damping", float, context="data")),
        acceleration=float(
            _value(data_raw, "acceleration", float, context="data")
        ),
        process_noise=float(
            _value(data_raw, "process_noise", float, context="data")
        ),
        nuisance_strength=float(
            _value(data_raw, "nuisance_strength", float, context="data")
        ),
        corrective_policy_probability=float(
            _value(
                data_raw,
                "corrective_policy_probability",
                float,
                context="data",
            )
        ),
        challenge_fraction=float(
            _value(data_raw, "challenge_fraction", float, context="data")
        ),
        challenge_corrective_probability=float(
            _value(
                data_raw,
                "challenge_corrective_probability",
                float,
                context="data",
            )
        ),
    )
    model = ModelConfig(
        family=str(_value(model_raw, "family", str, context="model")),
        history_encoder=str(
            _value(model_raw, "history_encoder", str, context="model")
        ),
        latent_dim=int(_value(model_raw, "latent_dim", int, context="model")),
        hidden_dim=int(_value(model_raw, "hidden_dim", int, context="model")),
        transition_hidden_dim=int(
            _value(model_raw, "transition_hidden_dim", int, context="model")
        ),
        posterior_logvar_min=float(
            _value(model_raw, "posterior_logvar_min", float, context="model")
        ),
        posterior_logvar_max=float(
            _value(model_raw, "posterior_logvar_max", float, context="model")
        ),
    )
    objective = ObjectiveConfig(
        safety_arm=str(
            _value(objective_raw, "safety_arm", str, context="objective")
        ),
        reconstruction_weight=float(
            _value(objective_raw, "reconstruction_weight", float, context="objective")
        ),
        transition_weight=float(
            _value(objective_raw, "transition_weight", float, context="objective")
        ),
        kl_weight=float(_value(objective_raw, "kl_weight", float, context="objective")),
        safety_weight=float(
            _value(objective_raw, "safety_weight", float, context="objective")
        ),
        margin_scale=float(
            _value(objective_raw, "margin_scale", float, context="objective")
        ),
        boundary_band=float(
            _value(objective_raw, "boundary_band", float, context="objective")
        ),
        positive_margin_band=float(
            _value(
                objective_raw,
                "positive_margin_band",
                float,
                context="objective",
            )
        ),
        contrastive_margin=float(
            _value(objective_raw, "contrastive_margin", float, context="objective")
        ),
        max_contrastive_pairs=int(
            _value(objective_raw, "max_contrastive_pairs", int, context="objective")
        ),
        fcsrl_head_hidden_dim=int(
            _value(objective_raw, "fcsrl_head_hidden_dim", int, context="objective")
        ),
    )
    evaluation = EvaluationConfig(
        rollout_horizons=_int_tuple(
            evaluation_raw, "rollout_horizons", context="evaluation"
        ),
        max_rollout_cases=int(
            _value(evaluation_raw, "max_rollout_cases", int, context="evaluation")
        ),
        emit_audit_records=bool(
            _value(evaluation_raw, "emit_audit_records", bool, context="evaluation")
        ),
        max_audit_records_per_split=int(
            _value(
                evaluation_raw,
                "max_audit_records_per_split",
                int,
                context="evaluation",
            )
        ),
    )
    config = LearningConfig(
        schema_version=schema_version,
        experiment=experiment,
        status=status,
        run=run,
        data=data,
        model=model,
        objective=objective,
        evaluation=evaluation,
    )
    validate_config(config)
    return config


def validate_config(config: LearningConfig) -> None:
    """Enforce invariants that make runs comparable and auditable."""

    if config.schema_version != 1:
        raise ConfigError("only schema_version = 1 is supported")
    if not config.experiment.strip():
        raise ConfigError("experiment must be non-empty")
    if config.run.device not in _DEVICES:
        raise ConfigError(f"run.device must be one of {sorted(_DEVICES)}")
    for field_name in ("epochs", "batch_size", "log_every"):
        _positive(getattr(config.run, field_name), f"run.{field_name}")
    _positive(config.run.num_workers, "run.num_workers", allow_zero=True)
    for field_name in (
        "learning_rate",
        "gradient_clip_norm",
    ):
        _positive(getattr(config.run, field_name), f"run.{field_name}")
    _positive(config.run.weight_decay, "run.weight_decay", allow_zero=True)

    if config.data.task not in _TASKS:
        raise ConfigError(f"data.task must be one of {sorted(_TASKS)}")
    _positive(config.data.trajectories, "data.trajectories")
    if config.data.trajectories < 8:
        raise ConfigError("data.trajectories must be at least eight")
    _positive(config.data.horizon, "data.horizon")
    _positive(config.data.history_length, "data.history_length")
    if config.data.history_length > config.data.horizon:
        raise ConfigError("data.history_length cannot exceed data.horizon")
    if config.data.image_size < 16:
        raise ConfigError("data.image_size must be at least 16")
    if config.data.channels not in {1, 3}:
        raise ConfigError("data.channels must be 1 or 3")
    if len(config.data.actions) < 2 or len(set(config.data.actions)) != len(
        config.data.actions
    ):
        raise ConfigError("data.actions must contain at least two unique values")
    _positive(config.data.action_profile_horizon, "data.action_profile_horizon")
    for name, fraction in config.data.splits.items():
        _fraction(fraction, f"data.splits.{name}")
    if not math.isclose(
        sum(value for _, value in config.data.splits.items()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ConfigError("data split fractions must sum to one")
    counts = trajectory_split_counts(config.data.trajectories, config.data.splits)
    if any(count == 0 for count in counts.values()):
        raise ConfigError("each trajectory split must receive at least one trajectory")
    for field_name in ("position_limit", "render_extent", "dt", "acceleration"):
        _positive(getattr(config.data, field_name), f"data.{field_name}")
    if config.data.render_extent <= config.data.position_limit:
        raise ConfigError("data.render_extent must exceed data.position_limit")
    if config.data.task == "controlled_pendulum_video":
        if config.data.position_limit >= math.pi:
            raise ConfigError(
                "controlled_pendulum_video requires data.position_limit < pi radians"
            )
        if config.data.render_extent > math.pi:
            raise ConfigError(
                "controlled_pendulum_video requires data.render_extent <= pi radians"
            )
        if min(config.data.actions) >= 0.0 or max(config.data.actions) <= 0.0:
            raise ConfigError(
                "controlled_pendulum_video actions must include both negative and "
                "positive torques"
            )
    if config.data.task == "controlled_dubins_navigation_pixels":
        if min(config.data.actions) >= 0.0 or max(config.data.actions) <= 0.0:
            raise ConfigError(
                "controlled_dubins_navigation_pixels actions must include both "
                "negative and positive steering rates"
            )
        if config.data.damping <= 0.0:
            raise ConfigError(
                "controlled_dubins_navigation_pixels requires positive forward speed "
                "in data.damping"
            )
    if not 0.0 <= config.data.damping <= 1.0:
        raise ConfigError("data.damping must lie in [0, 1]")
    _positive(config.data.process_noise, "data.process_noise", allow_zero=True)
    for field_name in (
        "nuisance_strength",
        "corrective_policy_probability",
        "challenge_fraction",
        "challenge_corrective_probability",
    ):
        value = getattr(config.data, field_name)
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"data.{field_name} must lie in [0, 1]")

    if config.model.family not in _MODEL_FAMILIES:
        raise ConfigError(f"model.family must be one of {sorted(_MODEL_FAMILIES)}")
    if config.model.history_encoder not in _HISTORY_ENCODERS:
        raise ConfigError(
            f"model.history_encoder must be one of {sorted(_HISTORY_ENCODERS)}"
        )
    for field_name in ("latent_dim", "hidden_dim", "transition_hidden_dim"):
        _positive(getattr(config.model, field_name), f"model.{field_name}")
    if config.model.posterior_logvar_min >= config.model.posterior_logvar_max:
        raise ConfigError("posterior_logvar_min must be below posterior_logvar_max")

    if config.objective.safety_arm not in _SAFETY_ARMS:
        raise ConfigError(
            f"objective.safety_arm must be one of {sorted(_SAFETY_ARMS)}"
        )
    for field_name in (
        "reconstruction_weight",
        "transition_weight",
        "kl_weight",
        "safety_weight",
    ):
        _positive(
            getattr(config.objective, field_name),
            f"objective.{field_name}",
            allow_zero=True,
        )
    if config.objective.reconstruction_weight + config.objective.transition_weight <= 0:
        raise ConfigError("at least one world-model utility weight must be positive")
    if config.model.family == "beta_vae" and config.objective.kl_weight <= 0:
        raise ConfigError("beta_vae requires a positive objective.kl_weight")
    if config.model.family == "ae" and config.objective.kl_weight != 0:
        raise ConfigError("ae requires objective.kl_weight = 0")
    if config.objective.safety_arm == "none" and config.objective.safety_weight != 0:
        raise ConfigError("the none arm requires objective.safety_weight = 0")
    if config.objective.safety_arm != "none" and config.objective.safety_weight <= 0:
        raise ConfigError("supervised safety arms require a positive safety_weight")
    for field_name in (
        "margin_scale",
        "boundary_band",
        "positive_margin_band",
        "contrastive_margin",
    ):
        _positive(getattr(config.objective, field_name), f"objective.{field_name}")
    if config.objective.contrastive_margin > 2.0:
        raise ConfigError(
            "objective.contrastive_margin cannot exceed two for unit-normalized codes"
        )
    _positive(config.objective.max_contrastive_pairs, "objective.max_contrastive_pairs")
    _positive(config.objective.fcsrl_head_hidden_dim, "objective.fcsrl_head_hidden_dim")
    if (
        config.objective.safety_arm == "fcsrl_feasibility_loss_adaptation"
        and config.data.horizon < FCSRL_RETURN_LENGTH
    ):
        raise ConfigError(
            "the FCSRL feasibility-loss adaptation requires data.horizon >= "
            f"{FCSRL_RETURN_LENGTH}"
        )

    horizons = config.evaluation.rollout_horizons
    if tuple(sorted(set(horizons))) != horizons or horizons[0] < 1:
        raise ConfigError("evaluation.rollout_horizons must be sorted unique positive integers")
    if horizons[-1] > config.data.horizon:
        raise ConfigError("rollout horizons cannot exceed the trajectory horizon")
    _positive(config.evaluation.max_rollout_cases, "evaluation.max_rollout_cases")
    _positive(
        config.evaluation.max_audit_records_per_split,
        "evaluation.max_audit_records_per_split",
    )


def trajectory_split_counts(total: int, splits: SplitConfig) -> dict[str, int]:
    """Allocate whole trajectories with the deterministic largest-remainder rule."""

    raw = [(name, total * fraction) for name, fraction in splits.items()]
    counts = {name: math.floor(value) for name, value in raw}
    remaining = total - sum(counts.values())
    ranked = sorted(raw, key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
    for name, _ in ranked[:remaining]:
        counts[name] += 1
    return counts


def dry_run_plan(config: LearningConfig) -> dict[str, Any]:
    """Summarize resource and protocol choices without importing PyTorch."""

    counts = trajectory_split_counts(config.data.trajectories, config.data.splits)
    sample_counts = {
        split: count * config.data.horizon for split, count in counts.items()
    }
    pixels_per_history = (
        config.data.history_length
        * config.data.channels
        * config.data.image_size
        * config.data.image_size
    )
    return {
        "schema_version": config.schema_version,
        "experiment": config.experiment,
        "status": config.status,
        "task": config.data.task,
        "torch_required": False,
        "selected_arm": config.objective.safety_arm,
        "representation": {
            "family": config.model.family,
            "history_mode": (
                f"{config.model.history_encoder}_h{config.data.history_length}"
            ),
            "history_encoder": config.model.history_encoder,
            "history_length": config.data.history_length,
            "latent_dim": config.model.latent_dim,
        },
        "trajectory_counts": counts,
        "sample_counts": sample_counts,
        "total_samples": sum(sample_counts.values()),
        "pixels_per_history": pixels_per_history,
        "approx_batch_input_mib": round(
            2 * config.run.batch_size * pixels_per_history * 4 / 2**20,
            3,
        ),
        "rollout_horizons": list(config.evaluation.rollout_horizons),
        "requested_device": config.run.device,
        "deterministic": config.run.deterministic,
        "fcsrl": (
            {
                "head_hidden_dim": config.objective.fcsrl_head_hidden_dim,
                "return_length": FCSRL_RETURN_LENGTH,
            }
            if config.objective.safety_arm == "fcsrl_feasibility_loss_adaptation"
            else None
        ),
    }


def cpu_smoke_config(config: LearningConfig) -> LearningConfig:
    """Return a tiny, scientifically faithful execution check for CPU-only environments.

    This is an engineering smoke test, not a paper-scale arm: it preserves rendered histories,
    trajectory-level four-way splitting, checkpoint selection, and the configured safety loss.
    """

    smoke_outputs = {
        "controlled_cart_video": "runs/e1_world_models/torch_cpu_smoke",
        "controlled_pendulum_video": "runs/e1_world_models/torch_pendulum_cpu_smoke",
        "controlled_dubins_navigation_pixels": (
            "runs/e1_world_models/torch_dubins_cpu_smoke"
        ),
    }
    smoke_output = smoke_outputs[config.data.task]
    smoke = dataclasses.replace(
        config,
        status="smoke_test_only",
        run=dataclasses.replace(
            config.run,
            output_dir=smoke_output,
            device="cpu",
            epochs=1,
            batch_size=8,
            num_workers=0,
            amp=False,
            log_every=1,
        ),
        data=dataclasses.replace(
            config.data,
            trajectories=16,
            horizon=10,
            image_size=16,
            history_length=2,
        ),
        model=dataclasses.replace(
            config.model,
            latent_dim=4,
            hidden_dim=32,
            transition_hidden_dim=48,
        ),
        evaluation=dataclasses.replace(
            config.evaluation,
            rollout_horizons=(1, 2),
            max_rollout_cases=8,
            max_audit_records_per_split=32,
        ),
    )
    validate_config(smoke)
    return smoke


def load_config(path: Path) -> LearningConfig:
    """Load a learning configuration from TOML."""

    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    return parse_config(payload)


def canonical_config_json(config: LearningConfig) -> str:
    """Stable serialization used in manifests and dataset identities."""

    return json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
