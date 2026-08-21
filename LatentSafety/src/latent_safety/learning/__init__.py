"""Optional learning stack for representation and safety-sufficiency experiments.

The public configuration surface has no third-party dependencies. PyTorch is imported lazily only
when an actual training run starts, so ``--dry-run`` remains usable in the minimal environment.
"""

from latent_safety.learning.config import (
    ConfigError,
    LearningConfig,
    cpu_smoke_config,
    dry_run_plan,
    load_config,
    parse_config,
)
from latent_safety.learning.data import (
    CartState,
    PendulumState,
    action_safety_profile,
    build_dataset_manifest,
    build_datasets,
    deterministic_step,
    generate_trajectories,
    observation_feature_vector,
    render_state,
    safety_margin,
    state_feature_vector,
)
from latent_safety.learning.models import build_world_model, load_world_model_checkpoint
from latent_safety.learning.runtime import require_torch

__all__ = [
    "ConfigError",
    "CartState",
    "LearningConfig",
    "PendulumState",
    "action_safety_profile",
    "build_dataset_manifest",
    "build_datasets",
    "build_world_model",
    "cpu_smoke_config",
    "dry_run_plan",
    "deterministic_step",
    "generate_trajectories",
    "load_config",
    "load_world_model_checkpoint",
    "observation_feature_vector",
    "parse_config",
    "render_state",
    "require_torch",
    "safety_margin",
    "state_feature_vector",
]
