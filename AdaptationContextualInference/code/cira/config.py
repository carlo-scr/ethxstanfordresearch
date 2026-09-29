"""All tunable settings in one place. Values mirror the placeholders in paper_corl where they exist."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    # Contexts: surface x load. Index 1/1 (rubber, 0.6 kg) is the training condition.
    surfaces: tuple = (("felt", 0.60), ("rubber", 0.35), ("acrylic", 0.12))
    loads: tuple = (0.2, 0.6, 1.0)
    train_context: tuple = (1, 1)
    # Geometry (m). Table top is z=0, x in [-half_l, edge_x], y in [-half_w, half_w].
    half_l: float = 0.6
    edge_x: float = 0.6
    half_w: float = 0.4
    puck_radius: float = 0.04
    puck_half_height: float = 0.012
    target: tuple = (0.35, 0.0)
    start_x: tuple = (-0.45, -0.35)
    start_y: tuple = (-0.10, 0.10)
    # Timing. One observation step = substeps physics steps; the impulse acts in the first impulse_steps.
    timestep: float = 0.002
    substeps: int = 20
    impulse_steps: int = 5
    max_steps: int = 50
    stop_speed: float = 0.02
    # Action = planar impulse (N s).
    action_low: tuple = (0.0, -0.6)
    action_high: tuple = (3.5, 0.6)
    # Nuisance variation, drawn per episode from a seed shared by all methods.
    friction_jitter: float = 0.03
    impulse_noise: float = 0.01
    impulse_angle_noise_deg: float = 1.0
    obs_pos_noise: float = 0.001
    obs_vel_noise: float = 0.01
    # Stick-slip: transitions starting or ending below this speed break the linear residual model,
    # so their noise scale is inflated (paper Sec. 3.1: sigma is large during stick-slip).
    slow_speed: float = 0.3
    slow_sigma_factor: float = 10.0
    success_tol: float = 0.05

    @property
    def dt(self) -> float:
        return self.timestep * self.substeps


@dataclass(frozen=True)
class StreamConfig:
    n_episodes: int = 500
    mean_dwell: float = 8.0
    rho: float = 0.8
    cue_dim_per_factor: int = 4
    cue_noise: float = 0.3
    include_train_context: bool = True


@dataclass(frozen=True)
class CiraConfig:
    use_cue: bool = True
    weights: str = "predictive"  # predictive | filtered | hard
    # Score hypothesis c with c's memory conditioned on all earlier steps of this episode at full
    # weight (exact when the context is fixed within an episode). Library updates still use `weights`.
    episode_conditional: bool = True
    # Within-context variation between episodes, as extra weight variance of the per-episode scorers.
    # Without it the new-context hypothesis wins by fitting episode-specific friction jitter that no
    # memory can represent.
    episode_var: float = 0.003  # on the passive (friction) weights
    episode_var_action: float = 1e-4  # on the action-driven weights (impulse execution noise)
    prior: str = "structured"  # structured (see frozen.structured_prior_var) | isotropic
    prior_var: float = 1.0  # variance of the free weights (all weights if isotropic)
    prior_var_cross: float = 0.01  # variance of the remaining weights under the structured prior
    crp_alpha: float = 0.5  # mass reserved for a new context
    stay_prob: float = 0.8  # sticky prior: probability that the previous episode's context persists
    train_pseudocount: float = 1.0  # prior count of the fixed training-condition memory
    create_after: int = 5  # steps the new-context hypothesis must stay most probable
    novel_reset_mass: float = 1e-3
    cue_pseudocount: float = 1.0
    cue_var_floor: float = 0.02
    cue_temperature: float = 1.0
    merge_z2: float = 9.0  # merge two memories if they never differ by more than 3 sd (0 disables)
    merge_buffer: int = 256  # recent feature vectors on which memories are compared
    prune: float = 0.01  # hypotheses below this probability are not rolled out


@dataclass(frozen=True)
class PlannerConfig:
    samples: int = 256
    elites: int = 24
    iters: int = 4
    horizon: int = 50
    off_table_cost: float = 0.6
    max_hypotheses: int = 6  # only the most probable hypotheses are rolled out
    init_std_frac: float = 0.3


@dataclass(frozen=True)
class FrozenConfig:
    n_episodes: int = 20000
    hidden: int = 256
    epochs: int = 150
    batch: int = 512
    lr: float = 1e-3
    sigma_floor: float = 0.02  # in normalized residual units
    seed: int = 0


@dataclass(frozen=True)
class SGDConfig:
    lr: float = 0.3  # normalized-LMS step size in (0, 1]
    steps: int = 1


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    cira: CiraConfig = field(default_factory=CiraConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    sgd: SGDConfig = field(default_factory=SGDConfig)
