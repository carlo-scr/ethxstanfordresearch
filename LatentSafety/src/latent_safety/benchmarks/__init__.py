"""Dependency-free, analytically labelled latent-safety benchmarks.

The public API exposes a controlled double integrator, partial and history observations,
deterministic dataset records, and two exact observation-aliasing fixtures.  Every safety quantity
is explicitly current-step or one-step; this package does not silently promote them to invariance
or long-horizon viability claims.
"""

from .double_integrator import ControlledDoubleIntegrator, DoubleIntegratorState
from .dynamic_oracle import (
    MEMORYLESS_PIXELS,
    PRIVILEGED_STATE,
    REPRESENTATIONS,
    TWO_FRAME_HISTORY,
    ControlledDynamicOracleReport,
    ControlledDynamicTree,
    DynamicFixtureRoot,
    DynamicRootOutcome,
    DynamicRootFiberSafety,
    DynamicStageSummary,
    RepresentationDynamicSummary,
    audit_controlled_dynamic_oracle,
    build_controlled_dynamic_tree,
)
from .fixtures import (
    ObservationAliasingPair,
    make_action_conflict_pair,
    make_static_aliasing_pair,
)
from .observations import ObservationKind, ObservationSpec, reconstruct_current_velocity
from .rollouts import BenchmarkTransitionRecord, RolloutSpec, build_dataset, rollout

__all__ = [
    "BenchmarkTransitionRecord",
    "ControlledDoubleIntegrator",
    "ControlledDynamicOracleReport",
    "ControlledDynamicTree",
    "DoubleIntegratorState",
    "DynamicFixtureRoot",
    "DynamicRootOutcome",
    "DynamicRootFiberSafety",
    "DynamicStageSummary",
    "MEMORYLESS_PIXELS",
    "ObservationAliasingPair",
    "ObservationKind",
    "ObservationSpec",
    "PRIVILEGED_STATE",
    "REPRESENTATIONS",
    "RepresentationDynamicSummary",
    "RolloutSpec",
    "TWO_FRAME_HISTORY",
    "audit_controlled_dynamic_oracle",
    "build_dataset",
    "build_controlled_dynamic_tree",
    "make_action_conflict_pair",
    "make_static_aliasing_pair",
    "reconstruct_current_velocity",
    "rollout",
]
