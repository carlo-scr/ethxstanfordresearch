"""Dependency-free statistical tools for the LatentSafety experiment protocol.

The public API deliberately names the resampling and selection units.  In particular,
``PairedBlock`` must contain one aggregate per independent seed or predeclared block; individual
video frames are not valid inferential units.
"""

from latent_safety.analysis.stats import (
    BLOCK_INDEPENDENCE_CAVEAT,
    VALIDATION_SELECTION_CAVEAT,
    ExactPairedSignFlipTest,
    HolmAdjustedPValue,
    HypothesisPValue,
    PairedBlock,
    PairedBootstrapCI,
    StandardizedPairedEffect,
    ValidationParetoFrontier,
    ValidationScore,
    exact_paired_sign_flip_test,
    holm_adjust,
    paired_block_bootstrap_ci,
    select_under_safety_budget,
    standardized_paired_effect,
    validation_pareto_frontier,
)

__all__ = [
    "BLOCK_INDEPENDENCE_CAVEAT",
    "VALIDATION_SELECTION_CAVEAT",
    "ExactPairedSignFlipTest",
    "HolmAdjustedPValue",
    "HypothesisPValue",
    "PairedBlock",
    "PairedBootstrapCI",
    "StandardizedPairedEffect",
    "ValidationParetoFrontier",
    "ValidationScore",
    "exact_paired_sign_flip_test",
    "holm_adjust",
    "paired_block_bootstrap_ci",
    "select_under_safety_budget",
    "standardized_paired_effect",
    "validation_pareto_frontier",
]
