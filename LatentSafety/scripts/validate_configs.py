#!/usr/bin/env python3
"""Parse experiment TOML and enforce provenance and frozen-grid invariants."""

from __future__ import annotations

import math
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.preconfirmation import (  # noqa: E402
    frozen_e2_eligibility_contract,
    frozen_e2_fail_closed_contract,
    frozen_e2_matched_radius_contract,
    frozen_e2_utility_selection_contract,
)

CONFIRMATORY_EXPERIMENT = "e2_confirmatory_core_192"
PRECONFIRMATION_EXPERIMENT = "e2_safety_utility_frontier"
PILOT_SEEDS = {0, 1, 2}
CONFIRMATORY_DOMAINS = (
    "controlled_cart_video",
    "controlled_pendulum_video",
    "controlled_dubins_navigation_pixels",
)
CONFIRMATORY_FAMILIES = ("ae", "beta_vae")
CONFIRMATORY_HISTORY_MODES = ("stack_h4",)
CONFIRMATORY_ARMS = (
    "none",
    "h_prediction",
    "nonprivileged_predicted_action_profile",
    "fcsrl_feasibility_loss_adaptation",
)
CONFIRMATORY_SEEDS = tuple(range(100, 108))
THIRD_DOMAIN_STATUS = (
    "implemented and CPU-smoked; GPU-scale preconfirmation and coverage-validation "
    "remain gated"
)
PREDICTED_PROFILE_IMPLEMENTATION_STATUS = (
    "teacher training, five-shard assembly, observed-rollout coverage, downstream "
    "ingestion, and post-fit validation audit implemented and CPU-smoked; production "
    "gates unexecuted; confirmation blocked"
)
FCSRL_IMPLEMENTATION_STATUS = (
    "EMA encoder, deterministic sequence loader, main-trainer checkpoint/resume, "
    "regression artifact, and post-fit validation audit implemented and CPU-smoked; "
    "production preconfirmation unexecuted; confirmation blocked"
)
REQUIRED_LEARNED_CORE_ARMS = frozenset(CONFIRMATORY_ARMS) - {"none"}
PRECONFIRMATION_ARM_GROUPS = {
    "required_core_learned_arms": {
        "names": [
            "h_prediction",
            "nonprivileged_predicted_action_profile",
            "fcsrl_feasibility_loss_adaptation",
        ],
        "gates_confirmation": True,
        "positive_weight_freeze": True,
    },
    "optional_same_backbone_sensitivities": {
        "names": ["pairwise_safety_faithfulness", "boundary_contrastive"],
        "gates_confirmation": False,
        "positive_weight_freeze": False,
    },
    "architecture_changing_sensitivities": {
        "names": ["safety_bisimulation_cvrl_bm"],
        "gates_confirmation": False,
        "positive_weight_freeze": False,
    },
    "end_to_end_e5_agents": {
        "names": ["srpl_steps_to_cost", "sdqc_reward_cost_decoupling"],
        "gates_confirmation": False,
        "positive_weight_freeze": False,
    },
    "unweighted_oracle_controls": {
        "names": ["append_known_safety_value_oracle"],
        "regularization": "not_applicable",
        "gates_confirmation": False,
        "positive_weight_freeze": False,
    },
}
ELIGIBILITY_GATE = frozen_e2_eligibility_contract()
FAIL_CLOSED_GATE = frozen_e2_fail_closed_contract()
UTILITY_SELECTION_GATE = frozen_e2_utility_selection_contract()
PRECONFIRMATION_SELECTION_ARTIFACT = {
    "entrypoint": "scripts/select_preconfirmation_weights.py",
    "aggregator_entrypoint": "scripts/aggregate_e2_preconfirmation.py",
    "aggregator_protocol": "e2_preconfirmation_validation_observations_v1",
    "task_handoff_protocol": "e2_preconfirmation_task_handoff_v1",
    "input_split": "validation_only",
    "expected_observations": 288,
    "required_radius_sidecars": 288,
    "observation_formula": (
        "18 none rows plus 3 learned arms x 5 positive weights x 3 domains x 2 "
        "families x 3 pilot seeds"
    ),
    "required_weight_freezes": 18,
    "freeze_formula": "3 learned arms x 3 domains x 2 model families",
    "selected_radius_freezes_required": True,
    "selected_radius_freeze_scope": (
        "every selected weight retains one complete matched-radius and audit-hash "
        "freeze for each of its three paired pilot seeds"
    ),
    "output": "runs/e2_frontier/preconfirmation_selected_weights.json",
    "final_test_access": False,
    "matched_mass_sign_scope": (
        "strict arm safety below paired none separately in every pilot seed"
    ),
    "profile_label_manifest_reuse": (
        "one checksum-identical five-shard manifest across all five weights in each "
        "domain-family-data-seed cell"
    ),
}
MATCHED_RADIUS_REDUCTION = frozen_e2_matched_radius_contract()
PREDICTED_PROFILE_TEACHER = {
    "fold_count": 5,
    "fold_assignment": (
        "sort training trajectory IDs; shuffle with Python random.Random(20260822); "
        "assign shuffled index modulo 5"
    ),
    "training_scope": "teacher k trains only on training folds other than k",
    "architecture": (
        "match downstream family; LatentWorldModel; stack_h4; latent_dim=8; "
        "hidden_dim=128; transition_hidden_dim=192"
    ),
    "objective": (
        "h_prediction with safety_weight=0.5; otherwise inherit frozen domain base config"
    ),
    "teacher_seed_rule": "10000 + 10 * data_seed + fold_index",
    "epochs": 30,
    "checkpoint_rule": (
        "earliest epoch minimizing ordinary-validation world_model_utility within family"
    ),
    "inference": "eval mode; posterior mean; deterministic latent transition rollout",
    "label_rule": (
        "held-out fold only; minimum predicted margin over t=0..H for each "
        "registered constant action"
    ),
    "privileged_training_inputs": False,
}
PREDICTED_PROFILE_VALIDATION = {
    "split": (
        "coverage_validation; disjoint from train, ordinary validation, calibration, "
        "and final test"
    ),
    "generator": "coverage_validation_observed_rollouts_v1",
    "coverage_seed_rule": (
        "70000000 + domain_offset{cart:0,pendulum:100000,dubins:200000} + data_seed"
    ),
    "bundle_namespace": "coverage-{domain}-seed-{data_seed}-bundle-{index}",
    "history_length": 4,
    "center_timestep": 3,
    "branch_process_noise": 0.0,
    "paired_initial_history_bundles_per_domain_seed": 200,
    "reuse_bundle_ids_across_model_families": True,
    "scenario_mix": "frozen deployment-generator mix",
    "branching": (
        "clone every initial history and execute every registered action constantly "
        "for the domain horizon"
    ),
    "teacher_assignment": (
        "sorted bundle index modulo 5; corresponding fold teacher; no ensemble"
    ),
    "target": "minimum observed h over t=0..H",
    "normalization": (
        "divide bundle-wise maximum absolute action-profile error by frozen domain "
        "margin_scale"
    ),
    "quantile": "nearest-rank empirical p95: sorted_error[ceil(0.95*N)-1]",
    "required_max": 0.10,
    "gate_scope": "every domain-family-data-seed cell",
    "preconfirmation_seeds": [0, 1, 2],
    "confirmation_seeds": list(CONFIRMATORY_SEEDS),
    "teachers_required_per_cell": 5,
    "all_cells_and_teachers_required": True,
    "on_any_confirmation_teacher_failure": (
        "cancel confirmation before final-test access"
    ),
    "oracle_use": "controlled-domain true-dynamics profiles are diagnostic only",
}
PREDICTED_PROFILE_GATE = {
    "gate_scope": "every domain-family-data-seed cell",
    "preconfirmation_seeds": [0, 1, 2],
    "confirmation_seeds": list(CONFIRMATORY_SEEDS),
    "teachers_required_per_cell": 5,
    "all_cells_and_teachers_required": True,
    "on_any_confirmation_teacher_failure": (
        "cancel confirmation before final-test access"
    ),
}
INFERENCE_CONTRACT = {
    "proposed_arm": "nonprivileged_predicted_action_profile",
    "comparators": [
        "none",
        "h_prediction",
        "fcsrl_feasibility_loss_adaptation",
        "append_true_h_none_view",
    ],
    "domains": list(CONFIRMATORY_DOMAINS),
    "endpoints": ["safety", "reconstruction", "rollout"],
    "elementary_bound_count": 36,
    "elementary_bound_formula": "4 comparators x 3 domains x 3 endpoints",
    "model_family_aggregation": (
        "within each seed and domain, average paired AE and beta-VAE "
        "differences equally"
    ),
    "paired_seeds": list(CONFIRMATORY_SEEDS),
    "paired_seed_count": 8,
    "bootstrap_resamples": 100000,
    "bootstrap_seed": 20260822,
    "bootstrap_unit": (
        "paired seed after equal AE/beta-VAE averaging within seed and domain"
    ),
    "confidence_sidedness": "one-sided upper",
    "family_alpha": 0.05,
    "simultaneous_ucb_method": "Bonferroni over all 36 elementary bounds",
    "primary_ucb_method": "non-studentized percentile upper quantile",
    "primary_ucb_probability_formula": "1 - 0.05 / 36",
    "primary_ucb_quantile": 0.9986111111111111,
    "studentized": False,
    "bca": False,
    "max_t": False,
    "all_planned_runs_must_be_complete": True,
    "all_elementary_bounds_must_pass": True,
    "safety": {
        "difference": "nonprivileged_predicted_action_profile - comparator",
        "contrast_count": 12,
        "point_mean_max": -0.10,
        "point_mean_operator": "less_than_or_equal",
        "ucb_max": 0.0,
        "ucb_operator": "strictly_less_than",
    },
    "utility": {
        "relative_degradation": (
            "(nonprivileged_predicted_action_profile - comparator) / comparator"
        ),
        "endpoints": ["reconstruction", "rollout"],
        "contrast_count": 24,
        "ucb_max": 0.05,
        "ucb_operator": "less_than_or_equal",
        "zero_denominator_rule": (
            "only zero profile error is admissible; define tied zero relative "
            "degradation as 0"
        ),
    },
    "secondary_sign_flip": {
        "decision_role": "optional secondary only",
        "contrast_scope": "12 safety contrasts",
        "enabled_only_if": (
            "paired method labels are exchangeable under a predeclared sharp null"
        ),
        "correction": "Holm",
        "family_alpha": 0.05,
    },
}


def _prefix(path: Path, message: str) -> str:
    return f"{path.as_posix()}: {message}"


def _table(
    config: dict[str, Any], field: str, path: Path, errors: list[str]
) -> dict[str, Any]:
    value = config.get(field)
    if not isinstance(value, dict):
        errors.append(_prefix(path, f"{field} must be a table"))
        return {}
    return value


def _list_value(
    table: dict[str, Any], field: str, path: Path, errors: list[str]
) -> list[Any]:
    value = table.get(field)
    if not isinstance(value, list):
        errors.append(_prefix(path, f"{field} must be a list"))
        return []
    return value


def _validate_relative_file(
    root: Path, path: Path, value: object, field: str, errors: list[str]
) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(_prefix(path, f"{field} must name a repository-relative file"))
        return
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(_prefix(path, f"{field} must be repository-relative"))
    elif not (root / relative).is_file():
        errors.append(_prefix(path, f"{field} does not exist: {value!r}"))


def validate_selection_gates(path: Path, config: dict[str, Any]) -> list[str]:
    """Require the frozen coverage, matched-mass, and fail-closed gates."""

    errors: list[str] = []
    eligibility = _table(config, "eligibility", path, errors)
    if eligibility != ELIGIBILITY_GATE:
        errors.append(
            _prefix(path, f"eligibility gate must be exactly {ELIGIBILITY_GATE!r}")
        )
    fail_closed = _table(config, "fail_closed", path, errors)
    if fail_closed != FAIL_CLOSED_GATE:
        errors.append(
            _prefix(path, f"fail_closed gate must be exactly {FAIL_CLOSED_GATE!r}")
        )
    selection_field = (
        "analysis"
        if config.get("experiment") == CONFIRMATORY_EXPERIMENT
        else "selection"
    )
    selection = _table(config, selection_field, path, errors)
    for field, expected in UTILITY_SELECTION_GATE.items():
        if selection.get(field) != expected:
            errors.append(
                _prefix(path, f"{selection_field}.{field} must be {expected!r}")
            )
    return errors


def validate_predicted_profile_gate(
    path: Path, table: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if table != PREDICTED_PROFILE_GATE:
        errors.append(
            _prefix(
                path,
                f"predicted-profile gate must be exactly {PREDICTED_PROFILE_GATE!r}",
            )
        )
    return errors


def validate_confirmatory_core(
    root: Path, path: Path, config: dict[str, Any]
) -> list[str]:
    """Validate the exact 192-job core and prohibit hidden tuning axes."""

    errors: list[str] = []
    if config.get("status") != "frozen_design_blocked_on_production_preconfirmation_gates":
        errors.append(_prefix(path, "unexpected confirmatory status"))
    axes = _table(config, "axes", path, errors)
    expected_axes = {"domains", "model_families", "history_modes", "arms", "seeds"}
    if set(axes) != expected_axes:
        errors.append(
            _prefix(
                path,
                f"confirmatory axes must be exactly {sorted(expected_axes)!r}",
            )
        )

    expected_values = {
        "domains": CONFIRMATORY_DOMAINS,
        "model_families": CONFIRMATORY_FAMILIES,
        "history_modes": CONFIRMATORY_HISTORY_MODES,
        "arms": CONFIRMATORY_ARMS,
        "seeds": CONFIRMATORY_SEEDS,
    }
    observed: dict[str, list[Any]] = {}
    for field, expected in expected_values.items():
        observed[field] = _list_value(axes, field, path, errors)
        if tuple(observed[field]) != expected:
            errors.append(
                _prefix(path, f"{field} must be exactly {list(expected)!r}")
            )

    seeds = observed.get("seeds", [])
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        errors.append(_prefix(path, "confirmatory seeds must be integers"))
    if len(set(seeds)) != len(seeds):
        errors.append(_prefix(path, "confirmatory seeds must be unique"))
    if PILOT_SEEDS.intersection(seeds):
        errors.append(_prefix(path, "confirmatory seeds overlap pilot seeds 0, 1, 2"))

    task_count = math.prod(len(observed.get(field, [])) for field in expected_axes)
    execution = _table(config, "execution", path, errors)
    if task_count != 192 or execution.get("expected_tasks") != 192:
        errors.append(
            _prefix(
                path,
                f"confirmatory core must expand to 192 tasks, got {task_count}",
            )
        )
    if execution.get("third_domain_status") != THIRD_DOMAIN_STATUS:
        errors.append(
            _prefix(
                path,
                f"third_domain_status must be exactly {THIRD_DOMAIN_STATUS!r}",
            )
        )

    freeze = _table(config, "loss_weight_freeze", path, errors)
    if freeze.get("weights_are_task_axis") is not False:
        errors.append(_prefix(path, "loss weights must not be a confirmatory task axis"))
    if freeze.get("control_arm") != "none":
        errors.append(_prefix(path, "loss-weight control arm must be none"))
    if freeze.get("control_regularization") != "not_applicable":
        errors.append(_prefix(path, "none must use not_applicable regularization"))
    tuning_path = freeze.get("tuning_config")
    if not isinstance(tuning_path, str) or not tuning_path.strip():
        errors.append(_prefix(path, "missing preconfirmation tuning_config"))
    elif Path(tuning_path).is_absolute() or ".." in Path(tuning_path).parts:
        errors.append(_prefix(path, "tuning_config must be repository-relative"))
    elif not (root / tuning_path).is_file():
        errors.append(_prefix(path, f"tuning_config does not exist: {tuning_path!r}"))

    arm_semantics = _table(config, "arm_semantics", path, errors)
    if set(arm_semantics) != set(CONFIRMATORY_ARMS):
        errors.append(_prefix(path, "arm_semantics must cover exactly the four core arms"))
    predicted = arm_semantics.get("nonprivileged_predicted_action_profile", {})
    if not isinstance(predicted, dict) or predicted.get(
        "uses_privileged_state_or_known_dynamics_for_training"
    ) is not False:
        errors.append(
            _prefix(path, "predicted-profile core arm must be explicitly nonprivileged")
        )
    if isinstance(predicted, dict):
        _validate_relative_file(
            root,
            path,
            predicted.get("specification"),
            "arm_semantics.nonprivileged_predicted_action_profile.specification",
            errors,
        )
    none = arm_semantics.get("none", {})
    if not isinstance(none, dict) or none.get("regularization") != "not_applicable":
        errors.append(_prefix(path, "none arm semantics must reject regularization"))
    implementation_statuses = {
        "nonprivileged_predicted_action_profile": (
            PREDICTED_PROFILE_IMPLEMENTATION_STATUS
        ),
        "fcsrl_feasibility_loss_adaptation": FCSRL_IMPLEMENTATION_STATUS,
    }
    for arm, expected_status in implementation_statuses.items():
        semantics = arm_semantics.get(arm, {})
        if (
            not isinstance(semantics, dict)
            or semantics.get("implementation_status") != expected_status
        ):
            errors.append(
                _prefix(
                    path,
                    f"{arm}.implementation_status must be exactly {expected_status!r}",
                )
            )
    fcsrl = arm_semantics.get("fcsrl_feasibility_loss_adaptation", {})
    if not isinstance(fcsrl, dict) or fcsrl.get("categorical_head_hidden_dim") != 64:
        errors.append(
            _prefix(
                path,
                "fcsrl_feasibility_loss_adaptation.categorical_head_hidden_dim "
                "must be exactly 64",
            )
        )
    errors.extend(validate_selection_gates(path, config))
    teacher = _table(config, "predicted_profile_teacher", path, errors)
    if teacher != PREDICTED_PROFILE_TEACHER:
        errors.append(
            _prefix(
                path,
                f"predicted_profile_teacher must be exactly {PREDICTED_PROFILE_TEACHER!r}",
            )
        )
    validation = _table(config, "predicted_profile_validation", path, errors)
    if validation != PREDICTED_PROFILE_VALIDATION:
        errors.append(
            _prefix(
                path,
                "predicted_profile_validation must match the frozen protocol",
            )
        )
    inference = _table(config, "inference", path, errors)
    if inference != INFERENCE_CONTRACT:
        errors.append(
            _prefix(path, "inference contract must match the frozen 36-bound protocol")
        )
    return errors


def validate_preconfirmation_grid(
    path: Path, config: dict[str, Any], *, root: Path | None = None
) -> list[str]:
    """Keep positive learned-arm tuning separate from the unweighted control."""

    errors: list[str] = []
    repository_root = root or Path(__file__).resolve().parents[1]
    if config.get("status") != "planned_preconfirmation_tuning_only":
        errors.append(_prefix(path, "intervention grid must be preconfirmation-only"))
    run = _table(config, "run", path, errors)
    if run.get("seeds") != [0, 1, 2]:
        errors.append(_prefix(path, "preconfirmation tuning must use pilot seeds 0, 1, 2"))
    controls = _table(config, "controls", path, errors)
    if controls.get("names") != ["none"]:
        errors.append(_prefix(path, "preconfirmation controls must contain only none"))
    if controls.get("regularization") != "not_applicable":
        errors.append(_prefix(path, "none control regularization must be not_applicable"))
    categorized_names: list[str] = []
    for field, expected in PRECONFIRMATION_ARM_GROUPS.items():
        observed = _table(config, field, path, errors)
        if observed != expected:
            errors.append(_prefix(path, f"{field} must be exactly {expected!r}"))
        names = observed.get("names", [])
        if isinstance(names, list):
            categorized_names.extend(str(name) for name in names)
    if "none" in categorized_names:
        errors.append(_prefix(path, "none cannot appear in an intervention category"))
    if len(set(categorized_names)) != len(categorized_names):
        errors.append(_prefix(path, "intervention categories must be disjoint"))
    regularization = _table(config, "regularization", path, errors)
    if "weights" in regularization:
        errors.append(_prefix(path, "legacy weights field is forbidden; use positive_weights"))
    weights = _list_value(regularization, "positive_weights", path, errors)
    if not weights or any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0.0
        for weight in weights
    ):
        errors.append(_prefix(path, "learned-arm tuning weights must be finite and positive"))
    if len(set(weights)) != len(weights):
        errors.append(_prefix(path, "learned-arm tuning weights must be unique"))
    if regularization.get("scope") != (
        "required_core_learned_arms only; every other category is outside the "
        "confirmation weight freeze"
    ):
        errors.append(
            _prefix(path, "positive weight freeze must apply only to required core arms")
        )
    predicted = _table(config, "predicted_profile", path, errors)
    _validate_relative_file(
        repository_root,
        path,
        predicted.get("specification"),
        "predicted_profile.specification",
        errors,
    )
    if predicted.get("implementation_status") != PREDICTED_PROFILE_IMPLEMENTATION_STATUS:
        errors.append(
            _prefix(
                path,
                "predicted_profile.implementation_status must match the frozen "
                "component-level blocker",
            )
        )
    profile_gate = _table(config, "predicted_profile_gate", path, errors)
    errors.extend(validate_predicted_profile_gate(path, profile_gate))
    selection_artifact = _table(config, "selection_artifact", path, errors)
    if selection_artifact != PRECONFIRMATION_SELECTION_ARTIFACT:
        errors.append(
            _prefix(
                path,
                "selection_artifact must match the exact 288-row validation-only "
                "weight-freeze contract",
            )
        )
    else:
        _validate_relative_file(
            repository_root,
            path,
            selection_artifact.get("entrypoint"),
            "selection_artifact.entrypoint",
            errors,
        )
        _validate_relative_file(
            repository_root,
            path,
            selection_artifact.get("aggregator_entrypoint"),
            "selection_artifact.aggregator_entrypoint",
            errors,
        )
    matched_radius = _table(config, "matched_radius_reduction", path, errors)
    if matched_radius != MATCHED_RADIUS_REDUCTION:
        errors.append(
            _prefix(
                path,
                "matched_radius_reduction must match the exact validation-only "
                "normalized-mass contract",
            )
        )
    else:
        _validate_relative_file(
            repository_root,
            path,
            matched_radius.get("entrypoint"),
            "matched_radius_reduction.entrypoint",
            errors,
        )
    errors.extend(validate_selection_gates(path, config))
    return errors


def validate_configs(root: Path) -> tuple[list[str], int]:
    paths = sorted((root / "configs").rglob("*.toml"))
    errors: list[str] = []
    seen_experiments: set[str] = set()
    for absolute_path in paths:
        path = absolute_path.relative_to(root)
        try:
            with absolute_path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            errors.append(_prefix(path, str(error)))
            continue
        for key in ("schema_version", "experiment"):
            if key not in config:
                errors.append(_prefix(path, f"missing {key!r}"))
        experiment = config.get("experiment")
        if experiment in seen_experiments:
            errors.append(_prefix(path, f"duplicate experiment {experiment!r}"))
        if isinstance(experiment, str):
            seen_experiments.add(experiment)
        if experiment == CONFIRMATORY_EXPERIMENT:
            errors.extend(validate_confirmatory_core(root, path, config))
        elif experiment == PRECONFIRMATION_EXPERIMENT:
            errors.extend(validate_preconfirmation_grid(path, config, root=root))

    if CONFIRMATORY_EXPERIMENT not in seen_experiments:
        errors.append("missing exact E2 confirmatory core config")
    if PRECONFIRMATION_EXPERIMENT not in seen_experiments:
        errors.append("missing E2 preconfirmation tuning config")
    return errors, len(paths)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors, count = validate_configs(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {count} experiment configs, including the exact 192-task E2 core")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
