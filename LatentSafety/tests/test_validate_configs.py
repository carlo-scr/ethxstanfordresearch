from __future__ import annotations

import copy
import importlib.util
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_configs_module", ROOT / "scripts" / "validate_configs.py"
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("could not load scripts/validate_configs.py")
VALIDATE_CONFIGS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE_CONFIGS)


def _load(relative: str) -> dict[str, object]:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


class ConfigValidatorTests(unittest.TestCase):
    def test_repository_configs_validate_with_exact_confirmatory_core(self) -> None:
        errors, count = VALIDATE_CONFIGS.validate_configs(ROOT)

        self.assertEqual(errors, [])
        self.assertEqual(count, 12)

    def test_confirmatory_core_rejects_pilot_seed_and_hidden_weight_axis(self) -> None:
        relative = Path("configs/e2_frontier/confirmatory_core.toml")
        config = _load(relative.as_posix())
        self.assertEqual(
            VALIDATE_CONFIGS.validate_confirmatory_core(ROOT, relative, config),
            [],
        )

        pilot_overlap = copy.deepcopy(config)
        pilot_overlap["axes"]["seeds"][0] = 0
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(
            ROOT, relative, pilot_overlap
        )
        self.assertTrue(any("overlap pilot seeds" in error for error in errors))

        hidden_weight_axis = copy.deepcopy(config)
        hidden_weight_axis["axes"]["regularization_weights"] = [0.0, 0.1]
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(
            ROOT, relative, hidden_weight_axis
        )
        self.assertTrue(
            any("confirmatory axes must be exactly" in error for error in errors)
        )

        stale_domain_status = copy.deepcopy(config)
        stale_domain_status["execution"]["third_domain_status"] = "not implemented"
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(
            ROOT, relative, stale_domain_status
        )
        self.assertTrue(any("third_domain_status must be exactly" in error for error in errors))

        wrong_fcsrl_width = copy.deepcopy(config)
        wrong_fcsrl_width["arm_semantics"]["fcsrl_feasibility_loss_adaptation"][
            "categorical_head_hidden_dim"
        ] = 32
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(
            ROOT, relative, wrong_fcsrl_width
        )
        self.assertTrue(any("categorical_head_hidden_dim" in error for error in errors))

    def test_confirmatory_core_requires_exact_four_arms_and_nonprivileged_profile(self) -> None:
        relative = Path("configs/e2_frontier/confirmatory_core.toml")
        config = _load(relative.as_posix())
        wrong_arm = copy.deepcopy(config)
        wrong_arm["axes"]["arms"][2] = "privileged_true_action_profile"
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(ROOT, relative, wrong_arm)
        self.assertTrue(any("arms must be exactly" in error for error in errors))

        privileged = copy.deepcopy(config)
        privileged["arm_semantics"]["nonprivileged_predicted_action_profile"][
            "uses_privileged_state_or_known_dynamics_for_training"
        ] = True
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(ROOT, relative, privileged)
        self.assertTrue(
            any("must be explicitly nonprivileged" in error for error in errors)
        )

        changed_teacher = copy.deepcopy(config)
        changed_teacher["predicted_profile_teacher"]["fold_count"] = 4
        errors = VALIDATE_CONFIGS.validate_confirmatory_core(
            ROOT, relative, changed_teacher
        )
        self.assertTrue(
            any("predicted_profile_teacher must be exactly" in error for error in errors)
        )

    def test_preconfirmation_grid_rejects_zero_weight_and_weighted_none(self) -> None:
        relative = Path("configs/e2_frontier/intervention.toml")
        config = _load(relative.as_posix())
        self.assertEqual(
            VALIDATE_CONFIGS.validate_preconfirmation_grid(relative, config), []
        )

        zero_weight = copy.deepcopy(config)
        zero_weight["regularization"]["positive_weights"].insert(0, 0.0)
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(relative, zero_weight)
        self.assertTrue(any("finite and positive" in error for error in errors))

        weighted_none = copy.deepcopy(config)
        weighted_none["required_core_learned_arms"]["names"].append("none")
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(relative, weighted_none)
        self.assertTrue(
            any("none cannot appear in an intervention category" in error for error in errors)
        )

        missing_core_arm = copy.deepcopy(config)
        missing_core_arm["required_core_learned_arms"]["names"].remove(
            "h_prediction"
        )
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(
            relative, missing_core_arm
        )
        self.assertTrue(
            any("required_core_learned_arms must be exactly" in error for error in errors)
        )

        sensitivity_gates = copy.deepcopy(config)
        sensitivity_gates["optional_same_backbone_sensitivities"][
            "gates_confirmation"
        ] = True
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(
            relative, sensitivity_gates
        )
        self.assertTrue(
            any(
                "optional_same_backbone_sensitivities must be exactly" in error
                for error in errors
            )
        )

        drifted_selector = copy.deepcopy(config)
        drifted_selector["selection_artifact"]["expected_observations"] = 240
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(
            relative, drifted_selector
        )
        self.assertTrue(
            any("exact 288-row validation-only" in error for error in errors)
        )

        aggregator_drift = copy.deepcopy(config)
        aggregator_drift["selection_artifact"]["required_radius_sidecars"] = 0
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(
            relative, aggregator_drift
        )
        self.assertTrue(
            any("exact 288-row validation-only" in error for error in errors)
        )

        radius_drift = copy.deepcopy(config)
        radius_drift["matched_radius_reduction"][
            "control_reference_relative_radius"
        ] = 0.10
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(
            relative, radius_drift
        )
        self.assertTrue(
            any("exact validation-only normalized-mass" in error for error in errors)
        )

    def test_selection_gates_reject_coverage_or_fail_closed_drift(self) -> None:
        relative = Path("configs/e2_frontier/confirmatory_core.toml")
        config = _load(relative.as_posix())

        low_coverage = copy.deepcopy(config)
        low_coverage["eligibility"]["minimum_eligible_center_coverage"] = 0.50
        errors = VALIDATE_CONFIGS.validate_selection_gates(relative, low_coverage)
        self.assertTrue(any("eligibility gate must be exactly" in error for error in errors))

        substitution = copy.deepcopy(config)
        substitution["fail_closed"]["post_hoc_arm_substitution_allowed"] = True
        errors = VALIDATE_CONFIGS.validate_selection_gates(relative, substitution)
        self.assertTrue(any("fail_closed gate must be exactly" in error for error in errors))

        vague_utility = copy.deepcopy(config)
        vague_utility["analysis"]["tie_break"] = "lower utility loss"
        errors = VALIDATE_CONFIGS.validate_selection_gates(relative, vague_utility)
        self.assertTrue(any("analysis.tie_break must be" in error for error in errors))

    def test_predicted_profile_gate_covers_both_seed_phases_and_every_teacher(self) -> None:
        relative = Path("configs/e2_frontier/intervention.toml")
        config = _load(relative.as_posix())

        missing_confirm_seed = copy.deepcopy(config)
        missing_confirm_seed["predicted_profile_gate"]["confirmation_seeds"].pop()
        errors = VALIDATE_CONFIGS.validate_preconfirmation_grid(
            relative, missing_confirm_seed
        )
        self.assertTrue(any("predicted-profile gate must be exactly" in error for error in errors))

        permissive_teacher_failure = copy.deepcopy(config)
        permissive_teacher_failure["fail_closed"][
            "any_confirmation_teacher_failure"
        ] = "continue with remaining teachers"
        errors = VALIDATE_CONFIGS.validate_selection_gates(
            relative, permissive_teacher_failure
        )
        self.assertTrue(any("fail_closed gate must be exactly" in error for error in errors))

    def test_inference_contract_rejects_multiplicity_or_decision_drift(self) -> None:
        relative = Path("configs/e2_frontier/confirmatory_core.toml")
        config = _load(relative.as_posix())
        self.assertEqual(
            VALIDATE_CONFIGS.validate_confirmatory_core(ROOT, relative, config), []
        )

        mutations = (
            (
                "bootstrap",
                lambda value: value["inference"].__setitem__(
                    "bootstrap_resamples", 10000
                ),
            ),
            (
                "bootstrap_unit",
                lambda value: value["inference"].__setitem__(
                    "bootstrap_unit", "individual run"
                ),
            ),
            (
                "ucb_method",
                lambda value: value["inference"].__setitem__(
                    "primary_ucb_method", "BCa"
                ),
            ),
            (
                "ucb_quantile",
                lambda value: value["inference"].__setitem__(
                    "primary_ucb_quantile", 0.95
                ),
            ),
            ("bounds", lambda value: value["inference"].__setitem__("elementary_bound_count", 12)),
            (
                "safety",
                lambda value: value["inference"]["safety"].__setitem__(
                    "point_mean_max", -0.05
                ),
            ),
            ("utility", lambda value: value["inference"]["utility"].__setitem__("ucb_max", 0.10)),
            (
                "utility_zero",
                lambda value: value["inference"]["utility"].__setitem__(
                    "zero_denominator_rule", "define 0/0 as 1"
                ),
            ),
            (
                "completion",
                lambda value: value["inference"].__setitem__(
                    "all_planned_runs_must_be_complete", False
                ),
            ),
            (
                "sign_flip",
                lambda value: value["inference"][
                    "secondary_sign_flip"
                ].__setitem__("correction", "none"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                changed = copy.deepcopy(config)
                mutate(changed)
                errors = VALIDATE_CONFIGS.validate_confirmatory_core(
                    ROOT, relative, changed
                )
                self.assertTrue(
                    any("inference contract must match" in error for error in errors)
                )


if __name__ == "__main__":
    unittest.main()
