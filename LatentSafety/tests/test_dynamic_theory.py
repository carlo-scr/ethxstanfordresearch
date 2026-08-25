import math
import random
import unittest
from fractions import Fraction
from itertools import combinations, product

from latent_safety.metrics.dynamic import (
    audit_exact_representation_viability,
    audit_finite_dynamic_sufficiency,
    audit_finite_q_greedy_policy,
    audit_finite_recursive_representation_viability,
    audit_finite_set_representation_viability,
    audit_finite_viable_history_set_family,
    calculate_conditional_euclidean_q_bound,
    calculate_verified_finite_domain_q_bound,
    check_dynamic_data_processing,
)


PREMISE_PROVENANCE = {
    "history_cover": "analytic compact-domain cover",
    "representation_lipschitz": "analytic encoder bound",
    "q_lipschitz": "analytic action-value bound",
    "q_estimation_error": "exact tabulation",
    "latent_metric": "Euclidean L2 throughout",
}


def one_step_conflict(*, merged: bool = True):
    actions = ("left", "right")
    histories = (
        ("plus_left", "plus_right", "minus_left", "minus_right"),
        ("plus", "minus"),
    )
    margins = {
        (0, "plus_left"): 1.0,
        (0, "plus_right"): -1.0,
        (0, "minus_left"): -1.0,
        (0, "minus_right"): 1.0,
        (1, "plus"): 1.0,
        (1, "minus"): 1.0,
    }
    successors = {
        (1, "plus", "left"): ("plus_left",),
        (1, "plus", "right"): ("plus_right",),
        (1, "minus", "left"): ("minus_left",),
        (1, "minus", "right"): ("minus_right",),
    }
    codes = {(0, history): history for history in histories[0]}
    codes[(1, "plus")] = "shared" if merged else "plus"
    codes[(1, "minus")] = "shared" if merged else "minus"
    return actions, histories, margins, successors, codes


def two_step_tight_fixture():
    actions = (0, 1)
    histories = (
        ("xia0", "xia1", "xib0", "xib1", "high", "low"),
        ("xi_a", "xi_b", "high_a", "high_b", "low_b"),
        ("eta_a", "eta_b"),
    )
    terminal_margins = {
        "xia0": 0.0,
        "xia1": 1.0,
        "xib0": 1.0,
        "xib1": -1.0,
        "high": 2.0,
        "low": 0.0,
    }
    margins = {(0, history): value for history, value in terminal_margins.items()}
    margins.update({(1, history): 2.0 for history in histories[1]})
    margins.update({(2, history): 2.0 for history in histories[2]})

    successors = {
        (1, "xi_a", 0): ("xia0",),
        (1, "xi_a", 1): ("xia1",),
        (1, "xi_b", 0): ("xib0",),
        (1, "xi_b", 1): ("xib1",),
        (1, "high_a", 0): ("high",),
        (1, "high_a", 1): ("high",),
        (1, "high_b", 0): ("high",),
        (1, "high_b", 1): ("high",),
        (1, "low_b", 0): ("low",),
        (1, "low_b", 1): ("low",),
        (2, "eta_a", 0): ("xi_a",),
        (2, "eta_a", 1): ("high_a",),
        (2, "eta_b", 0): ("high_b",),
        (2, "eta_b", 1): ("low_b",),
    }
    codes = {(0, history): history for history in histories[0]}
    codes.update(
        {
            (1, "xi_a"): "xi",
            (1, "xi_b"): "xi",
            (1, "high_a"): "high_a",
            (1, "high_b"): "high_b",
            (1, "low_b"): "low_b",
            (2, "eta_a"): "eta",
            (2, "eta_b"): "eta",
        }
    )
    return actions, histories, margins, successors, codes


def common_safe_but_margin_minimax_unsafe_fixture():
    actions = ("safe", "margin")
    histories = (("h1s", "h1m", "h2s", "h2m"), ("h1", "h2"))
    margins = {
        (0, "h1s"): 0.0,
        (0, "h1m"): 2.0,
        (0, "h2s"): 0.0,
        (0, "h2m"): -1.0,
        (1, "h1"): 2.0,
        (1, "h2"): 2.0,
    }
    successors = {
        (1, "h1", "safe"): ("h1s",),
        (1, "h1", "margin"): ("h1m",),
        (1, "h2", "safe"): ("h2s",),
        (1, "h2", "margin"): ("h2m",),
    }
    codes = {(0, history): history for history in histories[0]}
    codes.update({(1, "h1"): "shared", (1, "h2"): "shared"})
    return actions, histories, margins, successors, codes


def recursively_unsafe_representation_fixture():
    actions = (0, 1)
    histories = (
        ("plus0", "plus1", "minus0", "minus1"),
        ("plus", "minus"),
        ("root",),
    )
    margins = {
        (0, "plus0"): 1.0,
        (0, "plus1"): -1.0,
        (0, "minus0"): -1.0,
        (0, "minus1"): 1.0,
        (1, "plus"): 1.0,
        (1, "minus"): 1.0,
        (2, "root"): 1.0,
    }
    successors = {
        (1, "plus", 0): ("plus0",),
        (1, "plus", 1): ("plus1",),
        (1, "minus", 0): ("minus0",),
        (1, "minus", 1): ("minus1",),
        (2, "root", 0): ("plus", "minus"),
        (2, "root", 1): ("plus", "minus"),
    }
    codes = {(0, history): history for history in histories[0]}
    codes.update(
        {
            (1, "plus"): "stage1_shared",
            (1, "minus"): "stage1_shared",
            (2, "root"): "root",
        }
    )
    return actions, histories, margins, successors, codes


def future_code_conflict_fixture(*, split_stage_one: bool, two_roots: bool):
    """Two-stage fixture whose continuation actions conflict after merging."""

    actions = (0, 1)
    histories = (
        ("plus0", "plus1", "minus0", "minus1"),
        ("plus", "minus"),
        (("root_plus", "root_minus") if two_roots else ("root",)),
    )
    margins = {
        (0, "plus0"): 1.0,
        (0, "plus1"): -1.0,
        (0, "minus0"): -1.0,
        (0, "minus1"): 1.0,
        (1, "plus"): 1.0,
        (1, "minus"): 1.0,
    }
    margins.update({(2, history): 1.0 for history in histories[2]})
    successors = {
        (1, "plus", 0): ("plus0",),
        (1, "plus", 1): ("plus1",),
        (1, "minus", 0): ("minus0",),
        (1, "minus", 1): ("minus1",),
    }
    if two_roots:
        for action in actions:
            successors[(2, "root_plus", action)] = ("plus",)
            successors[(2, "root_minus", action)] = ("minus",)
    else:
        for action in actions:
            successors[(2, "root", action)] = ("plus", "minus")

    codes = {(0, history): history for history in histories[0]}
    codes[(1, "plus")] = "plus" if split_stage_one else "shared"
    codes[(1, "minus")] = "minus" if split_stage_one else "shared"
    if two_roots:
        codes[(2, "root_plus")] = "root_plus"
        codes[(2, "root_minus")] = "root_minus"
    else:
        codes[(2, "root")] = "root"
    return actions, histories, margins, successors, codes


def brute_force_set_policy_value(
    *,
    actions,
    histories,
    margins,
    successors,
    codes,
    remaining_steps,
    initial_histories,
):
    """Independent full-policy enumeration for a same-layer initial set."""

    policy_keys = []
    for remaining in range(1, remaining_steps + 1):
        stage_codes = []
        for history in histories[remaining]:
            code = codes[(remaining, history)]
            if code not in stage_codes:
                stage_codes.append(code)
        policy_keys.extend((remaining, code) for code in stage_codes)

    best = -math.inf
    for assignment in product(actions, repeat=len(policy_keys)):
        policy = dict(zip(policy_keys, assignment, strict=True))
        policy_values = [dict() for _ in range(remaining_steps + 1)]
        policy_values[0] = {
            history: margins[(0, history)] for history in histories[0]
        }
        for remaining in range(1, remaining_steps + 1):
            for history in histories[remaining]:
                action = policy[(remaining, codes[(remaining, history)])]
                policy_values[remaining][history] = min(
                    margins[(remaining, history)],
                    min(
                        policy_values[remaining - 1][next_history]
                        for next_history in successors[
                            (remaining, history, action)
                        ]
                    ),
                )
        best = max(
            best,
            min(
                policy_values[remaining_steps][history]
                for history in initial_histories
            ),
        )
    return best


class FiniteDynamicTheoryTests(unittest.TestCase):
    def test_one_step_action_conflict_is_exact_and_factor_two_is_tight(self) -> None:
        actions, histories, margins, successors, codes = one_step_conflict()
        audit = audit_finite_dynamic_sufficiency(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )

        stage = audit.stage_audits[0]
        self.assertEqual(audit.optimal_values[1]["plus"], 1.0)
        self.assertEqual(audit.optimal_values[1]["minus"], 1.0)
        self.assertEqual(stage.max_optimal_margin_regret, 2.0)
        self.assertEqual(stage.best_uniform_q_error, 1.0)
        self.assertTrue(stage.factor_two_bound_holds)
        self.assertEqual(audit.policy_values[1]["minus"], -1.0)
        self.assertEqual(audit.max_actual_loss[1], 2.0)
        self.assertTrue(audit.theorem_bound_holds)

    def test_uniform_q_approximation_is_sufficient_but_not_necessary(self) -> None:
        actions = ("left", "right")
        histories = (("h1l", "h1r", "h2l", "h2r"), ("h1", "h2"))
        margins = {
            (0, "h1l"): 1.0,
            (0, "h1r"): 0.0,
            (0, "h2l"): 0.8,
            (0, "h2r"): -1.0,
            (1, "h1"): 2.0,
            (1, "h2"): 2.0,
        }
        successors = {
            (1, "h1", "left"): ("h1l",),
            (1, "h1", "right"): ("h1r",),
            (1, "h2", "left"): ("h2l",),
            (1, "h2", "right"): ("h2r",),
        }
        codes = {(0, history): history for history in histories[0]}
        codes.update({(1, "h1"): "shared", (1, "h2"): "shared"})

        audit = audit_finite_dynamic_sufficiency(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )
        stage = audit.stage_audits[0]

        self.assertEqual(stage.max_optimal_margin_regret, 0.0)
        self.assertEqual(stage.best_uniform_q_error, 0.5)
        self.assertEqual(audit.policy_values[1], audit.optimal_values[1])

    def test_local_and_global_horizon_bounds_can_be_attained_at_closed_endpoint(self) -> None:
        actions, histories, margins, successors, codes = two_step_tight_fixture()
        audit = audit_finite_dynamic_sufficiency(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )

        self.assertEqual(
            tuple(stage.max_optimal_margin_regret for stage in audit.stage_audits),
            (1.0, 1.0),
        )
        self.assertEqual(audit.optimal_values[2]["eta_a"], 2.0)
        self.assertEqual(audit.certificate_bounds[2]["eta_a"], 2.0)
        self.assertEqual(audit.policy_values[2]["eta_a"], 0.0)
        self.assertEqual(audit.max_actual_loss[2], 2.0)
        self.assertTrue(audit.theorem_bound_holds)
        self.assertTrue(audit.closed_endpoint_holds)

    def test_margin_minimax_can_be_unsafe_when_a_common_safe_action_exists(self) -> None:
        problem = common_safe_but_margin_minimax_unsafe_fixture()
        audit = audit_finite_dynamic_sufficiency(
            actions=problem[0],
            histories_by_remaining=problem[1],
            margins=problem[2],
            successors=problem[3],
            codes=problem[4],
        )
        stage = audit.stage_audits[0]

        self.assertEqual(stage.max_optimal_margin_regret, 1.0)
        self.assertEqual(dict(stage.optimal_margin_actions)["shared"], "margin")
        self.assertEqual(stage.max_first_action_obstruction, 0.0)
        self.assertEqual(dict(stage.first_action_safety_actions)["shared"], "safe")
        self.assertEqual(dict(stage.common_safe_actions)["shared"], ("safe",))
        self.assertEqual(audit.policy_values[1]["h2"], -1.0)
        self.assertEqual(audit.full_history_viable_counts[1], 2)
        self.assertEqual(audit.optimal_margin_policy_retained_viable_counts[1], 1)
        self.assertEqual(
            audit.first_action_safety_policy_retained_viable_counts[1], 2
        )
        self.assertTrue(audit.all_stage_first_action_feasible)
        self.assertTrue(audit.preserves_all_full_history_viability)
        self.assertTrue(audit.global_sign_characterization_holds)

        exact = audit_exact_representation_viability(
            actions=problem[0],
            histories_by_remaining=problem[1],
            margins=problem[2],
            successors=problem[3],
            codes=problem[4],
        )
        self.assertEqual(dict(exact.fiber_latent_values[1])["shared"], 0.0)
        self.assertTrue(dict(exact.fiber_latent_viable[1])["shared"])
        self.assertEqual(
            dict(exact.fiber_safe_initial_actions[1])["shared"], ("safe",)
        )
        self.assertEqual(exact.max_retained_viable_counts[1], 2)

    def test_first_action_obstruction_does_not_replace_recursive_viability(self) -> None:
        problem = recursively_unsafe_representation_fixture()
        audit = audit_finite_dynamic_sufficiency(
            actions=problem[0],
            histories_by_remaining=problem[1],
            margins=problem[2],
            successors=problem[3],
            codes=problem[4],
        )
        self.assertEqual(
            dict(audit.stage_audits[1].fiber_first_action_obstructions)["root"],
            0.0,
        )
        self.assertFalse(audit.all_stage_first_action_feasible)
        self.assertFalse(audit.preserves_all_full_history_viability)
        self.assertTrue(audit.global_sign_characterization_holds)

        exact = audit_exact_representation_viability(
            actions=problem[0],
            histories_by_remaining=problem[1],
            margins=problem[2],
            successors=problem[3],
            codes=problem[4],
        )
        self.assertEqual(exact.policy_count, 4)
        self.assertEqual(dict(exact.fiber_latent_values[2])["root"], -1.0)
        self.assertFalse(dict(exact.fiber_latent_viable[2])["root"])
        self.assertEqual(dict(exact.fiber_safe_initial_actions[2])["root"], ())
        self.assertEqual(exact.max_retained_viable_counts[2], 0)

    def test_q_greedy_policy_different_from_minimax_composes_across_stages(self) -> None:
        actions, histories, margins, successors, codes = two_step_tight_fixture()
        optimal_margin = audit_finite_dynamic_sufficiency(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )
        estimates = {}
        for remaining in range(1, len(histories)):
            stage_codes = dict.fromkeys(
                codes[(remaining, history)] for history in histories[remaining]
            )
            for code in stage_codes:
                fiber = [
                    history
                    for history in histories[remaining]
                    if codes[(remaining, history)] == code
                ]
                for action in actions:
                    q_values = [
                        optimal_margin.q_values[remaining][(history, action)]
                        for history in fiber
                    ]
                    estimates[(remaining, code, action)] = 0.5 * (
                        min(q_values) + max(q_values)
                    )

        estimates[(1, "xi", 0)] = 0.0
        estimates[(1, "xi", 1)] = 0.1
        estimates[(2, "eta", 0)] = 1.4
        estimates[(2, "eta", 1)] = 1.6
        q_audit = audit_finite_q_greedy_policy(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
            latent_q_estimates=estimates,
        )

        minimax_stage1 = dict(optimal_margin.stage_audits[0].optimal_margin_actions)
        minimax_stage2 = dict(optimal_margin.stage_audits[1].optimal_margin_actions)
        self.assertEqual(minimax_stage1["xi"], 0)
        self.assertEqual(minimax_stage2["eta"], 0)
        self.assertEqual(q_audit.composition.selected_actions[1]["xi"], 1)
        self.assertEqual(q_audit.composition.selected_actions[2]["eta"], 1)
        self.assertTrue(all(q_audit.stage_factor_two_bounds_hold))
        self.assertTrue(q_audit.composition.theorem_bound_holds)
        self.assertTrue(q_audit.composed_factor_two_bound_holds)
        for remaining in range(len(histories)):
            self.assertLessEqual(
                q_audit.composition.max_actual_loss[remaining],
                q_audit.cumulative_factor_two_bounds[remaining] + 1e-12,
            )

    def test_optimal_margin_regret_is_monotone_under_coarsening(self) -> None:
        actions, histories, margins, successors, coarse_codes = one_step_conflict()
        fine_codes = dict(coarse_codes)
        fine_codes[(1, "plus")] = "plus"
        fine_codes[(1, "minus")] = "minus"
        result = check_dynamic_data_processing(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            fine_codes=fine_codes,
            coarse_codes=coarse_codes,
        )

        self.assertTrue(result.is_deterministic_postprocessing)
        self.assertEqual(result.stage_postprocessing, (True,))
        self.assertEqual(result.fine_regrets, (0.0,))
        self.assertEqual(result.coarse_regrets, (2.0,))
        self.assertEqual(result.stage_monotone, (True,))
        self.assertTrue(result.monotone)

    def test_split_code_is_not_a_postprocessing_of_merged_code(self) -> None:
        actions, histories, margins, successors, merged_codes = one_step_conflict()
        split_codes = dict(merged_codes)
        split_codes[(1, "plus")] = "plus"
        split_codes[(1, "minus")] = "minus"
        result = check_dynamic_data_processing(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            fine_codes=merged_codes,
            coarse_codes=split_codes,
        )

        self.assertFalse(result.is_deterministic_postprocessing)
        self.assertEqual(result.stage_postprocessing, (False,))
        self.assertFalse(result.monotone)

    def test_terminal_only_code_mismatch_does_not_invalidate_stage_theorem(self) -> None:
        actions, histories, margins, successors, fine_codes = one_step_conflict()
        coarse_codes = dict(fine_codes)
        for terminal_history in histories[0]:
            fine_codes[(0, terminal_history)] = "terminal_shared"
            coarse_codes[(0, terminal_history)] = terminal_history

        result = check_dynamic_data_processing(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            fine_codes=fine_codes,
            coarse_codes=coarse_codes,
        )

        self.assertFalse(result.terminal_postprocessing)
        self.assertEqual(result.stage_postprocessing, (True,))
        self.assertTrue(result.is_deterministic_postprocessing)
        self.assertEqual(result.stage_monotone, (True,))
        self.assertTrue(result.monotone)

    def test_incomplete_successor_model_fails_closed(self) -> None:
        actions, histories, margins, successors, codes = one_step_conflict()
        del successors[(1, "minus", "right")]
        with self.assertRaisesRegex(ValueError, "missing successor"):
            audit_finite_dynamic_sufficiency(
                actions=actions,
                histories_by_remaining=histories,
                margins=margins,
                successors=successors,
                codes=codes,
            )

    def test_main_audit_rejects_boolean_and_string_real_surrogates(self) -> None:
        actions, histories, margins, successors, codes = one_step_conflict()
        margin_key = (0, "plus_left")
        for invalid_margin in (True, "1.0"):
            invalid_margins = dict(margins)
            invalid_margins[margin_key] = invalid_margin  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "real number"):
                audit_finite_dynamic_sufficiency(
                    actions=actions,
                    histories_by_remaining=histories,
                    margins=invalid_margins,
                    successors=successors,
                    codes=codes,
                )
        for invalid_tolerance in (True, "0.0"):
            with self.assertRaisesRegex(ValueError, "real number"):
                audit_finite_dynamic_sufficiency(
                    actions=actions,
                    histories_by_remaining=histories,
                    margins=margins,
                    successors=successors,
                    codes=codes,
                    tolerance=invalid_tolerance,  # type: ignore[arg-type]
                )

    def test_exact_oracles_reject_inexact_binary64_conversion(self) -> None:
        actions, histories, margins, successors, codes = one_step_conflict()
        margin_key = (0, "plus_left")

        for inexact_margin in (Fraction(-1, 10**400), 2**53 + 1):
            invalid_margins = dict(margins)
            invalid_margins[margin_key] = inexact_margin  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "exactly representable"):
                audit_finite_set_representation_viability(
                    actions=actions,
                    histories_by_remaining=histories,
                    margins=invalid_margins,
                    successors=successors,
                    codes=codes,
                    remaining_steps=1,
                    initial_histories=histories[1],
                )

        overflow_margins = dict(margins)
        overflow_margins[margin_key] = 10**5000  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "finite and exactly representable"):
            audit_finite_set_representation_viability(
                actions=actions,
                histories_by_remaining=histories,
                margins=overflow_margins,
                successors=successors,
                codes=codes,
                remaining_steps=1,
                initial_histories=histories[1],
            )

        exact_margins = dict(margins)
        exact_margins[margin_key] = Fraction(1, 2)  # type: ignore[assignment]
        accepted = audit_finite_set_representation_viability(
            actions=actions,
            histories_by_remaining=histories,
            margins=exact_margins,
            successors=successors,
            codes=codes,
            remaining_steps=1,
            initial_histories=histories[1],
        )
        self.assertEqual(accepted.value, -1.0)

    def test_terminal_only_viability_conventions(self) -> None:
        common = {
            "actions": ("stay",),
            "histories_by_remaining": (("terminal",),),
            "margins": {(0, "terminal"): 0.0},
            "successors": {},
            "codes": {(0, "terminal"): "terminal_code"},
        }

        set_audit = audit_finite_set_representation_viability(
            **common,
            remaining_steps=0,
            initial_histories=("terminal",),
        )
        self.assertEqual(set_audit.value, 0.0)
        self.assertTrue(set_audit.viable)
        self.assertEqual(set_audit.optimal_first_action_maps, ())
        self.assertEqual(set_audit.safe_first_action_maps, ())
        self.assertEqual(set_audit.evaluated_action_map_count, 0)

        recursive = audit_finite_recursive_representation_viability(**common)
        self.assertEqual(recursive.horizon, 0)
        self.assertEqual(recursive.recursive_shortfalls, (0.0,))
        self.assertEqual(recursive.stage_first_action_obstructions, ())
        self.assertTrue(recursive.all_layers_preservable)
        self.assertTrue(recursive.all_stage_first_action_feasible)
        self.assertTrue(recursive.zero_shortfall_characterization_holds)

        family = audit_finite_viable_history_set_family(
            **common,
            remaining_steps=0,
        )
        self.assertEqual(family.viable_history_sets, ((), ("terminal",)))
        self.assertEqual(family.maximum_retained_fraction, 1.0)
        self.assertTrue(family.jointly_viable)
        self.assertEqual(family.recursive_shortfall, 0.0)

    def test_exact_viability_matches_independent_exhaustive_policy_enumeration(self) -> None:
        actions, histories, margins, successors, codes = (
            recursively_unsafe_representation_fixture()
        )
        exact = audit_exact_representation_viability(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )

        root_policy_margins = []
        for stage1_action, root_action in product(actions, repeat=2):
            terminal = {
                history: margins[(0, history)] for history in histories[0]
            }
            stage1 = {}
            for history in histories[1]:
                next_history = successors[(1, history, stage1_action)][0]
                stage1[history] = min(margins[(1, history)], terminal[next_history])
            root_next = successors[(2, "root", root_action)]
            root_policy_margins.append(
                min(margins[(2, "root")], *(stage1[item] for item in root_next))
            )

        self.assertEqual(len(root_policy_margins), exact.policy_count)
        self.assertEqual(max(root_policy_margins), -1.0)
        self.assertEqual(
            max(root_policy_margins), dict(exact.fiber_latent_values[2])["root"]
        )
        self.assertTrue(exact.all_policy_values_below_optimal)

    def test_set_bellman_matches_policy_enumeration_for_every_small_subset(
        self,
    ) -> None:
        generator = random.Random(20260823)
        actions = (0, 1)
        for game_index in range(12):
            histories = tuple(
                tuple(f"g{game_index}_s{remaining}_h{index}" for index in range(3))
                for remaining in range(3)
            )
            margins = {
                (remaining, history): float(generator.randint(-2, 2))
                for remaining, layer in enumerate(histories)
                for history in layer
            }
            successors = {}
            for remaining in range(1, len(histories)):
                for history in histories[remaining]:
                    for action in actions:
                        count = generator.randint(1, len(histories[remaining - 1]))
                        successors[(remaining, history, action)] = tuple(
                            generator.sample(histories[remaining - 1], count)
                        )
            codes = {
                (remaining, history): generator.randrange(2)
                for remaining, layer in enumerate(histories)
                for history in layer
            }

            for remaining, layer in enumerate(histories):
                for subset_size in range(1, len(layer) + 1):
                    for subset in combinations(layer, subset_size):
                        expected = brute_force_set_policy_value(
                            actions=actions,
                            histories=histories,
                            margins=margins,
                            successors=successors,
                            codes=codes,
                            remaining_steps=remaining,
                            initial_histories=subset,
                        )
                        actual = audit_finite_set_representation_viability(
                            actions=actions,
                            histories_by_remaining=histories,
                            margins=margins,
                            successors=successors,
                            codes=codes,
                            remaining_steps=remaining,
                            initial_histories=reversed(subset),
                        )
                        self.assertEqual(actual.value, expected)
                        self.assertEqual(actual.viable, expected >= 0.0)
                        self.assertEqual(
                            actual.recursive_shortfall,
                            max(0.0, -expected),
                        )
                        self.assertEqual(
                            actual.initial_histories,
                            tuple(history for history in layer if history in subset),
                        )

    def test_set_bellman_recovers_existing_fiber_values_and_safe_actions(self) -> None:
        actions, histories, margins, successors, codes = (
            recursively_unsafe_representation_fixture()
        )
        exhaustive = audit_exact_representation_viability(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )

        for remaining in range(1, len(histories)):
            fibers = {}
            for history in histories[remaining]:
                fibers.setdefault(codes[(remaining, history)], []).append(history)
            expected_values = dict(exhaustive.fiber_latent_values[remaining])
            expected_actions = dict(exhaustive.fiber_safe_initial_actions[remaining])
            for code, fiber in fibers.items():
                result = audit_finite_set_representation_viability(
                    actions=actions,
                    histories_by_remaining=histories,
                    margins=margins,
                    successors=successors,
                    codes=codes,
                    remaining_steps=remaining,
                    initial_histories=fiber,
                )
                self.assertEqual(result.value, expected_values[code])
                safe_actions = tuple(
                    action_map[0][1] for action_map in result.safe_first_action_maps
                )
                self.assertEqual(safe_actions, expected_actions[code])

    def test_dynamic_viability_data_processing_is_strict_on_action_conflict(
        self,
    ) -> None:
        actions, histories, margins, successors, coarse_codes = one_step_conflict()
        fine_codes = dict(coarse_codes)
        fine_codes[(1, "plus")] = "plus"
        fine_codes[(1, "minus")] = "minus"
        initial = histories[1]

        fine = audit_finite_set_representation_viability(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=fine_codes,
            remaining_steps=1,
            initial_histories=initial,
        )
        coarse = audit_finite_set_representation_viability(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=coarse_codes,
            remaining_steps=1,
            initial_histories=initial,
        )
        self.assertEqual(fine.value, 1.0)
        self.assertEqual(coarse.value, -1.0)
        self.assertEqual(fine.recursive_shortfall, 0.0)
        self.assertEqual(coarse.recursive_shortfall, 1.0)

        fine_family = audit_finite_viable_history_set_family(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=fine_codes,
            remaining_steps=1,
        )
        coarse_family = audit_finite_viable_history_set_family(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=coarse_codes,
            remaining_steps=1,
        )
        self.assertTrue(fine_family.jointly_viable)
        self.assertFalse(coarse_family.jointly_viable)
        self.assertEqual(fine_family.maximum_retained_count, 2)
        self.assertEqual(coarse_family.maximum_retained_count, 1)
        self.assertTrue(fine_family.hereditary)
        self.assertTrue(coarse_family.hereditary)
        self.assertTrue(
            set(coarse_family.viable_history_sets).issubset(
                set(fine_family.viable_history_sets)
            )
        )

    def test_set_viability_is_monotone_under_random_stagewise_coarsening(self) -> None:
        generator = random.Random(20260824)
        actions = (0, 1)
        for game_index in range(40):
            histories = tuple(
                tuple(f"m{game_index}_s{remaining}_h{index}" for index in range(3))
                for remaining in range(3)
            )
            margins = {
                (remaining, history): float(generator.randint(-3, 3))
                for remaining, layer in enumerate(histories)
                for history in layer
            }
            successors = {}
            for remaining in range(1, len(histories)):
                for history in histories[remaining]:
                    for action in actions:
                        successors[(remaining, history, action)] = tuple(
                            generator.sample(
                                histories[remaining - 1],
                                generator.randint(1, 3),
                            )
                        )
            fine_codes = {
                (remaining, history): (remaining, generator.randrange(3))
                for remaining, layer in enumerate(histories)
                for history in layer
            }
            coarse_map = {
                fine_code: (fine_code[0], fine_code[1] % 2)
                for fine_code in set(fine_codes.values())
            }
            coarse_codes = {
                key: coarse_map[fine_code]
                for key, fine_code in fine_codes.items()
            }
            for remaining in range(len(histories)):
                subset = tuple(
                    history
                    for history in histories[remaining]
                    if generator.random() < 0.7
                ) or (histories[remaining][0],)
                fine = audit_finite_set_representation_viability(
                    actions=actions,
                    histories_by_remaining=histories,
                    margins=margins,
                    successors=successors,
                    codes=fine_codes,
                    remaining_steps=remaining,
                    initial_histories=subset,
                )
                coarse = audit_finite_set_representation_viability(
                    actions=actions,
                    histories_by_remaining=histories,
                    margins=margins,
                    successors=successors,
                    codes=coarse_codes,
                    remaining_steps=remaining,
                    initial_histories=subset,
                )
                self.assertGreaterEqual(fine.value, coarse.value)
                self.assertLessEqual(
                    fine.recursive_shortfall, coarse.recursive_shortfall
                )

    def test_current_layer_postprocessing_alone_does_not_order_recursive_value(
        self,
    ) -> None:
        fine_problem = future_code_conflict_fixture(
            split_stage_one=True, two_roots=False
        )
        coarse_problem = future_code_conflict_fixture(
            split_stage_one=False, two_roots=False
        )
        self.assertEqual(fine_problem[4][(2, "root")], coarse_problem[4][(2, "root")])

        fine = audit_finite_set_representation_viability(
            actions=fine_problem[0],
            histories_by_remaining=fine_problem[1],
            margins=fine_problem[2],
            successors=fine_problem[3],
            codes=fine_problem[4],
            remaining_steps=2,
            initial_histories=("root",),
        )
        coarse = audit_finite_set_representation_viability(
            actions=coarse_problem[0],
            histories_by_remaining=coarse_problem[1],
            margins=coarse_problem[2],
            successors=coarse_problem[3],
            codes=coarse_problem[4],
            remaining_steps=2,
            initial_histories=("root",),
        )
        self.assertEqual(fine.value, 1.0)
        self.assertEqual(coarse.value, -1.0)

    def test_individually_viable_fibers_can_fail_when_a_union_couples_them(self) -> None:
        problem = future_code_conflict_fixture(
            split_stage_one=False, two_roots=True
        )
        common = {
            "actions": problem[0],
            "histories_by_remaining": problem[1],
            "margins": problem[2],
            "successors": problem[3],
            "codes": problem[4],
            "remaining_steps": 2,
        }
        plus = audit_finite_set_representation_viability(
            **common, initial_histories=("root_plus",)
        )
        minus = audit_finite_set_representation_viability(
            **common, initial_histories=("root_minus",)
        )
        union = audit_finite_set_representation_viability(
            **common, initial_histories=("root_plus", "root_minus")
        )
        self.assertEqual(plus.value, 1.0)
        self.assertEqual(minus.value, 1.0)
        self.assertEqual(union.value, -1.0)

        family = audit_finite_viable_history_set_family(**common)
        self.assertEqual(family.maximum_retained_count, 1)
        self.assertEqual(family.maximum_retained_fraction, 0.5)
        self.assertFalse(family.jointly_viable)
        self.assertEqual(family.recursive_shortfall, 1.0)

    def test_recursive_shortfall_matches_global_first_action_characterization(
        self,
    ) -> None:
        merged = recursively_unsafe_representation_fixture()
        merged_audit = audit_finite_recursive_representation_viability(
            actions=merged[0],
            histories_by_remaining=merged[1],
            margins=merged[2],
            successors=merged[3],
            codes=merged[4],
        )
        self.assertEqual(merged_audit.recursive_shortfalls, (0.0, 1.0, 1.0))
        self.assertEqual(merged_audit.stage_first_action_obstructions, (1.0, 0.0))
        self.assertFalse(merged_audit.all_layers_preservable)
        self.assertFalse(merged_audit.all_stage_first_action_feasible)
        self.assertTrue(merged_audit.zero_shortfall_characterization_holds)

        split = future_code_conflict_fixture(
            split_stage_one=True, two_roots=False
        )
        split_audit = audit_finite_recursive_representation_viability(
            actions=split[0],
            histories_by_remaining=split[1],
            margins=split[2],
            successors=split[3],
            codes=split[4],
        )
        self.assertEqual(split_audit.recursive_shortfalls, (0.0, 0.0, 0.0))
        self.assertTrue(split_audit.all_layers_preservable)
        self.assertTrue(split_audit.all_stage_first_action_feasible)
        self.assertTrue(split_audit.zero_shortfall_characterization_holds)

    def test_set_viability_guards_exponential_enumeration(self) -> None:
        actions, histories, margins, successors, codes = one_step_conflict()
        with self.assertRaisesRegex(ValueError, "max_action_map_count"):
            audit_finite_set_representation_viability(
                actions=actions,
                histories_by_remaining=histories,
                margins=margins,
                successors=successors,
                codes=codes,
                remaining_steps=1,
                initial_histories=histories[1],
                max_action_map_count=1,
            )
        with self.assertRaisesRegex(ValueError, "above max_subset_count"):
            audit_finite_viable_history_set_family(
                actions=actions,
                histories_by_remaining=histories,
                margins=margins,
                successors=successors,
                codes=codes,
                remaining_steps=1,
                max_subset_count=3,
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            audit_finite_set_representation_viability(
                actions=actions,
                histories_by_remaining=histories,
                margins=margins,
                successors=successors,
                codes=codes,
                remaining_steps=1,
                initial_histories=(),
            )

    def test_model_local_lipschitz_hausdorff_bound_is_numerically_exact(self) -> None:
        actions = ("a",)
        histories = ((0.0, 1.0, 2.0), ("left", "right"))
        margins = {
            (0, 0.0): 0.0,
            (0, 1.0): 1.0,
            (0, 2.0): 2.0,
            (1, "left"): 3.0,
            (1, "right"): 2.0,
        }
        successors = {
            (1, "left", "a"): (0.0, 1.0),
            (1, "right", "a"): (1.0, 2.0),
        }
        codes = {
            (0, 0.0): 0.0,
            (0, 1.0): 1.0,
            (0, 2.0): 2.0,
            (1, "left"): "shared",
            (1, "right"): "shared",
        }
        audit = audit_finite_dynamic_sufficiency(
            actions=actions,
            histories_by_remaining=histories,
            margins=margins,
            successors=successors,
            codes=codes,
        )

        left_set = successors[(1, "left", "a")]
        right_set = successors[(1, "right", "a")]
        directed_left = max(min(abs(x - y) for y in right_set) for x in left_set)
        directed_right = max(min(abs(y - x) for x in left_set) for y in right_set)
        hausdorff = max(directed_left, directed_right)
        alpha = abs(margins[(1, "left")] - margins[(1, "right")])
        bound = max(alpha, hausdorff)
        q_difference = abs(
            audit.q_values[1][("left", "a")]
            - audit.q_values[1][("right", "a")]
        )
        self.assertEqual(hausdorff, 1.0)
        self.assertEqual(q_difference, bound)

    def test_random_finite_games_satisfy_local_and_global_bounds(self) -> None:
        generator = random.Random(20260822)
        actions = (0, 1, 2)
        for _ in range(100):
            histories = tuple(
                tuple(f"s{remaining}_h{index}" for index in range(4))
                for remaining in range(4)
            )
            margins = {
                (remaining, history): generator.uniform(-2.0, 2.0)
                for remaining, layer in enumerate(histories)
                for history in layer
            }
            successors = {}
            for remaining in range(1, len(histories)):
                for history in histories[remaining]:
                    for action in actions:
                        count = generator.randint(1, len(histories[remaining - 1]))
                        successors[(remaining, history, action)] = tuple(
                            generator.sample(histories[remaining - 1], count)
                        )
            codes = {
                (remaining, history): generator.randrange(2)
                for remaining, layer in enumerate(histories)
                for history in layer
            }

            audit = audit_finite_dynamic_sufficiency(
                actions=actions,
                histories_by_remaining=histories,
                margins=margins,
                successors=successors,
                codes=codes,
            )

            self.assertTrue(audit.theorem_bound_holds)
            self.assertTrue(audit.optimal_dominates_policy)
            self.assertTrue(audit.local_bound_holds)
            self.assertTrue(audit.certificate_within_global_bound)
            self.assertTrue(audit.closed_endpoint_holds)
            self.assertTrue(audit.global_sign_characterization_holds)
            self.assertTrue(
                all(stage.factor_two_bound_holds for stage in audit.stage_audits)
            )
            for remaining, layer in enumerate(histories):
                for history in layer:
                    loss = (
                        audit.optimal_values[remaining][history]
                        - audit.policy_values[remaining][history]
                    )
                    local_bound = audit.certificate_bounds[remaining][history]
                    self.assertGreaterEqual(loss, -1e-12)
                    self.assertLessEqual(loss, local_bound + 1e-12)
                    self.assertLessEqual(
                        local_bound,
                        audit.cumulative_global_regret[remaining] + 1e-12,
                    )

    def test_conditional_bound_recovers_exact_duplicate_code_oscillation(self) -> None:
        certificate = calculate_conditional_euclidean_q_bound(
            sample_latents=[(0.0,), (0.0,), (1.0,)],
            sample_q_values=[(1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)],
            cover_radius=0.0,
            representation_lipschitz=1.0,
            q_lipschitz=1.0,
            premise_provenance=PREMISE_PROVENANCE,
        )

        self.assertEqual(certificate.nominal_search_radius, 0.0)
        self.assertEqual(certificate.latent_metric, "euclidean_l2")
        self.assertIn("not_verified_certificate", certificate.status)
        self.assertEqual(certificate.cover_radius, 0.0)
        self.assertEqual(certificate.representation_lipschitz, 1.0)
        self.assertEqual(certificate.q_lipschitz, 1.0)
        self.assertEqual(certificate.q_estimation_error, 0.0)
        self.assertEqual(certificate.sampled_q_oscillation, 2.0)
        self.assertEqual(certificate.optimal_margin_regret_upper_bound, 2.0)
        self.assertIn("not inferred", certificate.premise)

    def test_conditional_bound_expands_radius_and_adds_uniform_errors(self) -> None:
        certificate = calculate_conditional_euclidean_q_bound(
            sample_latents=[(-1.0,), (0.0,), (1.0,)],
            sample_q_values=[(-1.0, 1.0), (0.0, 0.0), (1.0, -1.0)],
            cover_radius=0.5,
            representation_lipschitz=1.0,
            q_lipschitz=1.0,
            premise_provenance=PREMISE_PROVENANCE,
            q_estimation_error=0.1,
        )

        self.assertEqual(certificate.nominal_search_radius, 1.0)
        self.assertEqual(certificate.sampled_q_oscillation, 1.0)
        self.assertAlmostEqual(certificate.q_oscillation_upper_bound, 2.2)

    def test_conditional_bound_is_tight_on_folded_interval_example(self) -> None:
        # H=[-1,1], S={-0.5,0,0.5} is a 0.5-net, R(x)=|x| is 1-Lipschitz,
        # and Q(x)=(x,-x) has coordinate Lipschitz constant one.  The omitted
        # endpoint pair {-1,1} is an exact fiber with Q oscillation/regret two.
        certificate = calculate_conditional_euclidean_q_bound(
            sample_latents=[(0.5,), (0.0,), (0.5,)],
            sample_q_values=[(-0.5, 0.5), (0.0, 0.0), (0.5, -0.5)],
            cover_radius=0.5,
            representation_lipschitz=1.0,
            q_lipschitz=1.0,
            premise_provenance=PREMISE_PROVENANCE,
        )

        self.assertEqual(certificate.nominal_search_radius, 1.0)
        self.assertEqual(certificate.sampled_q_oscillation, 1.0)
        self.assertEqual(certificate.optimal_margin_regret_upper_bound, 2.0)

    def test_conditional_bound_is_explicitly_euclidean_and_tolerance_is_visible(self) -> None:
        without_tolerance = calculate_conditional_euclidean_q_bound(
            sample_latents=[(0.0, 0.0), (0.8, 0.8)],
            sample_q_values=[(0.0,), (10.0,)],
            cover_radius=0.0,
            representation_lipschitz=1.0,
            q_lipschitz=0.0,
            operational_radius=1.0,
            premise_provenance=PREMISE_PROVENANCE,
        )
        with_tolerance = calculate_conditional_euclidean_q_bound(
            sample_latents=[(0.0, 0.0), (0.8, 0.8)],
            sample_q_values=[(0.0,), (10.0,)],
            cover_radius=0.0,
            representation_lipschitz=1.0,
            q_lipschitz=0.0,
            operational_radius=1.0,
            radius_comparison_tolerance=0.14,
            premise_provenance=PREMISE_PROVENANCE,
        )

        self.assertEqual(without_tolerance.latent_metric, "euclidean_l2")
        self.assertEqual(without_tolerance.nominal_search_radius, 1.0)
        self.assertEqual(without_tolerance.effective_search_radius, 1.0)
        self.assertEqual(without_tolerance.sampled_q_oscillation, 0.0)
        self.assertEqual(with_tolerance.radius_comparison_tolerance, 0.14)
        self.assertAlmostEqual(with_tolerance.effective_search_radius, 1.14)
        self.assertEqual(with_tolerance.sampled_q_oscillation, 10.0)

    def test_conditional_bound_requires_every_premise_provenance_field(self) -> None:
        incomplete = dict(PREMISE_PROVENANCE)
        del incomplete["history_cover"]
        with self.assertRaisesRegex(ValueError, "history_cover"):
            calculate_conditional_euclidean_q_bound(
                sample_latents=[(0.0,)],
                sample_q_values=[(0.0,)],
                cover_radius=0.0,
                representation_lipschitz=1.0,
                q_lipschitz=1.0,
                premise_provenance=incomplete,
            )

    def test_conditional_bound_rejects_unverifiable_numeric_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover_radius"):
            calculate_conditional_euclidean_q_bound(
                sample_latents=[(0.0,)],
                sample_q_values=[(0.0,)],
                cover_radius=-0.1,
                representation_lipschitz=1.0,
                q_lipschitz=1.0,
                premise_provenance=PREMISE_PROVENANCE,
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            calculate_conditional_euclidean_q_bound(
                sample_latents=[(0.0,)],
                sample_q_values=[],
                cover_radius=0.0,
                representation_lipschitz=1.0,
                q_lipschitz=1.0,
                premise_provenance=PREMISE_PROVENANCE,
            )
        with self.assertRaisesRegex(ValueError, "real number"):
            calculate_conditional_euclidean_q_bound(
                sample_latents=[(0.0,)],
                sample_q_values=[(0.0,)],
                cover_radius="0.0",  # type: ignore[arg-type]
                representation_lipschitz=1.0,
                q_lipschitz=1.0,
                premise_provenance=PREMISE_PROVENANCE,
            )
        with self.assertRaisesRegex(ValueError, "real number"):
            calculate_conditional_euclidean_q_bound(
                sample_latents=[(False,)],
                sample_q_values=[(0.0,)],
                cover_radius=0.0,
                representation_lipschitz=1.0,
                q_lipschitz=1.0,
                premise_provenance=PREMISE_PROVENANCE,
            )

    def test_verified_finite_domain_bound_derives_and_checks_every_premise(self) -> None:
        population = (-1.0, -0.5, 0.0, 0.5, 1.0)
        result = calculate_verified_finite_domain_q_bound(
            history_distances=[
                [abs(left - right) for right in population] for left in population
            ],
            population_latents=[(abs(point),) for point in population],
            population_q_values=[(point, -point) for point in population],
            sample_indices=(1, 2, 3),
        )

        self.assertEqual(result.status, "verified_complete_finite_domain_only")
        self.assertEqual(
            result.calculation.status,
            "premises_verified_on_complete_finite_domain_only",
        )
        self.assertEqual(result.domain_count, 5)
        self.assertEqual(result.sample_indices, (1, 2, 3))
        self.assertEqual(result.latent_metric, "euclidean_l2")
        self.assertEqual(result.cover_radius, 0.5)
        self.assertEqual(result.exact_representation_lipschitz, 1.0)
        self.assertEqual(result.exact_q_lipschitz, 1.0)
        self.assertEqual(result.exact_sample_q_estimation_error, 0.0)
        self.assertEqual(result.population_q_oscillation, 2.0)
        self.assertEqual(
            result.outward_population_q_oscillation,
            math.nextafter(2.0, math.inf),
        )
        self.assertEqual(
            result.q_oscillation_upper_bound,
            result.outward_population_q_oscillation,
        )
        self.assertEqual(result.bound_to_global_oscillation_ratio, 1.0)
        self.assertFalse(result.nonvacuous_against_global_oscillation)
        self.assertTrue(result.population_bound_holds)

        noisy = calculate_verified_finite_domain_q_bound(
            history_distances=[
                [abs(left - right) for right in population] for left in population
            ],
            population_latents=[(abs(point),) for point in population],
            population_q_values=[(point, -point) for point in population],
            sample_indices=(1, 2, 3),
            sample_q_estimates=[
                (population[index] + 0.1, -population[index] + 0.1)
                for index in (1, 2, 3)
            ],
        )
        self.assertAlmostEqual(noisy.exact_sample_q_estimation_error, 0.1)
        self.assertAlmostEqual(noisy.q_oscillation_upper_bound, 2.2)
        self.assertTrue(noisy.population_bound_holds)

    def test_verified_finite_domain_bound_outward_corrects_roundoff(self) -> None:
        inner = 106.29554989233057
        outer = 394.47360659037525
        population = (-outer, -inner, 0.0, inner, outer)
        result = calculate_verified_finite_domain_q_bound(
            history_distances=[
                [abs(left - right) for right in population] for left in population
            ],
            population_latents=[(abs(point),) for point in population],
            population_q_values=[(point, -point) for point in population],
            sample_indices=(1, 2, 3),
        )

        self.assertGreater(
            result.population_q_oscillation,
            result.raw_cover_q_oscillation_upper_bound,
        )
        self.assertEqual(
            result.q_oscillation_upper_bound,
            result.outward_population_q_oscillation,
        )
        self.assertEqual(
            result.optimal_margin_regret_upper_bound,
            result.outward_population_q_oscillation,
        )
        self.assertGreater(result.outward_bound_correction, 0.0)
        self.assertTrue(result.population_bound_holds)

    def test_verified_finite_domain_bound_encloses_exact_stored_float_regret(
        self,
    ) -> None:
        small = 2.0**-54
        result = calculate_verified_finite_domain_q_bound(
            history_distances=((0.0, 1.0), (1.0, 0.0)),
            population_latents=((0.0,), (0.0,)),
            population_q_values=((1.0, -small), (-small, 1.0)),
            sample_indices=(0, 1),
        )

        exact_regret = Fraction.from_float(1.0) - Fraction.from_float(-small)
        returned_bound = Fraction.from_float(
            result.optimal_margin_regret_upper_bound
        )
        self.assertEqual(result.population_q_oscillation, 1.0)
        self.assertEqual(
            result.outward_population_q_oscillation,
            math.nextafter(1.0, math.inf),
        )
        self.assertGreaterEqual(returned_bound, exact_regret)
        self.assertGreater(
            result.optimal_margin_regret_upper_bound,
            result.population_q_oscillation,
        )
        self.assertTrue(result.population_bound_holds)

    def test_verified_finite_domain_outward_step_is_not_absolute_tolerance_limited(
        self,
    ) -> None:
        large = 1.0e24
        result = calculate_verified_finite_domain_q_bound(
            history_distances=((0.0, 1.0), (1.0, 0.0)),
            population_latents=((0.0,), (0.0,)),
            population_q_values=((0.0,), (large,)),
            sample_indices=(0, 1),
        )

        self.assertEqual(
            result.raw_cover_q_oscillation_upper_bound,
            result.population_q_oscillation,
        )
        self.assertEqual(
            result.q_oscillation_upper_bound,
            math.nextafter(result.population_q_oscillation, math.inf),
        )
        self.assertGreater(result.outward_bound_correction, 1e-12)
        self.assertTrue(result.population_bound_holds)

    def test_verified_finite_domain_bound_rejects_invalid_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "triangle"):
            calculate_verified_finite_domain_q_bound(
                history_distances=((0.0, 1.0, 3.0), (1.0, 0.0, 1.0), (3.0, 1.0, 0.0)),
                population_latents=((0.0,), (1.0,), (2.0,)),
                population_q_values=((0.0,), (1.0,), (2.0,)),
                sample_indices=(0, 2),
            )

    def test_dynamic_and_cover_audits_reject_nonfinite_derived_arithmetic(self) -> None:
        actions, histories, margins, successors, codes = one_step_conflict()
        huge_margins = {
            key: (1e308 if value > 0.0 else -1e308)
            for key, value in margins.items()
        }
        with self.assertRaisesRegex(ValueError, "derived action regret"):
            audit_finite_dynamic_sufficiency(
                actions=actions,
                histories_by_remaining=histories,
                margins=huge_margins,
                successors=successors,
                codes=codes,
            )

        with self.assertRaisesRegex(ValueError, "derived nominal search radius"):
            calculate_conditional_euclidean_q_bound(
                sample_latents=((0.0,),),
                sample_q_values=((0.0,),),
                cover_radius=1e308,
                representation_lipschitz=1e308,
                q_lipschitz=0.0,
                premise_provenance=PREMISE_PROVENANCE,
            )


if __name__ == "__main__":
    unittest.main()
