import dataclasses
import math
import unittest
from pathlib import Path

from latent_safety.benchmarks.dynamic_oracle import (
    MEMORYLESS_PIXELS,
    PRIVILEGED_STATE,
    TWO_FRAME_HISTORY,
    audit_controlled_dynamic_oracle,
    build_controlled_dynamic_tree,
)
from latent_safety.learning import (
    CartState,
    PendulumState,
    deterministic_step,
    load_config,
    render_state,
    safety_margin,
)
from latent_safety.metrics.dynamic import audit_finite_dynamic_sufficiency


ROOT = Path(__file__).resolve().parents[1]


class ControlledDynamicOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cart_config = load_config(
            ROOT / "configs/e1_world_models/torch_pilot.toml"
        ).data
        cls.pendulum_config = load_config(
            ROOT / "configs/e1_world_models/torch_pendulum_pilot.toml"
        ).data
        cls.cart_tree = build_controlled_dynamic_tree(cls.cart_config, horizon=3)
        cls.pendulum_tree = build_controlled_dynamic_tree(
            cls.pendulum_config, horizon=3
        )
        cls.cart_report = audit_controlled_dynamic_oracle(
            cls.cart_config, horizon=3
        )
        cls.pendulum_report = audit_controlled_dynamic_oracle(
            cls.pendulum_config, horizon=3
        )

    @staticmethod
    def _summary(report, name):
        return next(
            summary
            for summary in report.representations
            if summary.representation == name
        )

    def test_complete_three_step_action_trees_have_expected_shape(self) -> None:
        for tree, config in (
            (self.cart_tree, self.cart_config),
            (self.pendulum_tree, self.pendulum_config),
        ):
            self.assertEqual(tree.node_count_by_remaining, (54, 18, 6, 2))
            self.assertEqual(len(tree.nodes_by_history), 80)
            self.assertEqual(len(tree.successors), 78)
            for remaining in range(1, tree.horizon + 1):
                for history in tree.histories_by_remaining[remaining]:
                    parent = tree.nodes_by_history[history]
                    for action in tree.actions:
                        successors = tree.successors[(remaining, history, action)]
                        self.assertEqual(len(successors), 1)
                        self.assertIn(
                            successors[0], tree.histories_by_remaining[remaining - 1]
                        )
                        child = tree.nodes_by_history[successors[0]]
                        self.assertEqual(child.previous_state, parent.state)
                        self.assertEqual(child.previous_action, action)
                        self.assertEqual(
                            child.action_path, parent.action_path + (action,)
                        )
                        self.assertEqual(
                            deterministic_step(parent.state, action, config),
                            child.state,
                        )

    def test_roots_are_reachable_pixel_aliases_resolved_by_history(self) -> None:
        for tree, config in (
            (self.cart_tree, self.cart_config),
            (self.pendulum_tree, self.pendulum_config),
        ):
            roots = tree.root_histories
            remaining = tree.horizon
            memoryless_codes = tree.codes_by_representation[MEMORYLESS_PIXELS]
            history_codes = tree.codes_by_representation[TWO_FRAME_HISTORY]
            state_codes = tree.codes_by_representation[PRIVILEGED_STATE]
            self.assertEqual(
                memoryless_codes[(remaining, roots[0])],
                memoryless_codes[(remaining, roots[1])],
            )
            self.assertNotEqual(
                history_codes[(remaining, roots[0])],
                history_codes[(remaining, roots[1])],
            )
            self.assertNotEqual(
                state_codes[(remaining, roots[0])],
                state_codes[(remaining, roots[1])],
            )
            first_history_code = history_codes[(remaining, roots[0])]
            second_history_code = history_codes[(remaining, roots[1])]
            self.assertEqual(
                first_history_code[1], memoryless_codes[(remaining, roots[0])]
            )
            self.assertEqual(
                second_history_code[1], memoryless_codes[(remaining, roots[1])]
            )
            self.assertNotEqual(first_history_code[0], second_history_code[0])

            for history in roots:
                node = tree.nodes_by_history[history]
                self.assertGreaterEqual(safety_margin(node.previous_state, config), 0.0)
                self.assertGreaterEqual(safety_margin(node.state, config), 0.0)
                reached = deterministic_step(
                    node.previous_state, node.previous_action, config
                )
                self.assertEqual(
                    render_state(reached, config), render_state(node.state, config)
                )
                if isinstance(node.state, CartState):
                    self.assertIsInstance(reached, CartState)
                    self.assertAlmostEqual(reached.position, node.state.position)
                    self.assertAlmostEqual(reached.velocity, node.state.velocity)
                else:
                    self.assertIsInstance(node.state, PendulumState)
                    self.assertIsInstance(reached, PendulumState)
                    angle_error = math.remainder(
                        reached.angle - node.state.angle, 2.0 * math.pi
                    )
                    self.assertAlmostEqual(angle_error, 0.0)
                    self.assertAlmostEqual(
                        reached.angular_velocity, node.state.angular_velocity
                    )

    def test_privileged_and_history_representations_retain_optimal_value(self) -> None:
        for report in (self.cart_report, self.pendulum_report):
            for name in (PRIVILEGED_STATE, TWO_FRAME_HISTORY):
                summary = self._summary(report, name)
                self.assertEqual(summary.root_code_count, 2)
                self.assertTrue(summary.theorem_bound_holds)
                self.assertTrue(summary.closed_endpoint_holds)
                self.assertTrue(
                    summary.stagewise_first_action_feasible_within_tolerance
                )
                self.assertTrue(
                    summary.exact_viability_preserving_code_policy_exists
                )
                self.assertTrue(summary.global_sign_characterization_holds)
                self.assertTrue(all(stage.rho_s == 0.0 for stage in summary.stages))
                self.assertTrue(all(stage.kappa_s == 0.0 for stage in summary.stages))
                self.assertTrue(
                    all(
                        stage.max_realized_policy_loss == 0.0
                        for stage in summary.stages
                    )
                )
                self.assertTrue(
                    all(
                        outcome.optimal_value
                        == outcome.representation_policy_value
                        for outcome in summary.root_outcomes
                    )
                )

    def test_cart_memoryless_alias_has_margin_regret_and_sign_obstruction(self) -> None:
        summary = self._summary(self.cart_report, MEMORYLESS_PIXELS)
        root_stage = summary.stages[-1]
        self.assertEqual(summary.root_code_count, 1)
        self.assertAlmostEqual(root_stage.rho_s, 0.03585254400000015)
        self.assertAlmostEqual(root_stage.kappa_s, 0.01798745600000007)
        self.assertAlmostEqual(root_stage.delta_s_star, 0.04126310400000005)
        self.assertAlmostEqual(
            root_stage.max_realized_policy_loss, root_stage.rho_s
        )
        self.assertAlmostEqual(root_stage.max_path_bound_B, root_stage.rho_s)
        self.assertAlmostEqual(root_stage.global_sum_bound, root_stage.rho_s)
        self.assertLessEqual(root_stage.rho_s, 2.0 * root_stage.delta_s_star)
        self.assertEqual(root_stage.full_history_viable_count, 2)
        self.assertEqual(root_stage.selected_policy_retained_viable_count, 0)
        self.assertEqual(root_stage.selected_policy_retained_viable_fraction, 0.0)
        self.assertEqual(
            root_stage.first_action_safety_policy_retained_viable_count, 0
        )
        self.assertEqual(
            root_stage.first_action_safety_policy_retained_viable_fraction, 0.0
        )
        self.assertEqual(root_stage.fibers_without_common_safe_action, 1)
        self.assertFalse(
            summary.stagewise_first_action_feasible_within_tolerance
        )
        self.assertFalse(summary.exact_viability_preserving_code_policy_exists)
        self.assertTrue(summary.global_sign_characterization_holds)
        self.assertEqual(len(summary.root_fiber_safety), 1)
        self.assertEqual(summary.root_fiber_safety[0].common_safe_actions, ())
        self.assertEqual(len(summary.root_fiber_safety[0].viable_histories), 2)
        self.assertTrue(
            all(
                outcome.finite_horizon_viable_under_optimal_policy
                for outcome in summary.root_outcomes
            )
        )
        self.assertTrue(
            all(
                not outcome.finite_horizon_viable_under_representation_policy
                for outcome in summary.root_outcomes
            )
        )

    def test_pendulum_memoryless_alias_has_sign_obstruction(self) -> None:
        summary = self._summary(self.pendulum_report, MEMORYLESS_PIXELS)
        root_stage = summary.stages[-1]
        self.assertEqual(summary.root_code_count, 1)
        self.assertAlmostEqual(root_stage.rho_s, 0.030506289136083886)
        self.assertAlmostEqual(root_stage.kappa_s, 0.015350554566108121)
        self.assertAlmostEqual(root_stage.delta_s_star, 0.04264522258899239)
        self.assertAlmostEqual(
            root_stage.max_realized_policy_loss, root_stage.rho_s
        )
        self.assertAlmostEqual(root_stage.max_path_bound_B, root_stage.rho_s)
        self.assertAlmostEqual(root_stage.global_sum_bound, root_stage.rho_s)
        self.assertEqual(root_stage.full_history_viable_count, 2)
        self.assertEqual(root_stage.selected_policy_retained_viable_count, 1)
        self.assertEqual(root_stage.selected_policy_retained_viable_fraction, 0.5)
        self.assertEqual(
            root_stage.first_action_safety_policy_retained_viable_count, 1
        )
        self.assertEqual(
            root_stage.first_action_safety_policy_retained_viable_fraction, 0.5
        )
        self.assertEqual(root_stage.fibers_without_common_safe_action, 1)
        self.assertFalse(
            summary.stagewise_first_action_feasible_within_tolerance
        )
        self.assertFalse(summary.exact_viability_preserving_code_policy_exists)
        self.assertTrue(summary.global_sign_characterization_holds)
        self.assertEqual(summary.root_fiber_safety[0].common_safe_actions, ())
        fast = summary.root_outcomes[0]
        self.assertGreater(fast.optimal_value, 0.0)
        self.assertLess(fast.representation_policy_value, 0.0)
        self.assertTrue(fast.finite_horizon_viable_under_optimal_policy)
        self.assertFalse(fast.finite_horizon_viable_under_representation_policy)

    def test_every_memoryless_root_action_fails_on_a_viable_history(self) -> None:
        for tree in (self.cart_tree, self.pendulum_tree):
            audit = audit_finite_dynamic_sufficiency(
                actions=tree.actions,
                histories_by_remaining=tree.histories_by_remaining,
                margins=tree.margins,
                successors=tree.successors,
                codes=tree.codes_by_representation[MEMORYLESS_PIXELS],
            )
            roots = tree.root_histories
            self.assertTrue(
                all(audit.optimal_values[tree.horizon][root] >= 0.0 for root in roots)
            )
            for action in tree.actions:
                self.assertTrue(
                    any(
                        audit.q_values[tree.horizon][(root, action)] < 0.0
                        for root in roots
                    )
                )
            stage = audit.stage_audits[-1]
            self.assertEqual(stage.viable_history_counts[0][1], 2)
            self.assertEqual(stage.common_safe_actions[0][1], ())
            self.assertGreater(stage.max_first_action_obstruction, 0.0)

    def test_reports_fail_closed_on_scope(self) -> None:
        for report in (self.cart_report, self.pendulum_report):
            self.assertFalse(report.population_certificate)
            self.assertIn("noise-free nominal tree", report.certificate_scope)
            self.assertIn("process noise is fixed to zero", report.transition_semantics)
            self.assertEqual(report.tolerance, 1e-12)
            self.assertEqual(len(report.config_sha256), 64)
            self.assertEqual(
                report.fixture_version, "two_domain_empty_common_safe_action_v1"
            )
            self.assertIn("binary floating-point", report.floating_point_semantics)
            self.assertIn("not sampled", report.training_support_status)
            self.assertTrue(
                all(root.current_margin >= 0.0 for root in report.fixture_roots)
            )
            self.assertTrue(
                all(root.previous_margin >= 0.0 for root in report.fixture_roots)
            )
        self.assertEqual(
            self.cart_report.config_sha256,
            "46997bc3fef0f3ca7f7789ff560ae90fac9732a6e9514919b8b8bf558d17d991",
        )
        self.assertEqual(
            self.pendulum_report.config_sha256,
            "575d545e0a5d5e0087f777d3a6bc44f6c6e1eb60ba697c71fd9bc9ef68caf3d9",
        )
        self.assertIn("exceed", self.cart_report.training_support_status)
        self.assertIn("no empirical", self.pendulum_report.training_support_status)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_controlled_dynamic_tree(self.cart_config, horizon=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_controlled_dynamic_tree(self.cart_config, horizon=True)
        with self.assertRaisesRegex(ValueError, "registered only"):
            build_controlled_dynamic_tree(self.cart_config, horizon=2)
        drifted = dataclasses.replace(self.cart_config, dt=0.13)
        with self.assertRaisesRegex(ValueError, "differ from the registered"):
            build_controlled_dynamic_tree(drifted, horizon=3)

    def test_public_nominal_step_rejects_off_grid_and_nonfinite_actions(self) -> None:
        state = CartState(position=0.0, velocity=0.0, nuisance_phase=0.0)
        with self.assertRaisesRegex(ValueError, "configured grid"):
            deterministic_step(state, 0.5, self.cart_config)
        with self.assertRaisesRegex(ValueError, "finite"):
            deterministic_step(state, float("nan"), self.cart_config)
        with self.assertRaisesRegex(ValueError, "real number"):
            deterministic_step(state, True, self.cart_config)
        with self.assertRaisesRegex(ValueError, "real number"):
            deterministic_step(state, "1.0", self.cart_config)


if __name__ == "__main__":
    unittest.main()
