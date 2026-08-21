from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_safety.benchmarks import (  # noqa: E402
    ControlledDoubleIntegrator,
    DoubleIntegratorState,
    ObservationSpec,
    RolloutSpec,
    build_dataset,
    make_action_conflict_pair,
    make_static_aliasing_pair,
    reconstruct_current_velocity,
    rollout,
)
from latent_safety.metrics.action import audit_exact_action_fibers  # noqa: E402
from latent_safety.metrics.defect import empirical_robust_defect  # noqa: E402


class DoubleIntegratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = ControlledDoubleIntegrator(
            time_step=0.5,
            position_limit=2.0,
            action_grid=(-2.0, 0.0, 2.0),
        )

    def test_exact_step_and_predecessor(self) -> None:
        state = DoubleIntegratorState(position=0.25, velocity=1.0)
        next_state = self.system.step(state, action=2.0)
        self.assertEqual(next_state, DoubleIntegratorState(position=1.0, velocity=2.0))
        self.assertEqual(self.system.predecessor(next_state, previous_action=2.0), state)

    def test_action_margins_are_analytic_next_position_margins(self) -> None:
        state = DoubleIntegratorState(position=1.25, velocity=1.0)
        margins = self.system.one_step_action_margins(state)
        expected = tuple(
            self.system.position_safety_margin(self.system.step(state, action))
            for action in self.system.action_grid
        )
        self.assertEqual(margins, expected)
        self.assertEqual(margins, (0.5, 0.25, 0.0))
        self.assertEqual(self.system.one_step_safe_actions(state), self.system.action_grid)

    def test_off_grid_action_is_not_silently_projected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite action grid"):
            self.system.step(DoubleIntegratorState(0.0, 0.0), action=0.1)

    def test_invalid_system_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ControlledDoubleIntegrator(action_grid=(0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "time_step must be positive"):
            ControlledDoubleIntegrator(time_step=0.0)


class ObservationTests(unittest.TestCase):
    def test_position_only_hides_velocity_but_full_state_does_not(self) -> None:
        fast = DoubleIntegratorState(position=0.25, velocity=3.0)
        slow = DoubleIntegratorState(position=0.25, velocity=-1.0)
        self.assertEqual(
            ObservationSpec.position_only().observe((fast,)),
            ObservationSpec.position_only().observe((slow,)),
        )
        self.assertNotEqual(
            ObservationSpec.full_state().observe((fast,)),
            ObservationSpec.full_state().observe((slow,)),
        )

    def test_saturated_position_creates_exact_alias(self) -> None:
        observer = ObservationSpec.saturated_position(sensor_limit=0.5)
        self.assertEqual(
            observer.observe((DoubleIntegratorState(0.5, 0.0),)),
            observer.observe((DoubleIntegratorState(4.0, 10.0),)),
        )
        self.assertEqual(observer.observe((DoubleIntegratorState(-4.0, 0.0),)), (-0.5,))

    def test_two_position_action_history_recovers_velocity_and_breaks_alias(self) -> None:
        pair = make_action_conflict_pair()
        previous_action = 0.0
        left_previous = pair.system.predecessor(pair.left_state, previous_action)
        right_previous = pair.system.predecessor(pair.right_state, previous_action)
        observer = ObservationSpec.position_action_history(history_length=2)

        left_observation = observer.observe(
            (left_previous, pair.left_state), (previous_action,)
        )
        right_observation = observer.observe(
            (right_previous, pair.right_state), (previous_action,)
        )
        self.assertNotEqual(left_observation, right_observation)
        self.assertEqual(observer.dimension, 4)
        self.assertAlmostEqual(
            reconstruct_current_velocity(
                previous_position=left_previous.position,
                current_position=pair.left_state.position,
                previous_action=previous_action,
                time_step=pair.system.time_step,
            ),
            pair.left_state.velocity,
        )
        self.assertAlmostEqual(
            reconstruct_current_velocity(
                previous_position=right_previous.position,
                current_position=pair.right_state.position,
                previous_action=previous_action,
                time_step=pair.system.time_step,
            ),
            pair.right_state.velocity,
        )

    def test_history_padding_is_fixed_width_and_explicit(self) -> None:
        observer = ObservationSpec.position_action_history(history_length=3)
        initial = DoubleIntegratorState(position=0.75, velocity=2.0)
        self.assertEqual(observer.observe((initial,)), (0.75, 0.75, 0.75, 0.0, 0.0, 1.0))
        self.assertEqual(len(observer.observe((initial,))), observer.dimension)


class GroundTruthFixtureTests(unittest.TestCase):
    def test_static_alias_has_known_exact_collision_witness(self) -> None:
        pair = make_static_aliasing_pair()
        self.assertEqual(pair.left_observation, pair.right_observation)
        self.assertEqual(pair.safety_margins, (0.5, -0.5))
        self.assertTrue(pair.has_static_safe_unsafe_collision)
        self.assertEqual(pair.static_witness_margin, 0.5)
        self.assertFalse(pair.has_one_step_action_conflict)
        self.assertEqual(pair.common_one_step_safe_actions, (-2.0,))

        estimate = empirical_robust_defect(
            (pair.left_observation, pair.right_observation),
            pair.safety_margins,
            delta=0.0,
        )
        self.assertEqual(estimate.witness_margin, pair.static_witness_margin)
        self.assertEqual(estimate.confounded_safe_count, 1)

    def test_hidden_velocity_pair_has_known_finite_grid_action_conflict(self) -> None:
        pair = make_action_conflict_pair()
        self.assertEqual(pair.left_observation, pair.right_observation)
        self.assertEqual(pair.safety_margins, (1.0, 1.0))
        self.assertFalse(pair.has_static_safe_unsafe_collision)
        self.assertTrue(pair.individually_one_step_viable)
        self.assertEqual(pair.common_one_step_safe_actions, ())
        self.assertEqual(
            pair.action_safety_margins,
            ((0.75, -0.25, -1.25), (-1.25, -0.25, 0.75)),
        )
        self.assertEqual(pair.best_common_one_step_margin, -0.25)
        self.assertEqual(pair.required_one_step_violation, 0.25)

        audit = audit_exact_action_fibers(
            (pair.left_observation, pair.right_observation),
            pair.action_safety_margins,
        )
        self.assertEqual(audit.conflicting_fiber_count, 1)
        self.assertEqual(audit.worst_required_violation, pair.required_one_step_violation)


class DeterministicRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = ControlledDoubleIntegrator(
            time_step=1.0,
            position_limit=2.0,
            action_grid=(-1.0, 0.0, 1.0),
        )
        self.spec = RolloutSpec(
            trajectory_id="trajectory-a",
            initial_state=DoubleIntegratorState(position=0.0, velocity=0.0),
            actions=(1.0, 0.0, -1.0),
            split="test",
        )

    def test_rollout_is_repeatable_and_records_exact_labels(self) -> None:
        observer = ObservationSpec.position_action_history(history_length=2)
        first = rollout(self.system, self.spec, observation_spec=observer)
        second = rollout(self.system, self.spec, observation_spec=observer)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(record.sample_id for record in first),
            (
                "trajectory-a:000000",
                "trajectory-a:000001",
                "trajectory-a:000002",
            ),
        )
        self.assertEqual(first[0].observation, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(first[1].observation, (0.0, 0.5, 1.0, 2.0))
        for record in first:
            chosen_index = record.action_grid.index(record.action)
            self.assertEqual(
                record.next_safety_margin,
                record.action_safety_margins[chosen_index],
            )
            self.assertEqual(
                record.next_safety_margin,
                self.system.position_safety_margin(record.next_state),
            )

    def test_dataset_preserves_declared_trajectory_order_and_split(self) -> None:
        second_spec = RolloutSpec(
            trajectory_id="trajectory-b",
            initial_state=DoubleIntegratorState(position=1.0, velocity=0.0),
            actions=(-1.0,),
            split="validation",
        )
        records = build_dataset(
            self.system,
            (self.spec, second_spec),
            observation_spec=ObservationSpec.position_only(),
        )
        self.assertEqual(
            tuple(record.trajectory_id for record in records),
            ("trajectory-a", "trajectory-a", "trajectory-a", "trajectory-b"),
        )
        self.assertEqual(tuple(record.split for record in records), ("test",) * 3 + ("validation",))
        self.assertEqual(records[-1].observation, (1.0,))

    def test_dataset_rejects_duplicate_trajectory_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "trajectory_id values must be unique"):
            build_dataset(self.system, (self.spec, self.spec))


if __name__ == "__main__":
    unittest.main()
