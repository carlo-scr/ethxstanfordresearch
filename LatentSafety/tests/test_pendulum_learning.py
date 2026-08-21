from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning import (  # noqa: E402
    CartState,
    ConfigError,
    PendulumState,
    action_safety_profile,
    build_dataset_manifest,
    cpu_smoke_config,
    dry_run_plan,
    generate_trajectories,
    load_config,
    observation_feature_vector,
    render_state,
    safety_margin,
    state_feature_vector,
)
from latent_safety.learning.config import validate_config  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "e1_world_models" / "torch_pendulum_pilot.toml"
GRID_PATH = ROOT / "configs" / "e1_world_models" / "pilot_pendulum_grid.toml"
CART_CONFIG_PATH = ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"


class ControlledPendulumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.cart_config = load_config(CART_CONFIG_PATH)

    def test_dependency_free_plan_and_cpu_smoke_are_task_explicit(self) -> None:
        plan = dry_run_plan(self.config)
        self.assertEqual(plan["task"], "controlled_pendulum_video")
        self.assertEqual(plan["total_samples"], 320 * 48)
        self.assertFalse(plan["torch_required"])

        smoke = cpu_smoke_config(self.config)
        self.assertEqual(smoke.data.task, "controlled_pendulum_video")
        self.assertEqual(smoke.run.device, "cpu")
        self.assertEqual(smoke.run.epochs, 1)
        self.assertEqual(smoke.run.output_dir, "runs/e1_world_models/torch_pendulum_cpu_smoke")

    def test_task_specific_validation_rejects_invalid_angular_domain(self) -> None:
        with self.assertRaisesRegex(ConfigError, "position_limit < pi"):
            validate_config(
                dataclasses.replace(
                    self.config,
                    data=dataclasses.replace(
                        self.config.data,
                        position_limit=math.pi,
                        render_extent=math.pi + 0.1,
                    ),
                )
            )
        with self.assertRaisesRegex(ConfigError, "negative and positive torques"):
            validate_config(
                dataclasses.replace(
                    self.config,
                    data=dataclasses.replace(self.config.data, actions=(0.0, 1.0)),
                )
            )

    def test_separate_pendulum_grid_expands_and_dry_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "pendulum_plan.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plan_e1_sweep.py"),
                    "--grid",
                    str(GRID_PATH),
                    "--output",
                    str(plan_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["task_count"], 144)
            self.assertEqual(len({task["output_dir"] for task in plan["tasks"]}), 144)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(plan_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            dry_run = json.loads(completed.stdout)
            self.assertEqual(dry_run["task"], "controlled_pendulum_video")
            self.assertEqual(dry_run["representation"]["history_mode"], "stack_h1")

    def test_generation_labels_and_manifest_are_deterministic(self) -> None:
        first = generate_trajectories(self.config.data, seed=19)
        second = generate_trajectories(self.config.data, seed=19)
        self.assertEqual(first, second)
        self.assertTrue(all(item.trajectory_id.startswith("pendulum-") for item in first))
        self.assertEqual(
            {item.split for item in first},
            {"train", "validation", "calibration", "test"},
        )
        self.assertTrue(
            all(isinstance(state, PendulumState) for item in first for state in item.states)
        )
        trajectory = first[0]
        self.assertEqual(len(trajectory.states), self.config.data.horizon + 1)
        self.assertEqual(len(trajectory.actions), self.config.data.horizon)
        self.assertTrue(
            all(
                math.isclose(margin, safety_margin(state, self.config.data))
                for margin, state in zip(trajectory.safety_margins, trajectory.states)
            )
        )
        self.assertTrue(
            any(
                max(profile) - min(profile) > 0.01
                for item in first
                for profile in item.action_safety_margins
            )
        )

        manifest_a, digest_a = build_dataset_manifest(first, self.config.data, 19)
        manifest_b, digest_b = build_dataset_manifest(second, self.config.data, 19)
        self.assertEqual((manifest_a, digest_a), (manifest_b, digest_b))
        self.assertEqual(manifest_a["generator"], "controlled_pendulum_video_v1")
        self.assertEqual(manifest_a["domain_kind"], "synthetic_known_dynamics_not_gym")
        self.assertEqual(
            manifest_a["state_fields"],
            ["angle", "angular_velocity", "nuisance_phase"],
        )
        self.assertTrue(
            manifest_a["label_semantics"]["action_profile"]["uses_hidden_velocity"]
        )

    def test_pixels_hide_angular_velocity_but_oracle_profile_uses_it(self) -> None:
        outward = PendulumState(angle=0.55, angular_velocity=1.2, nuisance_phase=0.3)
        inward = PendulumState(angle=0.55, angular_velocity=-1.2, nuisance_phase=0.3)
        outward_pixels = render_state(outward, self.config.data)
        inward_pixels = render_state(inward, self.config.data)
        self.assertEqual(outward_pixels, inward_pixels)
        self.assertNotEqual(
            action_safety_profile(outward, self.config.data),
            action_safety_profile(inward, self.config.data),
        )
        self.assertEqual(
            observation_feature_vector((outward,), 0, self.config.data),
            observation_feature_vector((inward,), 0, self.config.data),
        )
        self.assertNotEqual(
            state_feature_vector(outward, self.config.data),
            state_feature_vector(inward, self.config.data),
        )
        self.assertEqual(len(state_feature_vector(outward, self.config.data)), 3)
        self.assertEqual(
            len(outward_pixels),
            self.config.data.channels
            * self.config.data.image_size
            * self.config.data.image_size,
        )
        self.assertGreaterEqual(min(outward_pixels), 0.0)
        self.assertLessEqual(max(outward_pixels), 1.0)

    def test_one_step_action_profile_matches_known_dynamics_formula(self) -> None:
        data = dataclasses.replace(self.config.data, action_profile_horizon=1)
        state = PendulumState(angle=0.55, angular_velocity=0.40, nuisance_phase=0.0)
        observed = action_safety_profile(state, data)
        expected = []
        initial_margin = data.position_limit - abs(state.angle)
        for action in data.actions:
            next_velocity = data.damping * state.angular_velocity + data.dt * (
                math.sin(state.angle) + data.acceleration * action
            )
            next_angle = (state.angle + data.dt * next_velocity + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            expected.append(
                min(initial_margin, data.position_limit - abs(next_angle))
            )
        self.assertEqual(len(observed), len(expected))
        self.assertTrue(
            all(
                math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12)
                for actual, target in zip(observed, expected, strict=True)
            )
        )

    def test_observation_oracle_h1_aliases_and_history_disambiguates(self) -> None:
        path_a = (
            PendulumState(0.40, -0.5, 0.0),
            PendulumState(0.30, -0.5, 0.1),
            PendulumState(0.20, -0.5, 0.2),
            PendulumState(0.10, -0.5, 0.3),
        )
        path_b = (
            PendulumState(-0.20, 0.5, 1.0),
            PendulumState(-0.10, 0.5, 1.1),
            PendulumState(0.00, 0.5, 1.2),
            PendulumState(0.10, 0.5, 1.3),
        )
        h1 = dataclasses.replace(self.config.data, history_length=1)
        h4 = dataclasses.replace(self.config.data, history_length=4)
        self.assertEqual(
            observation_feature_vector(path_a, 3, h1),
            observation_feature_vector(path_b, 3, h1),
        )
        self.assertNotEqual(
            observation_feature_vector(path_a, 3, h4),
            observation_feature_vector(path_b, 3, h4),
        )
        self.assertEqual(len(observation_feature_vector(path_a, 3, h1)), 2)
        self.assertEqual(len(observation_feature_vector(path_a, 3, h4)), 8)

    def test_cart_observation_oracle_has_the_same_history_semantics(self) -> None:
        path_a = (
            CartState(-0.30, 0.5, 0.0),
            CartState(-0.20, 0.5, 0.1),
            CartState(-0.10, 0.5, 0.2),
            CartState(0.00, 0.5, 0.3),
        )
        path_b = (
            CartState(0.30, -0.5, 1.0),
            CartState(0.20, -0.5, 1.1),
            CartState(0.10, -0.5, 1.2),
            CartState(0.00, -0.5, 1.3),
        )
        h1 = dataclasses.replace(self.cart_config.data, history_length=1)
        h4 = dataclasses.replace(self.cart_config.data, history_length=4)
        self.assertEqual(
            observation_feature_vector(path_a, 3, h1),
            observation_feature_vector(path_b, 3, h1),
        )
        self.assertNotEqual(
            observation_feature_vector(path_a, 3, h4),
            observation_feature_vector(path_b, 3, h4),
        )
        self.assertEqual(len(observation_feature_vector(path_a, 3, h1)), 1)
        self.assertEqual(len(observation_feature_vector(path_a, 3, h4)), 4)
        self.assertEqual(
            state_feature_vector(path_a[-1], self.cart_config.data),
            (0.0, 0.5),
        )


if __name__ == "__main__":
    unittest.main()
