from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning import (  # noqa: E402
    DubinsState,
    action_safety_profile,
    build_dataset_manifest,
    cpu_smoke_config,
    deterministic_step,
    dry_run_plan,
    generate_trajectories,
    load_config,
    observation_feature_vector,
    render_state,
    safety_margin,
    state_feature_vector,
)


CONFIG_PATH = ROOT / "configs" / "e1_world_models" / "torch_dubins_pilot.toml"


class ControlledDubinsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_plan_and_cpu_smoke_are_task_explicit(self) -> None:
        plan = dry_run_plan(self.config)
        self.assertEqual(plan["task"], "controlled_dubins_navigation_pixels")
        self.assertEqual(plan["total_samples"], 320 * 48)
        smoke = cpu_smoke_config(self.config)
        self.assertEqual(smoke.run.output_dir, "runs/e1_world_models/torch_dubins_cpu_smoke")
        self.assertEqual(smoke.data.history_length, 2)

    def test_generation_labels_and_manifest_are_deterministic(self) -> None:
        first = generate_trajectories(self.config.data, seed=23)
        second = generate_trajectories(self.config.data, seed=23)
        self.assertEqual(first, second)
        self.assertTrue(all(item.trajectory_id.startswith("dubins-") for item in first))
        self.assertEqual(
            {item.split for item in first},
            {"train", "validation", "calibration", "test"},
        )
        self.assertTrue(
            all(isinstance(state, DubinsState) for item in first for state in item.states)
        )
        self.assertTrue(
            any(
                max(profile) - min(profile) > 1e-5
                for item in first
                for profile in item.action_safety_margins
            )
        )
        manifest, digest = build_dataset_manifest(first, self.config.data, 23)
        self.assertEqual(manifest["generator"], "controlled_dubins_navigation_pixels_v1")
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            manifest["state_fields"],
            [
                "x",
                "y",
                "heading",
                "obstacle_phase",
                "obstacle_direction",
                "nuisance_phase",
            ],
        )
        self.assertTrue(
            manifest["label_semantics"]["action_profile"][
                "uses_hidden_obstacle_velocity"
            ]
        )

    def test_single_frame_aliases_opposite_obstacle_motion(self) -> None:
        clockwise = DubinsState(
            x=0.0,
            y=-0.10,
            heading=0.0,
            obstacle_phase=0.4,
            obstacle_direction=-1,
            nuisance_phase=0.2,
        )
        counterclockwise = dataclasses.replace(clockwise, obstacle_direction=1)
        self.assertEqual(
            render_state(clockwise, self.config.data),
            render_state(counterclockwise, self.config.data),
        )
        h1 = dataclasses.replace(self.config.data, history_length=1)
        self.assertEqual(
            observation_feature_vector((clockwise,), 0, h1),
            observation_feature_vector((counterclockwise,), 0, h1),
        )
        self.assertNotEqual(
            state_feature_vector(clockwise, self.config.data),
            state_feature_vector(counterclockwise, self.config.data),
        )

        clockwise_next = deterministic_step(clockwise, 0.0, self.config.data)
        counterclockwise_next = deterministic_step(
            counterclockwise, 0.0, self.config.data
        )
        h2 = dataclasses.replace(self.config.data, history_length=2)
        self.assertNotEqual(
            observation_feature_vector((clockwise, clockwise_next), 1, h2),
            observation_feature_vector(
                (counterclockwise, counterclockwise_next), 1, h2
            ),
        )

    def test_margin_and_action_profile_match_declared_geometry(self) -> None:
        state = DubinsState(
            x=0.95,
            y=0.0,
            heading=0.0,
            obstacle_phase=math.pi,
            obstacle_direction=1,
            nuisance_phase=0.0,
        )
        self.assertAlmostEqual(safety_margin(state, self.config.data), 0.25)
        profile = action_safety_profile(state, self.config.data)
        self.assertEqual(len(profile), len(self.config.data.actions))
        self.assertTrue(all(math.isfinite(value) for value in profile))
        with self.assertRaisesRegex(ValueError, "obstacle_direction"):
            deterministic_step(
                dataclasses.replace(state, obstacle_direction=0),
                0.0,
                self.config.data,
            )

    def test_render_shape_and_range(self) -> None:
        trajectory = generate_trajectories(self.config.data, seed=7)[0]
        pixels = render_state(trajectory.states[0], self.config.data)
        self.assertEqual(
            len(pixels),
            self.config.data.channels
            * self.config.data.image_size
            * self.config.data.image_size,
        )
        self.assertGreaterEqual(min(pixels), 0.0)
        self.assertLessEqual(max(pixels), 1.0)

    def test_generator_quality_gates_cover_splits_scenarios_and_boundary(self) -> None:
        trajectories = generate_trajectories(self.config.data, seed=0)
        ids = [trajectory.trajectory_id for trajectory in trajectories]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            Counter(trajectory.split for trajectory in trajectories),
            Counter({"train": 208, "validation": 48, "calibration": 32, "test": 32}),
        )
        split_scenarios = {
            split: {
                trajectory.scenario
                for trajectory in trajectories
                if trajectory.split == split
            }
            for split in ("train", "validation", "calibration", "test")
        }
        self.assertTrue(
            all(
                scenarios
                == {"nominal", "boundary_challenge", "moving_obstacle_challenge"}
                for scenarios in split_scenarios.values()
            )
        )

        margins = [
            margin for trajectory in trajectories for margin in trajectory.safety_margins
        ]
        profiles = [
            profile
            for trajectory in trajectories
            for profile in trajectory.action_safety_margins
        ]
        unsafe_fraction = sum(margin < 0.0 for margin in margins) / len(margins)
        boundary_fraction = sum(
            abs(margin) <= self.config.objective.boundary_band for margin in margins
        ) / len(margins)
        profile_variation_fraction = sum(
            max(profile) - min(profile) > 1e-6 for profile in profiles
        ) / len(profiles)
        self.assertGreater(unsafe_fraction, 0.10)
        self.assertLess(unsafe_fraction, 0.60)
        self.assertGreater(boundary_fraction, 0.10)
        self.assertGreater(profile_variation_fraction, 0.50)

        action_counts = Counter(
            action for trajectory in trajectories for action in trajectory.actions
        )
        self.assertEqual(set(action_counts), set(self.config.data.actions))
        self.assertTrue(
            all(count / sum(action_counts.values()) > 0.10 for count in action_counts.values())
        )


if __name__ == "__main__":
    unittest.main()
