from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning import (  # noqa: E402
    ConfigError,
    cpu_smoke_config,
    dry_run_plan,
    load_config,
    parse_config,
)
from latent_safety.learning.data import generate_trajectories, render_state  # noqa: E402
from latent_safety.learning.models import model_metadata  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"


def _payload() -> dict[str, object]:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


class LearningConfigTests(unittest.TestCase):
    def test_model_metadata_names_joint_history_mode(self) -> None:
        class EmptyModel:
            @staticmethod
            def parameters() -> tuple[object, ...]:
                return ()

        config = load_config(CONFIG_PATH)
        metadata = model_metadata(EmptyModel(), config.model, config.data)
        self.assertEqual(metadata["history_mode"], "gru_h4")
        self.assertEqual(metadata["history_encoder"], "gru")
        self.assertEqual(metadata["history_length"], 4)

    def test_pilot_plan_is_dependency_free_and_trajectory_split(self) -> None:
        config = load_config(CONFIG_PATH)
        plan = dry_run_plan(config)
        self.assertFalse(plan["torch_required"])
        self.assertEqual(plan["trajectory_counts"], {
            "train": 208,
            "validation": 48,
            "calibration": 32,
            "test": 32,
        })
        self.assertEqual(plan["total_samples"], 320 * 40)
        self.assertEqual(config.model.family, "beta_vae")
        self.assertEqual(config.model.history_encoder, "gru")
        self.assertEqual(config.objective.safety_arm, "safe_action_profile")

    def test_rejects_non_partitioning_split_fractions(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["data"]["splits"]["test"] = 0.2  # type: ignore[index]
        with self.assertRaisesRegex(ConfigError, "must sum to one"):
            parse_config(payload)

    def test_rejects_safety_weight_in_unsupervised_baseline(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["objective"]["safety_arm"] = "none"  # type: ignore[index]
        with self.assertRaisesRegex(ConfigError, "none arm"):
            parse_config(payload)

    def test_rejects_infeasible_normalized_contrastive_margin(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["objective"]["contrastive_margin"] = 2.01  # type: ignore[index]
        with self.assertRaisesRegex(ConfigError, "cannot exceed two"):
            parse_config(payload)

    def test_all_preregistered_safety_arms_and_model_families_parse(self) -> None:
        for family in ("ae", "beta_vae"):
            for arm in (
                "none",
                "h_prediction",
                "boundary_contrastive",
                "safe_action_profile",
            ):
                with self.subTest(family=family, arm=arm):
                    payload = copy.deepcopy(_payload())
                    payload["model"]["family"] = family  # type: ignore[index]
                    payload["objective"]["kl_weight"] = (  # type: ignore[index]
                        0.0 if family == "ae" else 0.0005
                    )
                    payload["objective"]["safety_arm"] = arm  # type: ignore[index]
                    payload["objective"]["safety_weight"] = (  # type: ignore[index]
                        0.0 if arm == "none" else 0.5
                    )
                    config = parse_config(payload)
                    self.assertEqual(config.model.family, family)
                    self.assertEqual(config.objective.safety_arm, arm)

    def test_resolved_checkpoint_config_round_trips(self) -> None:
        config = cpu_smoke_config(load_config(CONFIG_PATH))
        self.assertEqual(parse_config(config.to_dict()), config)

    def test_importing_planner_does_not_import_torch(self) -> None:
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(ROOT / "src") if not existing else f"{ROOT / 'src'}{os.pathsep}{existing}"
        )
        command = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import latent_safety.learning; "
                "import latent_safety.learning.cli; "
                "import latent_safety.learning.trainer; "
                "print(json.dumps({'torch_loaded': 'torch' in sys.modules}))"
            ),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), {"torch_loaded": False})

    def test_cli_dry_run_uses_the_real_config(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_e1_torch.py"),
                "--config",
                str(CONFIG_PATH),
                "--dry-run",
                "--device",
                "cpu",
                "--seed",
                "17",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["requested_device"], "cpu")
        self.assertFalse(plan["torch_required"])
        self.assertEqual(plan["selected_arm"], "safe_action_profile")

    def test_cli_dry_run_accepts_factorial_model_overrides(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_e1_torch.py"),
                "--config",
                str(CONFIG_PATH),
                "--dry-run",
                "--model-family",
                "ae",
                "--history-encoder",
                "stack",
                "--history-length",
                "1",
                "--latent-dim",
                "4",
                "--safety-arm",
                "none",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(completed.stdout)
        self.assertEqual(
            plan["representation"],
            {
                "family": "ae",
                "history_mode": "stack_h1",
                "history_encoder": "stack",
                "history_length": 1,
                "latent_dim": 4,
            },
        )
        self.assertEqual(plan["selected_arm"], "none")

    def test_cpu_smoke_preserves_real_split_and_objective(self) -> None:
        smoke = cpu_smoke_config(load_config(CONFIG_PATH))
        plan = dry_run_plan(smoke)
        self.assertEqual(smoke.run.device, "cpu")
        self.assertEqual(smoke.run.epochs, 1)
        self.assertEqual(smoke.run.num_workers, 0)
        self.assertEqual(sum(plan["trajectory_counts"].values()), 16)
        self.assertTrue(all(value > 0 for value in plan["trajectory_counts"].values()))
        self.assertEqual(plan["selected_arm"], "safe_action_profile")

    def test_controlled_video_fixture_is_deterministic_and_privileged(self) -> None:
        config = cpu_smoke_config(load_config(CONFIG_PATH))
        first = generate_trajectories(config.data, seed=config.run.seed)
        second = generate_trajectories(config.data, seed=config.run.seed)
        self.assertEqual(first, second)
        self.assertEqual(len(first), config.data.trajectories)
        self.assertEqual({trajectory.split for trajectory in first}, {
            "train",
            "validation",
            "calibration",
            "test",
        })
        trajectory = first[0]
        self.assertEqual(len(trajectory.states), config.data.horizon + 1)
        self.assertTrue(
            any(max(profile) - min(profile) > 0.01 for profile in trajectory.action_safety_margins)
        )
        pixels = render_state(trajectory.states[0], config.data)
        self.assertEqual(
            len(pixels),
            config.data.channels * config.data.image_size * config.data.image_size,
        )
        self.assertGreaterEqual(min(pixels), 0.0)
        self.assertLessEqual(max(pixels), 1.0)


if __name__ == "__main__":
    unittest.main()
