from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning.config import (  # noqa: E402
    cpu_smoke_config,
    load_config,
    validate_config,
)
from latent_safety.learning.runtime import TorchUnavailableError, require_torch  # noqa: E402
from latent_safety.learning.trainer import (  # noqa: E402
    TRAIN_VALIDATION_ONLY_ACCESS,
    run_experiment,
)

try:
    MODULES = require_torch()
except TorchUnavailableError:
    MODULES = None

CONFIG_PATH = ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"


def _tiny_config(*, arm: str, epochs: int):
    config = cpu_smoke_config(load_config(CONFIG_PATH))
    safety_weight = 0.0 if arm == "none" else 0.1
    config = dataclasses.replace(
        config,
        run=dataclasses.replace(
            config.run,
            epochs=epochs,
            batch_size=4,
            num_workers=0,
            amp=False,
        ),
        data=dataclasses.replace(config.data, trajectories=8),
        model=dataclasses.replace(
            config.model,
            latent_dim=3,
            hidden_dim=16,
            transition_hidden_dim=16,
        ),
        objective=dataclasses.replace(
            config.objective,
            safety_arm=arm,
            safety_weight=safety_weight,
            fcsrl_head_hidden_dim=7,
        ),
        evaluation=dataclasses.replace(
            config.evaluation,
            rollout_horizons=(1,),
            max_rollout_cases=2,
            emit_audit_records=False,
            max_audit_records_per_split=4,
        ),
    )
    validate_config(config)
    return config


def _assert_nested_equal(test: unittest.TestCase, left: Any, right: Any) -> None:
    assert MODULES is not None
    torch = MODULES.torch
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(test, left_value, right_value)
    else:
        test.assertEqual(left, right)


@unittest.skipIf(MODULES is None, "PyTorch research extra is unavailable")
class FCSRLTrainerIntegrationTests(unittest.TestCase):
    def test_validation_only_fcsrl_run_emits_postfit_audit_without_test_access(self) -> None:
        config = _tiny_config(
            arm="fcsrl_feasibility_loss_adaptation",
            epochs=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "validation-only-fcsrl"
            result = run_experiment(
                config,
                config_path=CONFIG_PATH,
                output_dir=output,
                repo_root=ROOT,
                access_scope=TRAIN_VALIDATION_ONLY_ACCESS,
                orchestration_plan_sha256="b" * 64,
            )
            evaluation = json.loads(
                (output / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            postfit = json.loads(
                (output / "validation_postfit_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertNotIn("test_metrics", result)
            self.assertEqual(set(evaluation["split_metrics"]), {"train", "validation"})
            self.assertEqual(set(evaluation["rollout_metrics"]), {"validation"})
            self.assertEqual(postfit["semantic_arm"], config.objective.safety_arm)
            self.assertEqual(
                postfit["split_access"]["not_materialized"],
                ["calibration", "test"],
            )
            self.assertTrue((output / "audit_validation.jsonl").is_file())

    def test_fcsrl_run_emits_complete_artifact_and_exact_resume(self) -> None:
        assert MODULES is not None
        torch = MODULES.torch
        one_epoch = _tiny_config(
            arm="fcsrl_feasibility_loss_adaptation",
            epochs=1,
        )
        two_epochs = dataclasses.replace(
            one_epoch,
            run=dataclasses.replace(one_epoch.run, epochs=2),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_dir = root / "first"
            resumed_dir = root / "resumed"
            uninterrupted_dir = root / "uninterrupted"
            first = run_experiment(
                one_epoch,
                config_path=CONFIG_PATH,
                output_dir=first_dir,
                repo_root=ROOT,
            )
            resumed = run_experiment(
                two_epochs,
                config_path=CONFIG_PATH,
                output_dir=resumed_dir,
                repo_root=ROOT,
                resume_from=first_dir / "checkpoint_last.pt",
            )
            uninterrupted = run_experiment(
                two_epochs,
                config_path=CONFIG_PATH,
                output_dir=uninterrupted_dir,
                repo_root=ROOT,
            )

            self.assertEqual(first["status"], "success")
            self.assertEqual(resumed["status"], "success")
            self.assertEqual(uninterrupted["status"], "success")
            artifact = json.loads(
                (first_dir / "fcsrl_regression_fixture.json").read_text(
                    encoding="utf-8"
                )
            )
            uninterrupted_artifact = json.loads(
                (uninterrupted_dir / "fcsrl_regression_fixture.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(artifact, uninterrupted_artifact)
            self.assertEqual(
                artifact["status"],
                "engineering_regression_not_learned_model_evidence",
            )
            self.assertEqual(artifact["protocol"]["head_hidden_dim"], 7)
            self.assertEqual(artifact["fixture"]["mask"], [True] * 10)
            self.assertEqual(artifact["fixture"]["train_mask"], [True] * 4)
            self.assertEqual(len(artifact["fixture"]["targets"]), 10)
            self.assertEqual(len(artifact["fixture"]["projected_targets"]), 10)
            for projection in artifact["fixture"]["projected_targets"]:
                self.assertEqual(len(projection), 63)
                self.assertAlmostEqual(sum(projection), 1.0, places=6)
            self.assertTrue(artifact["ema"]["matches_online_at_fixture_time"])
            self.assertEqual(len(artifact["ema"]["latents"]), 10)

            first_checkpoint = torch.load(
                first_dir / "checkpoint_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertIsInstance(first_checkpoint["fcsrl_state"], dict)
            self.assertEqual(
                first_checkpoint["fcsrl_state"]["protocol"]["head_hidden_dim"],
                7,
            )
            self.assertEqual(
                set(first_checkpoint["training_state"]),
                {
                    "python_random_state",
                    "torch_rng_state",
                    "cuda_rng_state_all",
                    "train_loader_generator_state",
                    "scaler_state",
                },
            )
            checkpoint_manifest = json.loads(
                (first_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                len(checkpoint_manifest["fcsrl_regression_fixture"]["sha256"]),
                64,
            )
            self.assertIsNotNone(
                checkpoint_manifest["state_hashes"]["selected_fcsrl_head_state"]
            )

            resumed_checkpoint = torch.load(
                resumed_dir / "checkpoint_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            uninterrupted_checkpoint = torch.load(
                uninterrupted_dir / "checkpoint_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            for key in (
                "model_state",
                "optimizer_state",
                "training_state",
                "fcsrl_state",
                "validation",
            ):
                _assert_nested_equal(
                    self,
                    resumed_checkpoint[key],
                    uninterrupted_checkpoint[key],
                )
            resumed_history = [
                json.loads(line)
                for line in (resumed_dir / "history.jsonl").read_text().splitlines()
            ]
            full_history = [
                json.loads(line)
                for line in (uninterrupted_dir / "history.jsonl").read_text().splitlines()
            ]
            self.assertEqual(resumed_history, [full_history[1]])
            resumed_manifest = json.loads(
                (resumed_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_manifest["parent_checkpoint"]["resumed_epoch"], 1)

    def test_none_and_margin_paths_do_not_construct_fcsrl_state(self) -> None:
        assert MODULES is not None
        torch = MODULES.torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in ("none", "h_prediction"):
                with self.subTest(arm=arm):
                    output = root / arm
                    result = run_experiment(
                        _tiny_config(arm=arm, epochs=1),
                        config_path=CONFIG_PATH,
                        output_dir=output,
                        repo_root=ROOT,
                    )
                    self.assertEqual(result["status"], "success")
                    self.assertIsNone(result["fcsrl_regression_fixture"])
                    self.assertFalse((output / "fcsrl_regression_fixture.json").exists())
                    checkpoint = torch.load(
                        output / "checkpoint_last.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    self.assertIsNone(checkpoint["fcsrl_state"])
                    run_manifest = json.loads(
                        (output / "run_manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertIsNone(run_manifest["fcsrl"])

    def test_resume_rejects_any_non_epoch_config_change(self) -> None:
        one_epoch = _tiny_config(
            arm="fcsrl_feasibility_loss_adaptation",
            epochs=1,
        )
        changed = dataclasses.replace(
            one_epoch,
            run=dataclasses.replace(one_epoch.run, epochs=2),
            objective=dataclasses.replace(
                one_epoch.objective,
                fcsrl_head_hidden_dim=8,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_dir = root / "first"
            run_experiment(
                one_epoch,
                config_path=CONFIG_PATH,
                output_dir=first_dir,
                repo_root=ROOT,
            )
            with self.assertRaisesRegex(ValueError, "beyond run.epochs"):
                run_experiment(
                    changed,
                    config_path=CONFIG_PATH,
                    output_dir=root / "invalid-resume",
                    repo_root=ROOT,
                    resume_from=first_dir / "checkpoint_last.pt",
                )


if __name__ == "__main__":
    unittest.main()
