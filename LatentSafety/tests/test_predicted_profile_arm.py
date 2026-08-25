from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning.config import (  # noqa: E402
    cpu_smoke_config,
    load_config,
    validate_config,
)
from latent_safety.learning.data import build_datasets  # noqa: E402
from latent_safety.learning.losses import compute_losses  # noqa: E402
from latent_safety.learning.runtime import TorchUnavailableError, require_torch  # noqa: E402
from latent_safety.learning.trainer import (  # noqa: E402
    FULL_EVALUATION_ACCESS,
    TRAIN_VALIDATION_ONLY_ACCESS,
    experiment_access_scope_payload,
    run_experiment,
)

try:
    MODULES = require_torch()
except TorchUnavailableError:
    MODULES = None

CONFIG_PATH = ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"


def _tiny_config(*, arm: str, epochs: int = 1):
    config = cpu_smoke_config(load_config(CONFIG_PATH))
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
            family="ae",
            latent_dim=3,
            hidden_dim=16,
            transition_hidden_dim=16,
        ),
        objective=dataclasses.replace(
            config.objective,
            safety_arm=arm,
            safety_weight=0.0 if arm == "none" else 0.1,
            kl_weight=0.0,
        ),
        evaluation=dataclasses.replace(
            config.evaluation,
            rollout_horizons=(1,),
            max_rollout_cases=2,
            emit_audit_records=True,
            max_audit_records_per_split=4,
        ),
    )
    validate_config(config)
    return config


@unittest.skipIf(MODULES is None, "PyTorch research extra is unavailable")
class PredictedProfileLossTests(unittest.TestCase):
    def _fixture(self):
        assert MODULES is not None
        torch = MODULES.torch
        history = torch.zeros(2, 4, 3, 4, 4)
        outputs = {
            "reconstruction": torch.zeros(2, 3, 4, 4),
            "predicted_next_z": torch.zeros(2, 3),
            "z": torch.zeros(2, 3, requires_grad=True),
            "is_variational": False,
            "predicted_margin": torch.tensor([0.1, -0.1]),
            "predicted_action_profile": torch.tensor(
                [[0.2, -0.1, 0.3], [0.0, 0.1, -0.2]]
            ),
        }
        target = torch.ones(2, 3)
        profile = torch.tensor([[0.1, -0.2, 0.4], [0.1, 0.0, -0.1]])
        batch = {
            "history": history,
            "safety_margin": torch.tensor([0.05, -0.05]),
        }
        return outputs, target, profile, batch

    def test_predicted_target_matches_existing_profile_loss_without_key_aliasing(
        self,
    ) -> None:
        assert MODULES is not None
        base = load_config(CONFIG_PATH).objective
        outputs, target, profile, batch = self._fixture()
        oracle_config = dataclasses.replace(base, safety_arm="safe_action_profile")
        predicted_config = dataclasses.replace(
            base,
            safety_arm="nonprivileged_predicted_action_profile",
        )
        oracle = compute_losses(
            MODULES,
            outputs,
            target,
            {**batch, "action_safety_margins": profile},
            oracle_config,
        )
        predicted = compute_losses(
            MODULES,
            outputs,
            target,
            {**batch, "predicted_action_profile_target": profile},
            predicted_config,
        )
        self.assertEqual(float(oracle["safety"]), float(predicted["safety"]))
        self.assertEqual(
            float(oracle["total"].detach()),
            float(predicted["total"].detach()),
        )
        with self.assertRaisesRegex(ValueError, "forbids oracle"):
            compute_losses(
                MODULES,
                outputs,
                target,
                {
                    **batch,
                    "predicted_action_profile_target": profile,
                    "action_safety_margins": profile,
                },
                predicted_config,
            )
        with self.assertRaisesRegex(ValueError, "requires predicted_action_profile_target"):
            compute_losses(MODULES, outputs, target, batch, predicted_config)

    def test_none_h_prediction_boundary_and_fcsrl_loss_paths_remain_available(self) -> None:
        assert MODULES is not None
        base = load_config(CONFIG_PATH).objective
        outputs, target, _, batch = self._fixture()
        losses = {
            arm: compute_losses(
                MODULES,
                outputs,
                target,
                batch,
                dataclasses.replace(base, safety_arm=arm),
            )
            for arm in (
                "none",
                "h_prediction",
                "boundary_contrastive",
                "fcsrl_feasibility_loss_adaptation",
            )
        }
        self.assertEqual(float(losses["none"]["safety"].detach()), 0.0)
        self.assertEqual(
            float(losses["fcsrl_feasibility_loss_adaptation"]["safety"].detach()),
            0.0,
        )
        self.assertGreaterEqual(float(losses["h_prediction"]["safety"].detach()), 0.0)
        self.assertGreaterEqual(
            float(losses["boundary_contrastive"]["safety"].detach()),
            0.0,
        )


@unittest.skipIf(MODULES is None, "PyTorch research extra is unavailable")
class RestrictedTrainingAccessTests(unittest.TestCase):
    def test_dataset_builder_never_calls_oracle_and_has_only_train_validation_keys(self) -> None:
        assert MODULES is not None
        config = _tiny_config(arm="h_prediction")
        with patch(
            "latent_safety.learning.data.action_safety_profile",
            side_effect=AssertionError("oracle action profile must not be called"),
        ):
            bundle = build_datasets(
                config.data,
                seed=config.run.seed,
                torch=MODULES.torch,
                include_splits=("train", "validation"),
                include_action_profiles=False,
            )
        self.assertEqual(set(bundle.datasets), {"train", "validation"})
        self.assertEqual(set(bundle.split_trajectory_ids), {"train", "validation"})
        self.assertNotIn("action_safety_margins", bundle.datasets["train"][0])
        self.assertEqual(
            bundle.manifest_payload["access_scope"],
            {
                "materialized_splits": ["train", "validation"],
                "not_materialized_splits": ["calibration", "test"],
                "oracle_action_profiles_materialized": False,
            },
        )

    def test_restricted_scope_rejects_both_profile_objectives(self) -> None:
        for arm in (
            "safe_action_profile",
            "nonprivileged_predicted_action_profile",
        ):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "forbids"):
                experiment_access_scope_payload(
                    TRAIN_VALIDATION_ONLY_ACCESS,
                    _tiny_config(arm=arm),
                )

    def test_one_epoch_restricted_run_emits_no_calibration_test_or_oracle_artifacts(self) -> None:
        config = _tiny_config(arm="h_prediction")
        plan_sha = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "restricted"
            with patch(
                "latent_safety.learning.data.action_safety_profile",
                side_effect=AssertionError("oracle action profile must not be called"),
            ):
                result = run_experiment(
                    config,
                    config_path=CONFIG_PATH,
                    output_dir=output,
                    repo_root=ROOT,
                    access_scope=TRAIN_VALIDATION_ONLY_ACCESS,
                    orchestration_plan_sha256=plan_sha,
                )
            dataset = json.loads((output / "dataset_manifest.json").read_text())
            checkpoint = json.loads((output / "checkpoint_manifest.json").read_text())
            evaluation = json.loads((output / "evaluation_manifest.json").read_text())
            run = json.loads((output / "run_manifest.json").read_text())
            postfit = json.loads(
                (output / "validation_postfit_manifest.json").read_text()
            )
            expected_scope = experiment_access_scope_payload(
                TRAIN_VALIDATION_ONLY_ACCESS,
                config,
            )
            self.assertNotIn("test_metrics", result)
            self.assertEqual(set(evaluation["split_metrics"]), {"train", "validation"})
            self.assertEqual(set(evaluation["rollout_metrics"]), {"validation"})
            self.assertEqual(
                evaluation["audit_records"],
                {"validation": str(output / "audit_validation.jsonl")},
            )
            self.assertEqual(
                dataset["access_scope"]["materialized_splits"],
                ["train", "validation"],
            )
            self.assertFalse(dataset["access_scope"]["oracle_action_profiles_materialized"])
            self.assertEqual(checkpoint["data_access_scope"], expected_scope)
            self.assertEqual(evaluation["data_access_scope"], expected_scope)
            self.assertEqual(run["data_access_scope"], expected_scope)
            self.assertEqual(checkpoint["orchestration_plan_sha256"], plan_sha)
            self.assertFalse(checkpoint["selection"]["calibration_or_test_materialized"])
            self.assertEqual(
                postfit["split_access"]["not_materialized"],
                ["calibration", "test"],
            )
            self.assertEqual(
                postfit["physical_profile_target"]["source"],
                "explicit_observed_physical_rollouts",
            )
            self.assertFalse(
                postfit["physical_profile_target"][
                    "action_safety_profile_helper_called"
                ]
            )
            self.assertIsNone(postfit["profile_prediction_diagnostics"])
            self.assertEqual(
                {path.name for path in output.glob("audit_*.jsonl")},
                {"audit_validation.jsonl"},
            )

            continuation = dataclasses.replace(
                config,
                run=dataclasses.replace(config.run, epochs=2),
            )
            with self.assertRaisesRegex(ValueError, "data-access scope"):
                run_experiment(
                    continuation,
                    config_path=CONFIG_PATH,
                    output_dir=Path(temporary) / "wrong-scope-resume",
                    repo_root=ROOT,
                    resume_from=output / "checkpoint_last.pt",
                    access_scope=FULL_EVALUATION_ACCESS,
                    orchestration_plan_sha256=plan_sha,
                )

    def test_default_access_scope_remains_full_evaluation(self) -> None:
        config = _tiny_config(arm="none")
        payload = experiment_access_scope_payload(FULL_EVALUATION_ACCESS, config)
        self.assertEqual(
            payload["materialized_splits"],
            ["train", "validation", "calibration", "test"],
        )
        self.assertTrue(payload["oracle_action_profiles_materialized"])


if __name__ == "__main__":
    unittest.main()
