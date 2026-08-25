from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning.config import cpu_smoke_config, load_config  # noqa: E402
from latent_safety.learning.data import build_datasets  # noqa: E402
from latent_safety.learning.fcsrl_protocol import (  # noqa: E402
    FCSRL_ATOM_COUNT,
    build_target_trace,
)
from latent_safety.learning.fcsrl_training import (  # noqa: E402
    ControlledFCSRLWindowDataset,
    FCSRLProtocolError,
    build_categorical_feasibility_head,
    build_ema_target_encoder,
    build_torch_target_trace,
    collate_fcsrl_windows,
    compute_fcsrl_feasibility_loss,
    deterministic_batch_indices,
    iter_fcsrl_batches,
    run_fcsrl_training_step,
    torch_project_categorical_targets,
    torch_sequence_mask,
    update_ema_target_encoder,
)
from latent_safety.learning.models import build_world_model  # noqa: E402
from latent_safety.learning.runtime import TorchUnavailableError, require_torch  # noqa: E402

try:
    MODULES = require_torch()
except TorchUnavailableError:
    MODULES = None


@unittest.skipIf(MODULES is None, "PyTorch research extra is unavailable")
class FCSRLTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert MODULES is not None
        cls.modules = MODULES
        cls.torch = MODULES.torch
        base = load_config(ROOT / "configs" / "e1_world_models" / "torch_pilot.toml")
        cls.config = cpu_smoke_config(base)
        cls.data_config = dataclasses.replace(cls.config.data, horizon=12)
        bundle = build_datasets(cls.data_config, seed=71, torch=cls.torch)
        cls.windows = ControlledFCSRLWindowDataset(
            bundle.datasets["train"],
            torch=cls.torch,
        )

    def _model_components(self):
        self.torch.manual_seed(20260822)
        model = build_world_model(
            self.modules,
            self.config.model,
            self.data_config,
        )
        target = build_ema_target_encoder(
            self.modules,
            model,
            self.config.model,
            self.data_config,
        )
        head = build_categorical_feasibility_head(
            self.modules,
            latent_dim=self.config.model.latent_dim,
            hidden_dim=13,
        )
        return model, target, head

    def _batch(self, size: int = 2):
        return collate_fcsrl_windows(
            self.torch,
            [self.windows[index] for index in range(size)],
        )

    def test_controlled_windows_are_sorted_local_and_retain_final_truncation(self) -> None:
        expected_per_trajectory = self.data_config.horizon
        self.assertEqual(
            len(self.windows),
            len(self.windows.dataset.trajectory_indices) * expected_per_trajectory,
        )
        identities = [
            (ref.trajectory_id, ref.start_timestep) for ref in self.windows.refs
        ]
        self.assertEqual(identities, sorted(identities))
        first_id = self.windows.refs[0].trajectory_id
        same_trajectory = [
            self.windows[index]
            for index, ref in enumerate(self.windows.refs)
            if ref.trajectory_id == first_id
        ]
        self.assertEqual(len(same_trajectory), expected_per_trajectory)
        self.assertFalse(bool(same_trajectory[0]["truncations"].any().item()))
        ending_at_window_end = same_trajectory[2]
        self.assertEqual(
            ending_at_window_end["truncations"].tolist(),
            [False] * 9 + [True],
        )
        self.assertEqual(ending_at_window_end["mask"].tolist(), [True] * 10)
        final = same_trajectory[-1]
        self.assertEqual(final["start_timestep"], 11)
        self.assertEqual(final["truncations"].tolist(), [True] + [False] * 9)
        self.assertEqual(final["mask"].tolist(), [True] + [False] * 9)
        self.assertTrue(all(":pad:" in sample for sample in final["sample_ids"][1:]))
        self.assertTrue(all(sample.startswith(f"{first_id}:") for sample in final["sample_ids"]))

    def test_batch_plan_and_iteration_are_epoch_deterministic(self) -> None:
        first = deterministic_batch_indices(
            len(self.windows), batch_size=4, seed=19, epoch=3
        )
        repeat = deterministic_batch_indices(
            len(self.windows), batch_size=4, seed=19, epoch=3
        )
        other_epoch = deterministic_batch_indices(
            len(self.windows), batch_size=4, seed=19, epoch=4
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other_epoch)
        flattened = [index for batch in first for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(self.windows))))
        batches = list(
            iter_fcsrl_batches(
                self.windows,
                torch=self.torch,
                batch_size=4,
                seed=19,
                epoch=3,
            )
        )
        self.assertEqual(sum(batch["histories"].shape[0] for batch in batches), len(self.windows))
        self.assertEqual(batches[0]["histories"].shape[1], 11)
        self.assertEqual(batches[0]["actions"].shape[1:], (10, 1))
        with self.assertRaisesRegex(FCSRLProtocolError, "integer"):
            deterministic_batch_indices(0, batch_size=4, seed=19)

    def test_vectorized_targets_match_scalar_contract_and_inclusive_masks(self) -> None:
        bootstrap = self.torch.tensor(
            [[0.2 + 0.1 * index for index in range(10)]] * 2,
            dtype=self.torch.float64,
            requires_grad=True,
        )
        violations = self.torch.zeros((2, 10), dtype=self.torch.bool)
        violations[0, 1] = True
        terminations = self.torch.zeros((2, 10), dtype=self.torch.bool)
        truncations = self.torch.zeros((2, 10), dtype=self.torch.bool)
        terminations[0, 2] = True
        truncations[1, 2] = True
        trace = build_torch_target_trace(
            self.torch,
            violations,
            terminations,
            truncations,
            bootstrap,
        )
        self.assertFalse(trace.targets.requires_grad)
        self.assertEqual(trace.mask.tolist(), [[True, True, True] + [False] * 7] * 2)
        for row in range(2):
            scalar = build_target_trace(
                violations=tuple(violations[row].tolist()),
                terminations=tuple(terminations[row].tolist()),
                truncations=tuple(truncations[row].tolist()),
                bootstrap_values=tuple(bootstrap[row].detach().tolist()),
            )
            self.assertEqual(trace.mask[row].tolist(), [bool(value) for value in scalar.mask])
            for observed, expected in zip(trace.targets[row].tolist(), scalar.targets, strict=True):
                self.assertAlmostEqual(observed, expected, places=12)
        projected = torch_project_categorical_targets(self.torch, trace.targets)
        self.assertEqual(projected.shape, (2, 10, FCSRL_ATOM_COUNT))
        self.assertTrue(
            self.torch.allclose(
                projected.sum(dim=-1),
                self.torch.ones((2, 10), dtype=self.torch.float64),
            )
        )

    def test_ema_initializes_exactly_freezes_and_uses_source_mix(self) -> None:
        model, target, _ = self._model_components()
        model.eval()
        history = self.torch.rand(
            2,
            self.data_config.history_length,
            self.data_config.channels,
            self.data_config.image_size,
            self.data_config.image_size,
        )
        online_latent = model.encode(history, sample=False)["z"]
        target_latent = target(history)
        self.assertTrue(self.torch.equal(online_latent, target_latent))
        self.assertFalse(target.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in target.parameters()))

        target_parameter = dict(target.named_parameters())[
            "frame_encoder.convolutions.0.weight"
        ]
        source_parameter = dict(model.named_parameters())[
            "frame_encoder.convolutions.0.weight"
        ]
        old_target = target_parameter.detach().clone()
        with self.torch.no_grad():
            source_parameter.add_(2.0)
        update_ema_target_encoder(
            self.torch,
            target,
            model,
            source_mix=0.25,
        )
        expected = 0.75 * old_target + 0.25 * source_parameter.detach()
        self.assertTrue(self.torch.equal(target_parameter, expected))
        target.train(True)
        self.assertFalse(target.training)

    def test_feasibility_loss_uses_first_four_rollouts_and_detached_targets(self) -> None:
        model, target, head = self._model_components()
        model.train()
        output = compute_fcsrl_feasibility_loss(
            self.modules,
            model,
            target,
            head,
            self._batch(),
        )
        self.assertEqual(output.logits.shape, (2, 4, 63))
        self.assertEqual(
            output.rollout_latents.shape,
            (2, 4, self.config.model.latent_dim),
        )
        self.assertEqual(output.target_trace.targets.shape, (2, 10))
        self.assertEqual(output.target_trace.projected_targets.shape, (2, 10, 63))
        self.assertEqual(output.active_positions, 8)
        self.assertTrue(output.loss.requires_grad)
        self.assertFalse(output.bootstrap_values.requires_grad)
        self.assertFalse(output.target_trace.targets.requires_grad)
        self.assertTrue(math.isfinite(float(output.loss.detach().item())))
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.transition.parameters())
        )
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

    def test_combined_step_updates_head_then_updates_ema_once(self) -> None:
        model, target, head = self._model_components()
        model.train()
        batch = self._batch()
        optimizer = self.torch.optim.Adam(
            [*model.parameters(), *head.parameters()],
            lr=1e-3,
        )
        outputs = model(batch["histories"][:, 0], batch["actions"][:, 0])
        base_loss = (
            (outputs["reconstruction"] - batch["histories"][:, 0, -1]).square().mean()
            + outputs["predicted_next_z"].square().mean()
        )
        target_name = "frame_encoder.convolutions.0.weight"
        target_before = dict(target.named_parameters())[target_name].detach().clone()
        head_before = next(head.parameters()).detach().clone()
        result = run_fcsrl_training_step(
            self.modules,
            model,
            target,
            head,
            optimizer,
            batch,
            base_loss=base_loss,
            feasibility_weight=0.1,
            gradient_clip_norm=5.0,
        )
        source_after = dict(model.named_parameters())[target_name].detach()
        target_after = dict(target.named_parameters())[target_name].detach()
        self.assertTrue(
            self.torch.allclose(
                target_after,
                0.99 * target_before + 0.01 * source_after,
                rtol=0.0,
                atol=1e-7,
            )
        )
        self.assertFalse(self.torch.equal(head_before, next(head.parameters()).detach()))
        self.assertEqual(result.active_positions, 8)
        self.assertEqual(result.feasibility_weight, 0.1)
        self.assertIsNotNone(result.gradient_norm)
        self.assertTrue(math.isfinite(result.total_loss))

    def test_fail_closed_batch_and_optimizer_checks(self) -> None:
        model, target, head = self._model_components()
        batch = self._batch()
        incorrect_mask = dict(batch)
        incorrect_mask["mask"] = batch["mask"].clone()
        incorrect_mask["mask"][0, 0] = False
        with self.assertRaisesRegex(FCSRLProtocolError, "exact ending-transition"):
            compute_fcsrl_feasibility_loss(
                self.modules,
                model,
                target,
                head,
                incorrect_mask,
            )
        nonfinite = dict(batch)
        nonfinite["actions"] = batch["actions"].clone()
        nonfinite["actions"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FCSRLProtocolError, "finite"):
            compute_fcsrl_feasibility_loss(
                self.modules,
                model,
                target,
                head,
                nonfinite,
            )
        overlapping = self.torch.zeros((1, 10), dtype=self.torch.bool)
        overlapping[0, 0] = True
        with self.assertRaisesRegex(FCSRLProtocolError, "both"):
            torch_sequence_mask(self.torch, overlapping, overlapping)

        optimizer = self.torch.optim.SGD(model.parameters(), lr=1e-3)
        base_loss = sum(parameter.square().mean() for parameter in model.parameters())
        with self.assertRaisesRegex(FCSRLProtocolError, "optimizer is missing"):
            run_fcsrl_training_step(
                self.modules,
                model,
                target,
                head,
                optimizer,
                batch,
                base_loss=base_loss,
                feasibility_weight=0.1,
            )


if __name__ == "__main__":
    unittest.main()
