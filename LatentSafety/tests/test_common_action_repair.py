from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.foundation.repair import (  # noqa: E402
    CommonActionRepairConfig,
    common_action_geometry_loss,
    common_action_geometry_reference_loss,
    encoder_only_state_dict,
)
from latent_safety.learning.runtime import require_torch  # noqa: E402


class CommonActionReferenceLossTests(unittest.TestCase):
    def test_shared_action_has_zero_hard_loss(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (0.0,)),
            ((1.0, -0.5), (0.25, 1.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(result.chosen_actions, (0, 0))
        self.assertEqual(result.active_nonself_counts, (1, 1))
        self.assertEqual(result.eligible_center_coverage, 1.0)

    def test_disjoint_actions_have_positive_known_loss(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (0.0,)),
            ((1.0, -0.5), (-0.25, 1.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.loss, 0.25)
        self.assertEqual(result.per_center_minimum, (0.25, 0.25))

    def test_three_way_conflict_is_not_reduced_to_independent_pairs(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (0.0,), (0.0,)),
            ((1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.loss, 1.0)
        self.assertEqual(result.per_center_minimum, (1.0, 1.0, 1.0))

    def test_nonviable_targets_are_removed_before_set_cost(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (0.0,), (0.0,)),
            ((-1.0, -1.0), (1.0, -0.5), (-0.25, 1.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.viable_center_count, 2)
        self.assertEqual(result.per_center_minimum[0], None)
        self.assertEqual(result.loss, 0.25)

    def test_training_hinge_is_strict_at_radius_boundary(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (1.0,)),
            ((1.0, -1.0), (-1.0, 1.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.loss, 0.0)

    def test_empty_viable_reference_returns_explicit_count_and_none_rows(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (0.0,)),
            ((-1.0, -0.5), (-0.25, -2.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(result.viable_center_count, 0)
        self.assertEqual(result.eligible_center_count, 0)
        self.assertIsNone(result.eligible_center_coverage)
        self.assertEqual(result.per_center_minimum, (None, None))

    def test_isolated_viable_centers_are_visible_in_coverage(self) -> None:
        result = common_action_geometry_reference_loss(
            ((0.0,), (5.0,)),
            ((1.0, -1.0), (-1.0, 1.0)),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(result.active_nonself_counts, (0, 0))
        self.assertEqual(result.eligible_center_count, 0)
        self.assertEqual(result.eligible_center_coverage, 0.0)

    def test_reference_loss_is_sample_and_action_permutation_invariant(self) -> None:
        latents = ((0.0,), (0.2,), (0.4,))
        profiles = ((1.0, -1.0), (-0.4, 1.0), (0.5, -0.2))
        baseline = common_action_geometry_reference_loss(
            latents,
            profiles,
            gamma=0.0,
            train_radius=1.0,
        )
        sample_order = (2, 0, 1)
        permuted_samples = common_action_geometry_reference_loss(
            tuple(latents[index] for index in sample_order),
            tuple(profiles[index] for index in sample_order),
            gamma=0.0,
            train_radius=1.0,
        )
        permuted_actions = common_action_geometry_reference_loss(
            latents,
            tuple((right, left) for left, right in profiles),
            gamma=0.0,
            train_radius=1.0,
        )
        self.assertAlmostEqual(baseline.loss, permuted_samples.loss)
        self.assertAlmostEqual(baseline.loss, permuted_actions.loss)

    def test_config_requires_a_training_buffer_beyond_audit_radius(self) -> None:
        config = CommonActionRepairConfig(
            gamma=0.0,
            audit_radius=1.0,
            train_radius=1.0,
            softmin_temperature=0.1,
        )
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            config.validate()

    def test_gamma_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            common_action_geometry_reference_loss(
                ((0.0,),),
                ((1.0,),),
                gamma=-0.1,
                train_radius=1.0,
            )
        config = CommonActionRepairConfig(
            gamma=-0.1,
            audit_radius=0.5,
            train_radius=1.0,
            softmin_temperature=0.1,
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            config.validate()

    def test_encoder_export_physically_removes_training_head(self) -> None:
        state = {
            "encoder.block.weight": object(),
            "repair_adapter.down.weight": object(),
            "profile_head.output.weight": object(),
            "profile_head.output.bias": object(),
        }
        exported = encoder_only_state_dict(state)
        self.assertEqual(
            tuple(exported),
            ("encoder.block.weight", "repair_adapter.down.weight"),
        )
        self.assertFalse(any(key.startswith("profile_head.") for key in exported))
        with self.assertRaisesRegex(ValueError, "no keys"):
            encoder_only_state_dict({"encoder.weight": object()})

    def test_encoder_export_rejects_undeclared_side_modules(self) -> None:
        state = {
            "encoder.weight": object(),
            "profile_head.weight": object(),
            "controller_head.weight": object(),
        }
        with self.assertRaisesRegex(ValueError, "undeclared non-encoder"):
            encoder_only_state_dict(state)

    def test_encoder_export_supports_an_explicit_backend_prefix(self) -> None:
        state = {
            "backbone.block.weight": object(),
            "profile_head.weight": object(),
        }
        exported = encoder_only_state_dict(
            state,
            retained_prefixes=("backbone.",),
        )
        self.assertEqual(tuple(exported), ("backbone.block.weight",))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is optional")
class CommonActionTorchLossTests(unittest.TestCase):
    def test_softmin_is_bounded_and_profiles_are_detached(self) -> None:
        modules = require_torch()
        torch = modules.torch
        latents = torch.tensor([[0.0], [0.25]], requires_grad=True)
        profiles = torch.tensor(
            [[1.0, -0.5], [-0.25, 1.0]],
            requires_grad=True,
        )
        temperature = 0.1
        result = common_action_geometry_loss(
            modules,
            latents,
            profiles,
            gamma=0.0,
            train_radius=1.0,
            temperature=temperature,
        )
        self.assertGreaterEqual(float(result.loss.item()), float(result.hard_loss.item()))
        self.assertLessEqual(
            float(result.loss.item()),
            float(result.hard_loss.item()) + temperature * math.log(2) + 1e-6,
        )
        result.loss.backward()
        self.assertIsNone(profiles.grad)
        self.assertIsNotNone(latents.grad)
        self.assertTrue(bool(torch.isfinite(latents.grad).all().item()))
        self.assertTrue(bool((latents.grad != 0).any().item()))

    def test_torch_hard_costs_match_the_reference(self) -> None:
        modules = require_torch()
        torch = modules.torch
        latents = ((0.0, 0.0), (0.2, 0.1), (1.5, 0.0))
        profiles = ((1.0, -0.5), (-0.25, 1.0), (0.75, 0.5))
        reference = common_action_geometry_reference_loss(
            latents,
            profiles,
            gamma=0.0,
            train_radius=1.0,
        )
        observed = common_action_geometry_loss(
            modules,
            torch.tensor(latents),
            torch.tensor(profiles),
            gamma=0.0,
            train_radius=1.0,
            temperature=0.1,
        )
        torch.testing.assert_close(
            observed.costs,
            torch.tensor(reference.per_center_costs),
        )
        self.assertAlmostEqual(float(observed.hard_loss.item()), reference.loss, places=6)

    def test_exact_duplicate_ca_gradient_limitation_is_explicit(self) -> None:
        modules = require_torch()
        torch = modules.torch
        latents = torch.tensor([[0.0], [0.0]], requires_grad=True)
        result = common_action_geometry_loss(
            modules,
            latents,
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
            gamma=0.0,
            train_radius=1.0,
            temperature=0.1,
        )
        result.loss.backward()
        self.assertTrue(bool(torch.equal(latents.grad, torch.zeros_like(latents))))

    def test_mask_must_be_boolean_symmetric_and_keep_viable_diagonal(self) -> None:
        modules = require_torch()
        torch = modules.torch
        latents = torch.tensor([[0.0], [0.1]])
        profiles = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        kwargs = {
            "gamma": 0.0,
            "train_radius": 1.0,
            "temperature": 0.1,
        }
        with self.assertRaisesRegex(ValueError, "boolean"):
            common_action_geometry_loss(
                modules,
                latents,
                profiles,
                comparison_mask=torch.ones((2, 2)),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            common_action_geometry_loss(
                modules,
                latents,
                profiles,
                comparison_mask=torch.tensor([[True, True], [False, True]]),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "diagonal"):
            common_action_geometry_loss(
                modules,
                latents,
                profiles,
                comparison_mask=torch.tensor([[False, True], [True, False]]),
                **kwargs,
            )

    def test_neighbor_coverage_is_explicit(self) -> None:
        modules = require_torch()
        torch = modules.torch
        result = common_action_geometry_loss(
            modules,
            torch.tensor([[0.0], [0.1], [5.0]]),
            torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
            gamma=0.0,
            train_radius=1.0,
            temperature=0.1,
        )
        self.assertEqual(result.active_nonself_counts.tolist(), [1, 1, 0])
        self.assertEqual(result.eligible_center_mask.tolist(), [True, True, False])
        self.assertAlmostEqual(result.eligible_center_coverage or 0.0, 2.0 / 3.0)

    def test_nonfinite_inputs_fail_closed(self) -> None:
        modules = require_torch()
        torch = modules.torch
        with self.assertRaisesRegex(ValueError, "finite"):
            common_action_geometry_loss(
                modules,
                torch.tensor([[float("nan")]]),
                torch.tensor([[1.0]]),
                gamma=0.0,
                train_radius=1.0,
                temperature=0.1,
            )


if __name__ == "__main__":
    unittest.main()
