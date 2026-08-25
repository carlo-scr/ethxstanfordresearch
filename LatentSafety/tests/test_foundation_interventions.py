from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.foundation.interventions import (  # noqa: E402
    assert_frozen_parameters_unchanged,
    build_action_conditioned_profile_head,
    build_residual_bottleneck_intervention,
    configure_trainable_parameter_policy,
    snapshot_frozen_parameters,
    validate_gradient_route,
    validate_trainable_parameter_policy,
)
from latent_safety.learning.runtime import require_torch  # noqa: E402


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is optional")
class FoundationInterventionTests(unittest.TestCase):
    def _model(self):
        modules = require_torch()
        nn = modules.nn

        class ToyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Linear(4, 4)
                self.repair_adapter = build_residual_bottleneck_intervention(
                    modules,
                    feature_dimension=4,
                    bottleneck_dimension=2,
                )
                self.profile_head = build_action_conditioned_profile_head(
                    modules,
                    representation_dimension=4,
                    action_count=3,
                    action_embedding_dimension=2,
                    hidden_dimension=5,
                )

            def forward(self, inputs):
                repaired = self.repair_adapter(self.encoder(inputs))
                return repaired, self.profile_head(repaired)

        return modules, ToyModel()

    def test_residual_intervention_is_identity_initialized(self) -> None:
        modules = require_torch()
        torch = modules.torch
        adapter = build_residual_bottleneck_intervention(
            modules,
            feature_dimension=4,
            bottleneck_dimension=2,
        )
        inputs = torch.randn(2, 3, 4)
        self.assertTrue(bool(torch.equal(adapter(inputs), inputs)))

    def test_profile_head_is_explicitly_action_conditioned(self) -> None:
        modules, model = self._model()
        output = model(modules.torch.randn(2, 4))[1]
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_parameter_policy_and_mutation_audit(self) -> None:
        modules, model = self._model()
        torch = modules.torch
        selected = configure_trainable_parameter_policy(
            model,
            trainable_prefixes=("repair_adapter.", "profile_head."),
        )
        self.assertTrue(selected)
        self.assertEqual(
            validate_trainable_parameter_policy(
                model,
                trainable_prefixes=("repair_adapter.", "profile_head."),
            ),
            selected,
        )
        snapshot = snapshot_frozen_parameters(
            modules,
            model,
            mutable_prefixes=("repair_adapter.", "profile_head."),
        )
        repaired, profiles = model(torch.randn(3, 4))
        (repaired.square().mean() + profiles.square().mean()).backward()
        route = validate_gradient_route(
            modules,
            model,
            required_gradient_prefixes=("repair_adapter.", "profile_head."),
            forbidden_gradient_prefixes=("encoder.",),
        )
        self.assertTrue(route["parameters_with_nonzero_finite_gradient"])
        assert_frozen_parameters_unchanged(modules, model, snapshot)

    def test_mutation_audit_detects_base_change(self) -> None:
        modules, model = self._model()
        snapshot = snapshot_frozen_parameters(
            modules,
            model,
            mutable_prefixes=("repair_adapter.", "profile_head."),
        )
        with modules.torch.no_grad():
            model.encoder.weight.add_(1.0)
        with self.assertRaisesRegex(ValueError, "frozen parameters changed"):
            assert_frozen_parameters_unchanged(modules, model, snapshot)


if __name__ == "__main__":
    unittest.main()
