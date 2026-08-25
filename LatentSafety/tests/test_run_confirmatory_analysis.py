from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_confirmatory_analysis.py"
SPEC = importlib.util.spec_from_file_location("run_confirmatory_analysis", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/run_confirmatory_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConfirmatoryAnalysisScriptTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, object]]:
        domains = (
            "controlled_cart_video",
            "controlled_pendulum_video",
            "controlled_dubins_navigation_pixels",
        )
        families = ("ae", "beta_vae")
        arms = (
            "nonprivileged_predicted_action_profile",
            "none",
            "h_prediction",
            "fcsrl_feasibility_loss_adaptation",
            "append_true_h_none_view",
        )
        rows = []
        for domain, family, arm, seed in product(
            domains, families, arms, range(100, 108)
        ):
            proposed = arm == "nonprivileged_predicted_action_profile"
            rows.append(
                {
                    "domain": domain,
                    "model_family": family,
                    "arm": arm,
                    "seed": seed,
                    "safety": 0.20 if proposed else 0.40,
                    "reconstruction": 0.102 if proposed else 0.10,
                    "rollout": 0.102 if proposed else 0.10,
                    "run_complete": True,
                    "coverage_complete": True,
                }
            )
        return rows

    def test_complete_json_runs_all_36_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "observations.json"
            input_path.write_text(json.dumps(self._rows()), encoding="utf-8")
            payload = MODULE.analyze(input_path)
        self.assertEqual(payload["input"]["observation_count"], 240)
        self.assertEqual(len(payload["input"]["sha256"]), 64)
        self.assertEqual(len(payload["result"]["bounds"]), 36)
        self.assertEqual(payload["result"]["passed_bound_count"], 36)
        self.assertTrue(payload["result"]["confirmatory_success"])

    def test_missing_cell_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "observations.json"
            input_path.write_text(json.dumps(self._rows()[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete confirmatory factorial"):
                MODULE.analyze(input_path)


if __name__ == "__main__":
    unittest.main()
