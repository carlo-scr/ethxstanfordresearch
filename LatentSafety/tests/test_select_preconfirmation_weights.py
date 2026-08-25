from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.preconfirmation import (
    CONTROL_ARM,
    DOMAINS,
    LEARNED_ARMS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    POSITIVE_WEIGHTS,
    PROFILE_ARM,
)


SCRIPT_PATH = ROOT / "scripts" / "select_preconfirmation_weights.py"
SPEC = importlib.util.spec_from_file_location("select_preconfirmation_weights", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/select_preconfirmation_weights.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain, family, seed in product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS):
        rows.append(
            {
                "domain": domain,
                "model_family": family,
                "arm": CONTROL_ARM,
                "seed": seed,
                "weight": None,
                "safety": 0.5,
                "reconstruction": 1.0,
                "rollout": 2.0,
                "eligible_center_coverage": 0.9,
                "neighborhood_mass": 1.0,
                "empty_trajectory_count": 0,
                "run_complete": True,
                "coverage_complete": True,
            }
        )
    for domain, family, arm, seed, weight in product(
        DOMAINS,
        MODEL_FAMILIES,
        LEARNED_ARMS,
        PILOT_SEEDS,
        POSITIVE_WEIGHTS,
    ):
        row: dict[str, object] = {
            "domain": domain,
            "model_family": family,
            "arm": arm,
            "seed": seed,
            "weight": weight,
            "safety": 0.3 + 0.01 * POSITIVE_WEIGHTS.index(weight),
            "reconstruction": 1.01,
            "rollout": 2.02,
            "eligible_center_coverage": 0.88,
            "neighborhood_mass": 1.02,
            "empty_trajectory_count": 0,
            "run_complete": True,
            "coverage_complete": True,
        }
        if arm == PROFILE_ARM:
            row.update(
                {
                    "profile_normalized_p95_error": 0.08,
                    "profile_teacher_count": 5,
                    "profile_teacher_failure_count": 0,
                    "profile_label_manifest_sha256": hashlib.sha256(
                        f"{domain}|{family}|{seed}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        rows.append(row)
    return rows


class PreconfirmationSelectionScriptTests(unittest.TestCase):
    def test_bare_rows_cannot_bypass_authenticated_radius_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "validation.json"
            input_path.write_text(json.dumps({"observations": _rows()}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authenticated canonical 288-row"):
                MODULE.select(input_path)

    def test_missing_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "validation.json"
            input_path.write_text(json.dumps(_rows()[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authenticated canonical 288-row"):
                MODULE.select(input_path)

    def test_selector_rejects_frozen_mass_threshold_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "intervention.toml"
            original = (ROOT / "configs/e2_frontier/intervention.toml").read_text(
                encoding="utf-8"
            )
            drifted = original.replace(
                "max_relative_mass_mismatch = 0.05",
                "max_relative_mass_mismatch = 0.10",
            )
            self.assertNotEqual(drifted, original)
            config_path.write_text(drifted, encoding="utf-8")
            input_path = root / "validation.json"
            input_path.write_text(
                json.dumps({"observations": _rows()}), encoding="utf-8"
            )
            with patch.object(MODULE, "CONFIG_PATH", config_path), self.assertRaisesRegex(
                ValueError, r"\[matched_radius_reduction\].*frozen E2 protocol"
            ):
                MODULE.select(input_path)


if __name__ == "__main__":
    unittest.main()
