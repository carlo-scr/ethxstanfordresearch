from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_e1_sweep.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metric_payload(
    *,
    utility: float,
    defect: float,
    conflicting: int,
    viable: int,
    worst: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = {
        "world_model_utility": utility,
        "reconstruction": utility / 2.0,
    }
    audit = {
        "record_count": 8,
        "static_curve": [
            {
                "relative_radius": 0.05,
                "absolute_radius": 0.5,
                "estimate": {
                    "delta": 0.5,
                    "witness_margin": defect,
                    "confounded_safe_count": 2,
                    "safe_count": 4,
                    "confounded_fraction": 0.5,
                    "margin_quantile": defect,
                    "quantile_level": 0.99,
                },
                "witnesses": [],
            }
        ],
        "action_curve": [
            {
                "relative_radius": 0.05,
                "delta": 0.5,
                "state_count": 8,
                "nontrivial_neighborhood_count": viable + 1,
                "individually_viable_neighborhood_count": viable,
                "conflicting_neighborhood_count": conflicting,
                "conflict_fraction": conflicting / viable if viable else 0.0,
                "worst_required_violation": worst,
                "trajectory_tail_summary": {
                    "all_trajectory_count": 8,
                    "evaluable_trajectory_count": viable,
                    "mean_trajectory_tail_required_violation": worst,
                },
                "witnesses": [],
            }
        ],
    }
    return evaluation, audit


def _rollout_payload(utility: float) -> dict[str, dict[str, float | int]]:
    return {
        "1": {
            "latent_mse": utility + 0.005,
            "pixel_mse": utility + 0.01,
            "cases": 8,
        },
        "4": {
            "latent_mse": utility + 0.02,
            "pixel_mse": utility + 0.04,
            "cases": 8,
        },
    }


def _task(
    task_id: int,
    output: Path,
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "model_family": "ae",
        "history_mode": "stack_h4",
        "history_encoder": "stack",
        "history_length": 4,
        "latent_dim": 8,
        "safety_arm": arm,
        "seed": seed,
        "kl_weight": 0.0,
        "safety_weight": 0.0 if arm == "none" else 0.5,
        "device": "cuda",
        "output_dir": str(output),
    }


def _write_run(
    task: dict[str, Any],
    *,
    plan_sha256: str,
    utility: float,
    defect: float,
    conflicting: int,
    viable: int,
    worst: float,
    valid_test: bool = False,
) -> None:
    output = Path(task["output_dir"])
    resolved_hash = f"resolved-{task['task_id']}"
    resolved_values = {
        "model": {
            "family": task["model_family"],
            "history_encoder": task["history_encoder"],
            "latent_dim": task["latent_dim"],
        },
        "data": {"history_length": task["history_length"]},
        "objective": {"safety_arm": task["safety_arm"]},
        "evaluation": {"rollout_horizons": [1, 4]},
        "run": {"seed": task["seed"]},
    }
    validation_metrics, validation_audit = _metric_payload(
        utility=utility,
        defect=defect,
        conflicting=conflicting,
        viable=viable,
        worst=worst,
    )
    if valid_test:
        test_metrics, test_audit = _metric_payload(
            utility=utility + 1.0,
            defect=defect + 1.0,
            conflicting=0,
            viable=2,
            worst=0.0,
        )
    else:
        # A non-unblinded aggregation must not traverse these values.
        test_metrics = "POISONED_TEST_VALUE"
        test_audit = "POISONED_TEST_VALUE"
    run = {
        "schema_version": 1,
        "status": "success",
        "experiment": "fabricated_training",
        "seed": task["seed"],
        "training_arm": task["safety_arm"],
        "config": {"sha256": "base-config-hash"},
        "dataset": {"manifest_sha256": "dataset-hash"},
        "evaluation_manifest": str(output / "evaluation_manifest.json"),
        "checkpoint_manifest": str(output / "checkpoint_manifest.json"),
        "resolved_config": {"sha256": resolved_hash, "values": resolved_values},
        "orchestration": {
            "plan_sha256": plan_sha256,
            "task_id": task["task_id"],
            "completed_audits": (
                [
                    "latent",
                    "observation_oracle",
                    "state_oracle",
                    "latent_plus_margin",
                    "latent_plus_action_profile",
                ]
                if task["safety_arm"] == "none"
                else ["latent"]
            ),
            "failed_audit": None,
            "updated_at_utc": "2026-08-22T12:00:00+00:00",
        },
        "failure": None,
    }
    evaluation = {
        "schema_version": 1,
        "status": "success",
        "experiment": "fabricated_training",
        "config": {"sha256": "base-config-hash"},
        "resolved_config_sha256": resolved_hash,
        "checkpoint_sha256": f"checkpoint-{task['task_id']}",
        "checkpoint_manifest": str(output / "checkpoint_manifest.json"),
        "dataset_manifest_sha256": "dataset-hash",
        "audit_records": {
            "calibration": str(output / "audit_calibration.jsonl"),
            "validation": str(output / "audit_validation.jsonl"),
            "test": str(output / "audit_test.jsonl"),
        },
        "split_metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "rollout_metrics": {
            "validation": _rollout_payload(utility),
            "test": (
                _rollout_payload(utility + 1.0)
                if valid_test
                else "POISONED_TEST_VALUE"
            ),
        },
    }
    audit = {
        "schema_version": 1,
        "status": "success",
        "config": {"sha256": "base-config-hash"},
        "representation_view": {"name": "latent", "oracle_control": False},
        "radius_calibration": {
            "uses_safety_labels": False,
            "calibration_scale": 10.0,
        },
        "inputs": {
            "calibration_records": str(output / "audit_calibration.jsonl"),
            "validation_records": str(output / "audit_validation.jsonl"),
            "test_records": str(output / "audit_test.jsonl"),
            "calibration_count": 8,
            "validation_count": 8,
            "test_count": 8,
        },
        "split_audits": {
            "validation": validation_audit,
            "test": test_audit,
        },
    }
    _write_json(output / "run_manifest.json", run)
    _write_json(output / "evaluation_manifest.json", evaluation)
    _write_json(output / "latent_safety_audit.json", audit)


def _write_plan(root: Path, *, valid_test: bool = False) -> Path:
    specifications = (
        ("none", 0, 0.20, 0.40, 1, 2, 0.20),
        ("none", 1, 0.30, 0.30, 2, 5, 0.15),
        ("h_prediction", 0, 0.25, 0.20, 0, 3, 0.00),
        ("h_prediction", 1, 0.35, 0.10, 1, 5, 0.05),
    )
    tasks = []
    run_specifications: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    for task_id, (arm, seed, utility, defect, conflicts, viable, worst) in enumerate(
        specifications
    ):
        task = _task(task_id, root / f"run-{task_id}", arm=arm, seed=seed)
        tasks.append(task)
        run_specifications.append(
            (task, (utility, defect, conflicts, viable, worst))
        )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fabricated_e1_grid",
        "status": "test_fixture",
        "grid": {"path": "fixture.toml", "sha256": "grid-hash"},
        "base_config": {"path": "fixture-base.toml", "sha256": "config-hash"},
        "axes": {
            "model_families": ["ae"],
            "history_modes": ["stack_h4"],
            "latent_dimensions": [8],
            "safety_arms": ["none", "h_prediction"],
            "seeds": [0, 1],
        },
        "history_mode_definitions": {
            "stack_h4": {"history_encoder": "stack", "history_length": 4}
        },
        "task_count": len(tasks),
        "tasks": tasks,
        "analysis": {
            "relative_radii": [0.0, 0.05, 0.10],
            "primary_relative_radius": 0.05,
            "max_validation_static_defect": 0.20,
            "bootstrap_resamples": 200,
            "bootstrap_seed": 17,
            "selection_split": "validation",
        },
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    for task, (utility, defect, conflicts, viable, worst) in run_specifications:
        _write_run(
            task,
            plan_sha256=plan["plan_sha256"],
            utility=utility,
            defect=defect,
            conflicting=conflicts,
            viable=viable,
            worst=worst,
            valid_test=valid_test,
        )
    path = root / "plan.json"
    _write_json(path, plan)
    return path


def _run(plan: Path, output: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plan",
            str(plan),
            "--output",
            str(output),
            *options,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


class AggregateE1SweepTests(unittest.TestCase):
    def test_complete_validation_only_aggregate_and_paired_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            declared_plan_hash = json.loads(plan.read_text(encoding="utf-8"))[
                "plan_sha256"
            ]
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "complete")
        self.assertFalse(payload["test_unblinded"])
        self.assertEqual(payload["analysis_contract"]["test_access"], "not_accessed")
        self.assertEqual(len(payload["rows"]), 4)
        self.assertTrue(all("test" not in row for row in payload["rows"]))
        self.assertTrue(
            all(
                row["orchestration"]["plan_sha256"] == declared_plan_hash
                and row["orchestration"]["task_id"] == row["task_id"]
                for row in payload["rows"]
            )
        )
        self.assertEqual(
            set(payload["rows"][0]["orchestration"]["completed_audits"]),
            {
                "latent",
                "observation_oracle",
                "state_oracle",
                "latent_plus_margin",
                "latent_plus_action_profile",
            },
        )
        self.assertTrue(
            all(
                all(
                    token in row["config_id"]
                    for token in ("task-", "family-", "history-", "dim-", "arm-")
                )
                for row in payload["rows"]
            )
        )
        self.assertEqual(len(payload["model_rows"]), 2)
        self.assertTrue(
            all(row["seed_set_status"] == "complete" for row in payload["model_rows"])
        )
        model_by_arm = {row["safety_arm"]: row for row in payload["model_rows"]}
        self.assertAlmostEqual(
            model_by_arm["none"]["validation_median"]["utility_loss"], 0.25
        )
        self.assertAlmostEqual(
            model_by_arm["none"]["validation_median"]["reconstruction_mse"],
            0.125,
        )
        self.assertAlmostEqual(
            model_by_arm["none"]["validation_mean"][
                "rollout_pixel_mse_at_max_horizon"
            ],
            0.29,
        )
        self.assertEqual(model_by_arm["none"]["rollout_summary_horizon"], 4)
        first_validation = payload["rows"][0]["validation"]
        self.assertEqual(
            first_validation["rollout_summary_rule"],
            "maximum_predeclared_configured_horizon",
        )
        self.assertEqual(first_validation["rollout_summary_horizon"], 4)
        self.assertEqual(
            {
                horizon: point["cases"]
                for horizon, point in first_validation[
                    "rollout_pixel_mse_by_horizon"
                ].items()
            },
            {"1": 8, "4": 8},
        )
        self.assertAlmostEqual(
            first_validation["rollout_pixel_mse_by_horizon"]["1"]["pixel_mse"],
            0.21,
        )
        self.assertAlmostEqual(
            first_validation["rollout_pixel_mse_by_horizon"]["4"]["pixel_mse"],
            0.24,
        )
        frontier = payload["validation_frontier"]
        self.assertEqual(frontier["selection_split"], "validation")
        self.assertIn("arm-h_prediction", frontier["selected"]["config_id"])
        comparison = payload["paired_seed_comparisons"][0]
        self.assertEqual(comparison["status"], "complete")
        self.assertEqual(comparison["paired_seeds"], [0, 1])
        static_ci = comparison["metrics"]["static_defect"]["paired_bootstrap_ci"]
        self.assertAlmostEqual(static_ci["estimate"], -0.2)
        self.assertEqual(static_ci["difference_convention"], "candidate - baseline")
        self.assertIn("action_tail_required_violation", comparison["metrics"])
        self.assertIn("reconstruction_mse", comparison["metrics"])
        self.assertIn(
            "rollout_pixel_mse_at_max_horizon", comparison["metrics"]
        )
        self.assertEqual(payload["source_plan"]["plan_sha256"], declared_plan_hash)

    def test_default_incomplete_plan_is_recorded_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            missing_audit = root / "run-3" / "latent_safety_audit.json"
            missing_audit.unlink()
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["status"], "incomplete_rejected")
        self.assertFalse(payload["analysis_ready"])
        self.assertEqual(payload["completeness"]["missing_task_count"], 1)
        self.assertIsNone(payload["validation_frontier"])
        self.assertEqual(payload["model_rows"], [])
        self.assertEqual(payload["paired_seed_comparisons"], [])

    def test_partial_mode_is_prominent_and_does_not_invent_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            (root / "run-3" / "latent_safety_audit.json").unlink()
            output = root / "aggregate.json"
            completed = _run(plan, output, "--allow-partial")
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["status"], "partial_exploratory")
        self.assertTrue(payload["exploratory_partial_mode"])
        self.assertIn("PARTIAL MODE", " ".join(payload["warnings"]))
        comparison = payload["paired_seed_comparisons"][0]
        self.assertEqual(comparison["status"], "insufficient_pairs")
        self.assertEqual(comparison["paired_seeds"], [0])
        self.assertEqual(comparison["metrics"], {})
        self.assertEqual(
            payload["validation_frontier"]["excluded_incomplete_config_ids"],
            ["family-ae__history-stack_h4__dim-8__arm-h_prediction"],
        )

    def test_unblind_flag_is_required_to_emit_test_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root, valid_test=True)
            output = root / "aggregate.json"
            completed = _run(plan, output, "--unblind-test")
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(payload["test_unblinded"])
        self.assertFalse(payload["selection_uses_test_data"])
        self.assertTrue(all("test" in row for row in payload["rows"]))
        self.assertAlmostEqual(payload["rows"][0]["test"]["utility_loss"], 1.2)
        self.assertAlmostEqual(payload["rows"][0]["test"]["reconstruction_mse"], 0.6)
        self.assertAlmostEqual(
            payload["rows"][0]["test"]["rollout_pixel_mse_at_max_horizon"],
            1.24,
        )
        self.assertIn("TEST UNBLINDED", " ".join(payload["warnings"]))

    def test_missing_orchestration_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            path = root / "run-0" / "run_manifest.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            del run["orchestration"]
            _write_json(path, run)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["status"], "incomplete_rejected")
        self.assertEqual(payload["completeness"]["failed_task_count"], 1)
        self.assertIn(
            "run.orchestration must be an object",
            payload["completeness"]["failed_tasks"][0]["reason"],
        )

    def test_orchestration_plan_and_task_mismatches_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            hash_path = root / "run-0" / "run_manifest.json"
            hash_run = json.loads(hash_path.read_text(encoding="utf-8"))
            hash_run["orchestration"]["plan_sha256"] = "0" * 64
            _write_json(hash_path, hash_run)
            task_path = root / "run-1" / "run_manifest.json"
            task_run = json.loads(task_path.read_text(encoding="utf-8"))
            task_run["orchestration"]["task_id"] = 999
            _write_json(task_path, task_run)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["completeness"]["failed_task_count"], 2)
        reasons = " ".join(
            failure["reason"] for failure in payload["completeness"]["failed_tasks"]
        )
        self.assertIn("plan_sha256 disagrees with the source plan", reasons)
        self.assertIn("task_id disagrees with the source plan task", reasons)

    def test_malformed_completed_audits_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            path = root / "run-2" / "run_manifest.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            run["orchestration"]["completed_audits"] = "latent"
            _write_json(path, run)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["completeness"]["failed_task_count"], 1)
        self.assertIn(
            "run.orchestration.completed_audits must be a list",
            payload["completeness"]["failed_tasks"][0]["reason"],
        )

    def test_success_status_cannot_bypass_required_audit_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            baseline_path = root / "run-0" / "run_manifest.json"
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline["orchestration"]["completed_audits"] = ["latent"]
            _write_json(baseline_path, baseline)
            candidate_path = root / "run-2" / "run_manifest.json"
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["orchestration"]["completed_audits"] = []
            _write_json(candidate_path, candidate)
            failed_state_path = root / "run-3" / "run_manifest.json"
            failed_state = json.loads(
                failed_state_path.read_text(encoding="utf-8")
            )
            failed_state["orchestration"]["failed_audit"] = "latent"
            _write_json(failed_state_path, failed_state)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["completeness"]["failed_task_count"], 3)
        reasons = " ".join(
            failure["reason"] for failure in payload["completeness"]["failed_tasks"]
        )
        self.assertIn("required views for safety arm 'none'", reasons)
        self.assertIn("missing=['latent']", reasons)
        self.assertIn("failed_audit must be null", reasons)

    def test_missing_validation_reconstruction_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            path = root / "run-0" / "evaluation_manifest.json"
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            del evaluation["split_metrics"]["validation"]["reconstruction"]
            _write_json(path, evaluation)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["status"], "incomplete_rejected")
        self.assertEqual(payload["completeness"]["failed_task_count"], 1)
        self.assertIn(
            "evaluation.split_metrics.validation.reconstruction must be a finite number",
            payload["completeness"]["failed_tasks"][0]["reason"],
        )

    def test_malformed_rollout_pixel_mse_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            path = root / "run-0" / "evaluation_manifest.json"
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            evaluation["rollout_metrics"]["validation"]["4"]["pixel_mse"] = -0.1
            _write_json(path, evaluation)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["completeness"]["failed_task_count"], 1)
        self.assertIn(
            "evaluation.rollout_metrics.validation.4.pixel_mse must be non-negative",
            payload["completeness"]["failed_tasks"][0]["reason"],
        )

    def test_missing_configured_rollout_horizon_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            path = root / "run-0" / "evaluation_manifest.json"
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            del evaluation["rollout_metrics"]["validation"]["4"]
            _write_json(path, evaluation)
            output = root / "aggregate.json"
            completed = _run(plan, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["completeness"]["failed_task_count"], 1)
        self.assertIn(
            "must exactly cover the resolved configured horizons",
            payload["completeness"]["failed_tasks"][0]["reason"],
        )

    def test_failed_run_and_manifest_mismatch_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            failed_path = root / "run-0" / "run_manifest.json"
            failed = json.loads(failed_path.read_text(encoding="utf-8"))
            failed.update({"status": "failed", "failure": {"type": "RuntimeError"}})
            _write_json(failed_path, failed)
            mismatch_path = root / "run-1" / "evaluation_manifest.json"
            mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
            mismatch["resolved_config_sha256"] = "wrong"
            _write_json(mismatch_path, mismatch)
            output = root / "aggregate.json"
            completed = _run(plan, output, "--allow-partial")
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["completeness"]["failed_task_count"], 2)
        reasons = " ".join(item["reason"] for item in payload["completeness"]["failed_tasks"])
        self.assertIn("run status", reasons)
        self.assertIn("hashes disagree", reasons)

    def test_tampered_plan_hash_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["analysis"]["primary_relative_radius"] = 0.10
            _write_json(plan, payload)
            output = root / "aggregate.json"
            completed = _run(plan, output)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("plan_sha256 mismatch", completed.stderr)
        self.assertFalse(output.exists())

    def test_rehashed_nonfactorial_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["tasks"].pop()
            payload["task_count"] = len(payload["tasks"])
            payload.pop("plan_sha256")
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["plan_sha256"] = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
            _write_json(plan, payload)
            output = root / "aggregate.json"
            completed = _run(plan, output)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("exactly cover the declared factorial axes", completed.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
