from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "configs" / "e1_world_models" / "pilot_grid.toml"
DECISIVE_GRIDS = (
    ROOT / "configs" / "e1_world_models" / "decisive_cart_grid.toml",
    ROOT / "configs" / "e1_world_models" / "decisive_pendulum_grid.toml",
)


class SweepPlanTests(unittest.TestCase):
    @staticmethod
    def _resign(plan: dict[str, object]) -> None:
        unsigned = dict(plan)
        unsigned.pop("plan_sha256", None)
        plan["plan_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def test_decisive_two_domain_gate_has_exactly_96_tasks(self) -> None:
        plans = []
        with tempfile.TemporaryDirectory() as directory:
            for grid_index, grid in enumerate(DECISIVE_GRIDS):
                plan_path = Path(directory) / f"decisive_{grid_index}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "plan_e1_sweep.py"),
                        "--grid",
                        str(grid),
                        "--output",
                        str(plan_path),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                plans.append(plan)
                for index in (0, 47):
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                            "--plan",
                            str(plan_path),
                            "--index",
                            str(index),
                            "--device",
                            "cpu",
                            "--dry-run",
                        ],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )

        self.assertEqual(sum(plan["task_count"] for plan in plans), 96)
        for plan in plans:
            self.assertEqual(plan["task_count"], 48)
            self.assertFalse(Path(plan["grid"]["path"]).is_absolute())
            self.assertFalse(Path(plan["base_config"]["path"]).is_absolute())
            self.assertEqual(set(plan["code"]), {"git_dirty", "git_revision"})
            self.assertEqual(plan["axes"]["history_modes"], ["stack_h1", "stack_h4"])
            self.assertEqual(plan["axes"]["latent_dimensions"], [8])
            self.assertEqual(plan["axes"]["seeds"], [0, 1, 2])
            self.assertEqual(len({task["output_dir"] for task in plan["tasks"]}), 48)

    def test_factorial_plan_is_complete_unique_and_dry_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plan_e1_sweep.py"),
                    "--grid",
                    str(GRID),
                    "--output",
                    str(plan_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(plan_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            corrupt_plan = json.loads(json.dumps(plan))
            corrupt_plan["tasks"][0]["history_length"] = 4
            corrupt_unsigned = dict(corrupt_plan)
            corrupt_unsigned.pop("plan_sha256")
            corrupt_plan["plan_sha256"] = hashlib.sha256(
                json.dumps(
                    corrupt_unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            corrupt_plan_path = Path(directory) / "corrupt_plan.json"
            corrupt_plan_path.write_text(json.dumps(corrupt_plan), encoding="utf-8")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(corrupt_plan_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            tampered_plan = json.loads(json.dumps(plan))
            tampered_plan["tasks"][0]["seed"] = 999
            tampered_plan_path = Path(directory) / "tampered_plan.json"
            tampered_plan_path.write_text(json.dumps(tampered_plan), encoding="utf-8")
            tamper_rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(tampered_plan_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            source_tampered_plan = json.loads(json.dumps(plan))
            source_tampered_plan["base_config"]["sha256"] = "0" * 64
            self._resign(source_tampered_plan)
            source_tampered_path = Path(directory) / "source_tampered_plan.json"
            source_tampered_path.write_text(
                json.dumps(source_tampered_plan), encoding="utf-8"
            )
            source_tamper_rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(source_tampered_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            absolute_source_plan = json.loads(json.dumps(plan))
            absolute_source_plan["base_config"]["path"] = str(
                ROOT / "configs" / "e1_world_models" / "torch_pilot.toml"
            )
            self._resign(absolute_source_plan)
            absolute_source_path = Path(directory) / "absolute_source_plan.json"
            absolute_source_path.write_text(
                json.dumps(absolute_source_plan), encoding="utf-8"
            )
            absolute_source_rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(absolute_source_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            escaped_output_plan = json.loads(json.dumps(plan))
            escaped_output_plan["tasks"][0]["output_dir"] = "../escaped-output"
            self._resign(escaped_output_plan)
            escaped_output_path = Path(directory) / "escaped_output_plan.json"
            escaped_output_path.write_text(
                json.dumps(escaped_output_plan), encoding="utf-8"
            )
            escaped_output_rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(escaped_output_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            dirty_plan = json.loads(json.dumps(plan))
            dirty_plan["code"] = {
                "git_revision": "engineering-only-revision",
                "git_dirty": True,
            }
            self._resign(dirty_plan)
            dirty_plan_path = Path(directory) / "dirty_plan.json"
            dirty_plan_path.write_text(json.dumps(dirty_plan), encoding="utf-8")
            dirty_plan_rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(dirty_plan_path),
                    "--index",
                    "0",
                    "--device",
                    "cpu",
                    "--skip-audit",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            resigned_task_plan = json.loads(json.dumps(plan))
            resigned_task_plan["tasks"][1]["safety_weight"] = 123.0
            self._resign(resigned_task_plan)
            resigned_task_path = Path(directory) / "resigned_task_plan.json"
            resigned_task_path.write_text(
                json.dumps(resigned_task_plan), encoding="utf-8"
            )
            resigned_task_rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_e1_sweep_task.py"),
                    "--plan",
                    str(resigned_task_path),
                    "--index",
                    "1",
                    "--device",
                    "cpu",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(plan["task_count"], 144)
        self.assertEqual(len({task["output_dir"] for task in plan["tasks"]}), 144)
        self.assertEqual({task["seed"] for task in plan["tasks"]}, {0, 1, 2})
        self.assertEqual(
            plan["history_mode_definitions"],
            {
                "stack_h1": {"history_encoder": "stack", "history_length": 1},
                "stack_h4": {"history_encoder": "stack", "history_length": 4},
                "gru_h4": {"history_encoder": "gru", "history_length": 4},
            },
        )
        tasks_by_mode = {
            mode: [task for task in plan["tasks"] if task["history_mode"] == mode]
            for mode in plan["axes"]["history_modes"]
        }
        self.assertEqual(
            {mode: len(tasks) for mode, tasks in tasks_by_mode.items()},
            {
                "stack_h1": 48,
                "stack_h4": 48,
                "gru_h4": 48,
            },
        )
        for mode, definition in plan["history_mode_definitions"].items():
            with self.subTest(history_mode=mode):
                self.assertTrue(
                    all(
                        task["history_encoder"] == definition["history_encoder"]
                        and task["history_length"] == definition["history_length"]
                        and f"_{mode}_" in task["output_dir"]
                        for task in tasks_by_mode[mode]
                    )
                )
        dry_run = json.loads(completed.stdout)
        self.assertFalse(dry_run["torch_required"])
        self.assertEqual(dry_run["requested_device"], "cpu")
        self.assertEqual(dry_run["representation"]["history_mode"], "stack_h1")
        self.assertEqual(dry_run["representation"]["history_encoder"], "stack")
        self.assertEqual(dry_run["representation"]["history_length"], 1)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("disagrees with its history_mode definition", rejected.stderr)
        self.assertNotEqual(tamper_rejected.returncode, 0)
        self.assertIn("plan_sha256 mismatch", tamper_rejected.stderr)
        self.assertNotEqual(source_tamper_rejected.returncode, 0)
        self.assertIn("plan.base_config.sha256 mismatch", source_tamper_rejected.stderr)
        self.assertNotEqual(absolute_source_rejected.returncode, 0)
        self.assertIn("must be repository-relative", absolute_source_rejected.stderr)
        self.assertNotEqual(escaped_output_rejected.returncode, 0)
        self.assertIn("must resolve below runs/e1_world_models", escaped_output_rejected.stderr)
        self.assertNotEqual(dirty_plan_rejected.returncode, 0)
        self.assertIn("same clean committed Git revision", dirty_plan_rejected.stderr)
        self.assertNotEqual(resigned_task_rejected.returncode, 0)
        self.assertIn("not the canonical expansion", resigned_task_rejected.stderr)


if __name__ == "__main__":
    unittest.main()
