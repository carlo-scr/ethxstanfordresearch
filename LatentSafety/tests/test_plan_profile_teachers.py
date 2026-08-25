from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "plan_profile_teachers.py"
SPEC = importlib.util.spec_from_file_location("plan_profile_teachers", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load scripts/plan_profile_teachers.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProfileTeacherPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = MODULE.build_plan()

    def test_plan_has_exact_factorial_and_unique_outputs(self) -> None:
        self.assertEqual(self.plan["schema_version"], 2)
        self.assertEqual(self.plan["task_count"], 330)
        self.assertEqual(len(self.plan["plan_sha256"]), 64)
        tasks = self.plan["tasks"]
        self.assertEqual([task["task_id"] for task in tasks], list(range(330)))
        self.assertEqual(len({task["output_dir"] for task in tasks}), 330)
        self.assertTrue(
            all(task["task_sha256"] == MODULE.task_sha256(task) for task in tasks)
        )
        self.assertEqual(
            Counter(task["domain"] for task in tasks),
            Counter(
                {
                    "controlled_cart_video": 110,
                    "controlled_pendulum_video": 110,
                    "controlled_dubins_navigation_pixels": 110,
                }
            ),
        )

    def test_seeds_folds_and_teacher_seed_rule_are_exact(self) -> None:
        tasks = self.plan["tasks"]
        self.assertEqual({task["data_seed"] for task in tasks}, set(MODULE.DATA_SEEDS))
        self.assertEqual({task["fold_index"] for task in tasks}, set(range(5)))
        self.assertTrue(
            all(
                task["teacher_seed"]
                == 10_000 + 10 * task["data_seed"] + task["fold_index"]
                for task in tasks
            )
        )
        self.assertTrue(
            all(
                task["training_trajectory_count"]
                + task["held_out_trajectory_count"]
                == 208
                for task in tasks
            )
        )
        self.assertTrue(
            all("strict downstream ingestion" in task["runner_status"] for task in tasks)
        )
        self.assertTrue(
            all("production cell unexecuted" in task["runner_status"] for task in tasks)
        )
        self.assertEqual(
            self.plan["status"],
            "pipeline_components_ready_production_cells_and_gates_unexecuted",
        )


if __name__ == "__main__":
    unittest.main()
