from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import plan_profile_teachers as planner  # noqa: E402
import run_profile_teacher as runner  # noqa: E402
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.learning.data import generate_trajectories  # noqa: E402
from latent_safety.learning.profile_teacher import (  # noqa: E402
    ProfileTeacherError,
    build_teacher_split_spec,
    resolve_teacher_config,
    run_profile_teacher_task,
    validate_task_split_hashes,
)


def _resign_plan(plan: dict[str, object]) -> None:
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ProfileTeacherRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = planner.build_plan()
        cls.task = cls.plan["tasks"][0]
        cls.base = load_config(ROOT / cls.task["base_config"])

    def test_exact_plan_and_task_hashes_validate(self) -> None:
        verified = runner._verify_plan(json.loads(json.dumps(self.plan)))
        self.assertEqual(verified["task_count"], 330)
        self.assertEqual(
            self.task["task_sha256"], planner.task_sha256(dict(self.task))
        )

    def test_self_resigned_task_tampering_is_not_canonical(self) -> None:
        tampered = json.loads(json.dumps(self.plan))
        tampered_task = tampered["tasks"][0]
        tampered_task["teacher_seed"] = 999
        tampered_task["task_sha256"] = planner.task_sha256(tampered_task)
        _resign_plan(tampered)
        with self.assertRaisesRegex(ValueError, "canonical 330-task"):
            runner._verify_plan(tampered)

    def test_task_digest_and_source_hash_corruption_fail_closed(self) -> None:
        bad_task_hash = json.loads(json.dumps(self.plan))
        bad_task_hash["tasks"][0]["task_sha256"] = "0" * 64
        _resign_plan(bad_task_hash)
        with self.assertRaisesRegex(ValueError, "task_sha256 mismatch"):
            runner._verify_plan(bad_task_hash)

        bad_config_hash = json.loads(json.dumps(self.plan))
        domain = bad_config_hash["tasks"][0]["domain"]
        bad_config_hash["base_configs"][domain]["sha256"] = "0" * 64
        bad_config_hash["tasks"][0]["base_config_sha256"] = "0" * 64
        bad_config_hash["tasks"][0]["task_sha256"] = planner.task_sha256(
            bad_config_hash["tasks"][0]
        )
        _resign_plan(bad_config_hash)
        with self.assertRaisesRegex(ValueError, "base_configs.*sha256 mismatch"):
            runner._verify_plan(bad_config_hash)

    def test_absolute_and_escaped_output_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            runner._task_output_path(Path(tempfile.gettempdir()).as_posix())
        with self.assertRaisesRegex(ValueError, "profile_teachers"):
            runner._task_output_path("runs/e2_frontier/not_teacher/task")
        with self.assertRaisesRegex(ValueError, "escapes"):
            runner._task_output_path("../escaped")

    def test_registered_overrides_and_seed_separation_are_exact(self) -> None:
        resolved = resolve_teacher_config(self.base, self.task, device="cpu")
        self.assertEqual(resolved.run.seed, 10_000)
        self.assertEqual(self.task["data_seed"], 0)
        self.assertEqual(resolved.run.epochs, 30)
        self.assertEqual(resolved.model.family, "ae")
        self.assertEqual(resolved.model.history_encoder, "stack")
        self.assertEqual(resolved.data.history_length, 4)
        self.assertEqual(resolved.model.latent_dim, 8)
        self.assertEqual(resolved.model.hidden_dim, 128)
        self.assertEqual(resolved.model.transition_hidden_dim, 192)
        self.assertEqual(resolved.objective.safety_arm, "h_prediction")
        self.assertEqual(resolved.objective.safety_weight, 0.5)
        self.assertEqual(resolved.objective.kl_weight, 0.0)

        beta_task = self.plan["tasks"][55]
        beta = resolve_teacher_config(
            load_config(ROOT / beta_task["base_config"]),
            beta_task,
            device="cpu",
        )
        self.assertEqual(beta.model.family, "beta_vae")
        self.assertGreater(beta.objective.kl_weight, 0.0)

    def test_fold_partition_and_planned_hashes_are_exact(self) -> None:
        resolved = resolve_teacher_config(self.base, self.task, device="cpu")
        split_spec = build_teacher_split_spec(
            resolved,
            data_seed=int(self.task["data_seed"]),
            fold_index=int(self.task["fold_index"]),
        )
        validate_task_split_hashes(self.task, split_spec)
        self.assertEqual(len(split_spec.all_training_ids), 208)
        self.assertEqual(len(split_spec.fitting_ids), 166)
        self.assertEqual(len(split_spec.held_out_ids), 42)
        self.assertEqual(len(split_spec.validation_ids), 48)
        self.assertFalse(set(split_spec.fitting_ids) & set(split_spec.held_out_ids))
        self.assertFalse(set(split_spec.validation_ids) & set(split_spec.all_training_ids))

        corrupted = dict(self.task)
        corrupted["held_out_trajectory_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ProfileTeacherError, "counts/hashes"):
            validate_task_split_hashes(corrupted, split_spec)

    def test_observed_only_generation_matches_full_trajectory_identity(self) -> None:
        smoke = resolve_teacher_config(
            self.base,
            self.task,
            device="cpu",
            engineering_smoke=True,
        )
        full = generate_trajectories(smoke.data, seed=int(self.task["data_seed"]))
        observed = generate_trajectories(
            smoke.data,
            seed=int(self.task["data_seed"]),
            include_splits=("train", "validation"),
            include_action_profiles=False,
        )
        full_by_id = {trajectory.trajectory_id: trajectory for trajectory in full}
        self.assertEqual({trajectory.split for trajectory in observed}, {"train", "validation"})
        self.assertTrue(all(not trajectory.action_safety_margins for trajectory in observed))
        for trajectory in observed:
            reference = full_by_id[trajectory.trajectory_id]
            self.assertEqual(trajectory.states, reference.states)
            self.assertEqual(trajectory.actions, reference.actions)
            self.assertEqual(trajectory.safety_margins, reference.safety_margins)
            self.assertTrue(reference.action_safety_margins)

    def test_generation_access_control_rejects_invalid_split_requests(self) -> None:
        smoke = resolve_teacher_config(
            self.base,
            self.task,
            device="cpu",
            engineering_smoke=True,
        )
        with self.assertRaisesRegex(ValueError, "unique registered"):
            generate_trajectories(smoke.data, seed=0, include_splits=())
        with self.assertRaisesRegex(ValueError, "unique registered"):
            generate_trajectories(
                smoke.data,
                seed=0,
                include_splits=("train", "train"),
            )
        with self.assertRaisesRegex(TypeError, "boolean"):
            generate_trajectories(
                smoke.data,
                seed=0,
                include_action_profiles=1,  # type: ignore[arg-type]
            )

    def test_existing_output_is_immutable_before_torch_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileExistsError, "even if empty"):
                run_profile_teacher_task(
                    self.base,
                    self.task,
                    base_config_path=str(self.task["base_config"]),
                    plan_sha256=str(self.plan["plan_sha256"]),
                    planned_code=self.plan["code"],
                    output_dir=Path(directory),
                    device="cpu",
                    producing_command=("python", "scripts/run_profile_teacher.py"),
                    evidence_eligible=False,
                )

    def test_cli_dry_run_executes_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
            output = Path(directory) / "must-not-exist"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_profile_teacher.py"),
                    "--plan",
                    str(plan_path),
                    "--index",
                    "0",
                    "--engineering-smoke",
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["engineering_smoke"])
            self.assertFalse(payload["evidence_eligible"])
            self.assertEqual(payload["trajectory_counts"]["fitting"], 10)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
