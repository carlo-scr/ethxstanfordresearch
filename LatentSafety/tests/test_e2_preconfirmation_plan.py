from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latent_safety.analysis.preconfirmation import (  # noqa: E402
    CONTROL_ARM,
    DOMAINS,
    EXPECTED_OBSERVATIONS,
    LEARNED_ARMS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    POSITIVE_WEIGHTS,
    PROFILE_ARM,
)
from plan_e2_preconfirmation import (  # noqa: E402
    INTERVENTION_CONFIG,
    OUTPUT_ROOT,
    READY_PLAN_STATUS,
    _profile_input_cell,
    _profile_teacher_plan_entry,
    _validate_intervention,
    build_plan,
    canonical_sha256,
    file_sha256,
    task_sha256,
)
import run_e2_preconfirmation_task as e2_runner  # noqa: E402
from run_e2_preconfirmation_task import _task_output_path  # noqa: E402
from run_predicted_profile_arm import _resolve_e2_task_binding  # noqa: E402


class E2PreconfirmationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan()

    def test_exact_288_row_factorial_and_canonical_order(self) -> None:
        plan = self.plan
        tasks = plan["tasks"]
        self.assertEqual(plan["task_count"], EXPECTED_OBSERVATIONS)
        self.assertEqual(plan["counts"], {"controls": 18, "learned": 270, "total": 288})
        self.assertEqual([task["task_id"] for task in tasks], list(range(288)))

        expected_controls = [
            (domain, family, CONTROL_ARM, seed, None)
            for domain, family, seed in itertools.product(
                DOMAINS, MODEL_FAMILIES, PILOT_SEEDS
            )
        ]
        expected_learned = [
            (domain, family, arm, seed, float(weight))
            for domain, family, arm, weight, seed in itertools.product(
                DOMAINS,
                MODEL_FAMILIES,
                LEARNED_ARMS,
                POSITIVE_WEIGHTS,
                PILOT_SEEDS,
            )
        ]
        observed = [
            (
                task["domain"],
                task["model_family"],
                task["safety_arm"],
                task["seed"],
                task["regularization_weight"],
            )
            for task in tasks
        ]
        self.assertEqual(observed[:18], expected_controls)
        self.assertEqual(observed[18:], expected_learned)
        self.assertEqual(len(set(observed)), 288)

    def test_tasks_freeze_training_semantics_and_managed_unique_paths(self) -> None:
        managed = (ROOT / OUTPUT_ROOT).resolve()
        outputs: set[Path] = set()
        for task in self.plan["tasks"]:
            with self.subTest(task_id=task["task_id"]):
                self.assertEqual(task["history_mode"], "stack_h4")
                self.assertEqual(task["history_encoder"], "stack")
                self.assertEqual(task["history_length"], 4)
                self.assertEqual(task["latent_dim"], 8)
                self.assertEqual(task["fcsrl_head_hidden_dim"], 64)
                self.assertEqual(task["selection_split"], "validation_only")
                self.assertFalse(task["calibration_or_final_test_selection_access"])
                self.assertEqual(task["task_sha256"], task_sha256(task))
                self.assertEqual(
                    task["safety_weight"],
                    0.0
                    if task["safety_arm"] == CONTROL_ARM
                    else task["regularization_weight"],
                )
                self.assertEqual(
                    task["kl_weight"],
                    0.0 if task["model_family"] == "ae" else 0.0005,
                )
                output = (ROOT / task["output_dir"]).resolve()
                output.relative_to(managed)
                self.assertNotEqual(output, managed)
                outputs.add(output)
        self.assertEqual(len(outputs), 288)

    def test_profile_cells_reuse_one_checksum_pinned_handoff_for_five_weights(self) -> None:
        profile_tasks = [
            task for task in self.plan["tasks"] if task["safety_arm"] == PROFILE_ARM
        ]
        self.assertEqual(len(profile_tasks), 90)
        cell_inventory = {
            (cell["domain"], cell["model_family"], cell["seed"]): cell
            for cell in self.plan["profile_handoff"]["cells"]
        }
        self.assertEqual(len(cell_inventory), 18)
        for key in itertools.product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS):
            rows = [
                task
                for task in profile_tasks
                if (task["domain"], task["model_family"], task["seed"]) == key
            ]
            with self.subTest(cell=key):
                self.assertEqual(len(rows), 5)
                self.assertEqual(
                    {row["regularization_weight"] for row in rows},
                    set(POSITIVE_WEIGHTS),
                )
                self.assertEqual(len({row["profile_label_manifest"] for row in rows}), 1)
                self.assertEqual(len({row["profile_coverage_manifest"] for row in rows}), 1)
                cell = cell_inventory[key]
                self.assertEqual(rows[0]["profile_label_manifest"], cell["assembly"]["path"])
                self.assertEqual(
                    rows[0]["profile_coverage_manifest"], cell["coverage"]["path"]
                )
                for field, location, key_name in (
                    ("profile_label_manifest_sha256", "assembly", "manifest_sha256"),
                    ("profile_label_manifest_file_sha256", "assembly", "file_sha256"),
                    ("profile_coverage_manifest_sha256", "coverage", "manifest_sha256"),
                    ("profile_coverage_manifest_file_sha256", "coverage", "file_sha256"),
                ):
                    self.assertEqual(
                        {row[field] for row in rows}, {cell[location][key_name]}
                    )
                self.assertEqual(
                    {row["profile_teacher_plan_sha256"] for row in rows},
                    {cell["teacher_plan_sha256"]},
                )
                teacher_plan = self.plan["profile_handoff"]["teacher_plan"]
                self.assertEqual(
                    {row["profile_teacher_plan"] for row in rows},
                    {teacher_plan["path"]},
                )
                self.assertEqual(
                    {row["profile_teacher_plan_file_sha256"] for row in rows},
                    {teacher_plan["file_sha256"]},
                )

        ordinary = [task for task in self.plan["tasks"] if task["safety_arm"] != PROFILE_ARM]
        self.assertTrue(
            all(
                task["profile_label_manifest"] is None
                and task["profile_coverage_manifest"] is None
                and task["profile_teacher_plan"] is None
                and task["profile_teacher_plan_sha256"] is None
                and task["profile_teacher_plan_file_sha256"] is None
                for task in ordinary
            )
        )

    def test_plan_hashes_source_files_and_is_reproducible(self) -> None:
        plan = self.plan
        unsigned = dict(plan)
        declared = unsigned.pop("plan_sha256")
        self.assertEqual(declared, canonical_sha256(unsigned))
        self.assertEqual(build_plan(), plan)
        intervention = ROOT / plan["intervention_config"]["path"]
        self.assertEqual(
            plan["intervention_config"]["sha256"],
            hashlib.sha256(intervention.read_bytes()).hexdigest(),
        )
        for entry in plan["base_configs"].values():
            self.assertEqual(
                entry["sha256"],
                hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(),
            )
        ready = plan["profile_handoff"]["all_inputs_ready"]
        self.assertEqual(plan["runner"]["execution_enabled"], ready)
        self.assertEqual(plan["runner"]["generic_execution_enabled"], ready)
        self.assertTrue(plan["runner"]["authenticated_task_handoff_implemented"])
        self.assertIn("downstream", plan["runner"]["downstream_status"])

    def test_intervention_drift_and_alternate_source_paths_fail_closed(self) -> None:
        with (ROOT / INTERVENTION_CONFIG).open("rb") as stream:
            payload = tomllib.load(stream)
        payload["run"]["seeds"] = [0, 1, 3]
        with self.assertRaisesRegex(ValueError, "run.seeds must be exactly"):
            _validate_intervention(payload)

        with (ROOT / INTERVENTION_CONFIG).open("rb") as stream:
            canonical = tomllib.load(stream)
        drifts = (
            ("selection", "utility_noninferiority", "each utility ratio at most 1.10"),
            ("eligibility", "minimum_eligible_center_coverage", 0.75),
            (
                "eligibility",
                "max_relative_neighborhood_mass_mismatch",
                0.10,
            ),
            ("matched_radius_reduction", "control_reference_relative_radius", 0.10),
            ("matched_radius_reduction", "trajectory_tail_quantile", 0.90),
        )
        for table, field, value in drifts:
            drifted = copy.deepcopy(canonical)
            drifted[table][field] = value
            with self.subTest(table=table, field=field), self.assertRaisesRegex(
                ValueError, rf"\[{table}\].*frozen E2 protocol"
            ):
                _validate_intervention(drifted)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "intervention.toml"
            copied.write_bytes((ROOT / INTERVENTION_CONFIG).read_bytes())
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                build_plan(copied)

    def test_digest_helpers_do_not_self_attest(self) -> None:
        task = dict(self.plan["tasks"][18])
        original = task.pop("task_sha256")
        self.assertEqual(original, canonical_sha256(task))
        task["safety_weight"] = 999.0
        self.assertNotEqual(original, canonical_sha256(task))
        task["task_sha256"] = original
        self.assertNotEqual(task_sha256(task), original)

    def test_planner_refuses_to_overwrite_existing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "plan_e2_preconfirmation.py"),
                "--output",
                str(output),
            ]
            first = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            second = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual(output.read_bytes(), original)

    def test_materialized_profile_inputs_freeze_raw_and_self_hashes(self) -> None:
        domain = DOMAINS[0]
        family = MODEL_FAMILIES[0]
        seed = PILOT_SEEDS[0]
        teacher_plan_sha = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell_root = (
                root
                / "runs"
                / "e2_frontier"
                / "profile_labels"
                / domain
                / family
                / f"data_seed_{seed}"
            )
            coverage_root = (
                root
                / "runs"
                / "e2_frontier"
                / "profile_coverage"
                / domain
                / family
                / f"data_seed_{seed}"
            )
            cell_root.mkdir(parents=True)
            coverage_root.mkdir(parents=True)
            assembly = {
                "schema_version": 1,
                "status": "complete_crossfit_label_manifest",
                "semantic_arm": PROFILE_ARM,
                "plan_sha256": teacher_plan_sha,
                "domain": domain,
                "model_family": family,
                "data_seed": seed,
                "engineering_smoke": False,
                "evidence_eligible": True,
            }
            assembly["manifest_sha256"] = canonical_sha256(assembly)
            assembly_path = cell_root / "crossfit_label_manifest.json"
            assembly_path.write_text(json.dumps(assembly), encoding="utf-8")
            coverage = {
                "schema_version": 1,
                "status": "gate_passed",
                "scientific_gate_passed": True,
                "evidence_eligible": True,
                "plan_sha256": teacher_plan_sha,
                "domain": domain,
                "model_family": family,
                "data_seed": seed,
                "engineering_smoke": False,
                "assembly": {
                    "manifest_sha256": assembly["manifest_sha256"],
                    "file_sha256": file_sha256(assembly_path),
                },
            }
            coverage["manifest_sha256"] = canonical_sha256(coverage)
            coverage_path = coverage_root / "profile_coverage_gate.json"
            coverage_path.write_text(json.dumps(coverage), encoding="utf-8")

            teacher_plan_path = root / "runs/e2_frontier/profile_teacher_plan.json"
            teacher_plan = {
                "schema_version": 2,
                "task_count": 330,
                "code": {"git_revision": "clean-revision", "git_dirty": False},
            }
            teacher_plan["plan_sha256"] = canonical_sha256(teacher_plan)
            teacher_plan_path.write_text(json.dumps(teacher_plan), encoding="utf-8")

            with patch("plan_e2_preconfirmation.ROOT", root):
                cell = _profile_input_cell(domain, family, seed)
                teacher = _profile_teacher_plan_entry(teacher_plan_path)
            self.assertTrue(cell["ready_for_evidence"])
            self.assertEqual(cell["teacher_plan_sha256"], teacher_plan_sha)
            self.assertEqual(cell["assembly"]["file_sha256"], file_sha256(assembly_path))
            self.assertEqual(cell["coverage"]["file_sha256"], file_sha256(coverage_path))
            self.assertTrue(teacher["ready_for_evidence"])
            self.assertEqual(teacher["plan_sha256"], teacher_plan["plan_sha256"])

            coverage["status"] = "gate_failed"
            coverage.pop("manifest_sha256")
            coverage["manifest_sha256"] = canonical_sha256(coverage)
            coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
            with patch("plan_e2_preconfirmation.ROOT", root), self.assertRaisesRegex(
                ValueError, "not evidence-ready"
            ):
                _profile_input_cell(domain, family, seed)


class E2PreconfirmationRunnerTests(unittest.TestCase):
    @staticmethod
    def _resign(plan: dict[str, object]) -> None:
        unsigned = dict(plan)
        unsigned.pop("plan_sha256", None)
        plan["plan_sha256"] = canonical_sha256(unsigned)

    @staticmethod
    def _run(plan_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_e2_preconfirmation_task.py"),
                "--plan",
                str(plan_path),
                *arguments,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_indexed_dry_run_authenticates_plan_without_writing(self) -> None:
        plan = build_plan()
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            profile_index = next(
                task["task_id"]
                for task in plan["tasks"]
                if task["safety_arm"] == PROFILE_ARM
            )
            for index in (0, 17, 18, profile_index, 287):
                completed = self._run(
                    plan_path,
                    "--index",
                    str(index),
                    "--device",
                    "cpu",
                    "--dry-run",
                )
                with self.subTest(index=index, stderr=completed.stderr):
                    self.assertEqual(completed.returncode, 0)
                    payload = json.loads(completed.stdout)
                    self.assertEqual(payload["task"]["task_id"], index)
                    self.assertEqual(payload["resolved_device"], "cpu")
                    self.assertFalse(payload["evidence_written"])
                    self.assertEqual(
                        payload["execution_scope"]["materialized_splits"],
                        ["train", "validation"],
                    )
                    self.assertFalse(
                        payload["execution_scope"]["final_test_materialized"]
                    )
                    self.assertTrue(
                        payload["execution_scope"][
                            "fit_oracle_action_profiles_forbidden"
                        ]
                    )
                    self.assertTrue(
                        payload["postfit_validation_audit"]["implemented"]
                    )
                    self.assertEqual(
                        payload["execution_ready"],
                        (
                            plan["status"] == READY_PLAN_STATUS
                            and (
                                payload["task"]["safety_arm"] != PROFILE_ARM
                                or payload["task"][
                                    "profile_teacher_plan_sha256"
                                ]
                                is not None
                            )
                        ),
                    )
                    self.assertEqual(
                        bool(payload["blocked_dependencies"]),
                        not payload["execution_ready"],
                    )
                    self.assertTrue(payload["downstream_dependencies"])
                    invocation = payload["fit_invocation"]
                    if payload["task"]["safety_arm"] == PROFILE_ARM:
                        self.assertEqual(invocation["kind"], "subprocess")
                        command = invocation["command"]
                        self.assertIn("run_predicted_profile_arm.py", command[1])
                        self.assertEqual(
                            command[command.index("--e2-task-id") + 1], str(index)
                        )
                        self.assertNotEqual(
                            invocation["authenticated_binding"][
                                "teacher_plan_sha256"
                            ],
                            invocation["authenticated_binding"][
                                "orchestration_plan_sha256"
                            ],
                        )
                    else:
                        self.assertEqual(invocation["kind"], "subprocess")
                        command = invocation["command"]
                        scope_index = command.index("--access-scope") + 1
                        self.assertEqual(command[scope_index], "train_validation_only")

    def test_tampered_and_resigned_noncanonical_plans_are_rejected(self) -> None:
        plan = build_plan()
        tampered = json.loads(json.dumps(plan))
        tampered["tasks"][0]["seed"] = 999
        escaped = json.loads(json.dumps(plan))
        escaped["tasks"][0]["output_dir"] = "../escaped"
        escaped["tasks"][0]["task_sha256"] = task_sha256(escaped["tasks"][0])
        self._resign(escaped)
        resigned = json.loads(json.dumps(plan))
        resigned["tasks"][18]["safety_weight"] = 123.0
        resigned["tasks"][18]["task_sha256"] = task_sha256(resigned["tasks"][18])
        self._resign(resigned)
        source = json.loads(json.dumps(plan))
        source["intervention_config"]["sha256"] = "0" * 64
        self._resign(source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "tampered": (tampered, "plan_sha256 mismatch"),
                "escaped": (escaped, "escapes the repository"),
                "resigned": (resigned, "not the canonical expansion"),
                "source": (source, "sha256 mismatch"),
            }
            for name, (payload, message) in cases.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                completed = self._run(path, "--index", "0", "--dry-run")
                with self.subTest(case=name, stderr=completed.stderr):
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(message, completed.stderr)

    def test_invalid_index_and_external_execution_plan_fail_without_output(self) -> None:
        plan = build_plan()
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            invalid = self._run(plan_path, "--index", "288", "--dry-run")
            blocked = self._run(
                plan_path,
                "--index",
                "0",
                "--allow-unversioned",
            )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("--index must lie in [0, 287]", invalid.stderr)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("--plan to resolve inside the repository", blocked.stderr)

    def test_generic_execution_waits_for_all_profile_handoffs(self) -> None:
        plan = build_plan()
        plan["status"] = "blocked_on_production_profile_inputs"
        plan["profile_handoff"]["all_inputs_ready"] = False
        plan["runner"]["execution_enabled"] = False
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            plan_path = Path(directory) / "e2_plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            stderr = io.StringIO()
            with (
                patch.object(e2_runner, "_verify_plan", return_value=plan),
                patch.object(e2_runner, "_run_task_subprocess") as launch,
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                e2_runner.main(
                    [
                        "--plan",
                        str(plan_path.relative_to(ROOT)),
                        "--index",
                        "0",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("all 18 production", stderr.getvalue())
        launch.assert_not_called()

    def test_simultaneous_duplicate_launches_have_exactly_one_atomic_claim(self) -> None:
        plan = build_plan()
        task = plan["tasks"][0]
        start = Barrier(2)
        attempted = Barrier(2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ordinary_task"

            def contend() -> tuple[bool, object]:
                claim = None
                result: tuple[bool, object]
                start.wait()
                try:
                    try:
                        claim = e2_runner._acquire_task_claim(
                            output_dir=output,
                            plan=plan,
                            task=task,
                        )
                        payload = json.loads(claim.path.read_text(encoding="utf-8"))
                        self.assertEqual(claim.path.stat().st_mode & 0o777, 0o600)
                        result = (True, payload)
                    except ValueError as error:
                        result = (False, str(error))
                finally:
                    attempted.wait()
                    if claim is not None:
                        claim.release()
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: contend(), range(2)))

            winners = [payload for acquired, payload in results if acquired]
            losers = [message for acquired, message in results if not acquired]
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(losers), 1)
            self.assertEqual(winners[0]["plan_sha256"], plan["plan_sha256"])
            self.assertEqual(winners[0]["task_sha256"], task["task_sha256"])
            self.assertIn("already claimed", losers[0])
            self.assertFalse(e2_runner._task_claim_path(output).exists())

    def test_stale_claim_fails_closed_and_explicit_owner_cleanup_allows_retry(
        self,
    ) -> None:
        plan = build_plan()
        task = plan["tasks"][0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ordinary_task"
            first = e2_runner._acquire_task_claim(
                output_dir=output,
                plan=plan,
                task=task,
            )
            try:
                with self.assertRaisesRegex(ValueError, "operator review"):
                    e2_runner._acquire_task_claim(
                        output_dir=output,
                        plan=plan,
                        task=task,
                    )
            finally:
                first.release()
            retry = e2_runner._acquire_task_claim(
                output_dir=output,
                plan=plan,
                task=task,
            )
            retry.release()
            self.assertFalse(e2_runner._task_claim_path(output).exists())

    def test_failed_launch_with_partial_output_releases_claim_but_remains_single_use(
        self,
    ) -> None:
        plan = build_plan()
        task = plan["tasks"][0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ordinary_task"
            claim = e2_runner._acquire_task_claim(
                output_dir=output,
                plan=plan,
                task=task,
            )
            output.mkdir()
            claim.release()
            self.assertFalse(e2_runner._task_claim_path(output).exists())
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                e2_runner._acquire_task_claim(
                    output_dir=output,
                    plan=plan,
                    task=task,
                )
            self.assertFalse(e2_runner._task_claim_path(output).exists())

    def _invoke_mocked_main(
        self,
        *,
        subprocess_returncode: int = 0,
    ) -> tuple[int, str, str, object, object, list[str], bool]:
        plan = build_plan()
        plan["status"] = READY_PLAN_STATUS
        plan["profile_handoff"]["all_inputs_ready"] = True
        plan["runner"]["execution_enabled"] = True
        task = plan["tasks"][0]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory)
            plan_path = temporary / "e2_plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            output = temporary / "task_output"
            handoff = output / "preconfirmation_task_manifest.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            current_code = {"git_revision": "current", "git_dirty": False}
            authenticated = {
                "manifest_sha256": "f" * 64,
                "scope": {
                    "selection_split": "validation_only",
                    "calibration_access": False,
                    "final_test_access": False,
                },
            }
            arguments = [
                "--plan",
                str(plan_path.relative_to(ROOT)),
                "--index",
                "0",
            ]
            with (
                patch.object(e2_runner, "_verify_plan", return_value=plan),
                patch.object(
                    e2_runner,
                    "_code_matches",
                    return_value=(True, current_code),
                ),
                patch.object(e2_runner, "_task_output_path", return_value=output),
                patch.object(
                    e2_runner,
                    "_run_task_subprocess",
                    return_value=subprocess_returncode,
                ) as launch,
                patch.object(
                    e2_runner,
                    "write_preconfirmation_task_manifest",
                    return_value=handoff,
                ) as write_handoff,
                patch.object(
                    e2_runner,
                    "authenticate_preconfirmation_task_manifest",
                    return_value=(authenticated, handoff),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = e2_runner.main(arguments)
            command = launch.call_args.args[0] if launch.called else []
            claim_remained = e2_runner._task_claim_path(output).exists()
            return (
                result,
                stdout.getvalue(),
                stderr.getvalue(),
                launch,
                write_handoff,
                command,
                claim_remained,
            )

    def test_successful_task_launch_writes_authenticated_uniform_handoff(self) -> None:
        result, stdout, stderr, launch, write_handoff, command, claim_remained = (
            self._invoke_mocked_main()
        )
        self.assertEqual(result, 0, stderr)
        self.assertTrue(launch.called)
        self.assertTrue(write_handoff.called)
        payload = json.loads(stdout)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["evidence_eligible"])
        self.assertTrue(payload["evidence_written"])
        self.assertEqual(
            command[command.index("--access-scope") + 1], "train_validation_only"
        )
        self.assertNotIn("calibration", " ".join(command).lower())
        self.assertNotIn("final-test", " ".join(command).lower())
        self.assertFalse(claim_remained)

    def test_failed_subprocess_never_writes_task_handoff(self) -> None:
        result, stdout, stderr, launch, write_handoff, _, claim_remained = (
            self._invoke_mocked_main(subprocess_returncode=7)
        )
        self.assertEqual(result, 7)
        self.assertEqual(stdout, "")
        self.assertTrue(launch.called)
        self.assertFalse(write_handoff.called)
        self.assertFalse(claim_remained)

    def test_existing_output_refuses_overwrite_before_launch(self) -> None:
        plan = build_plan()
        plan["status"] = READY_PLAN_STATUS
        plan["profile_handoff"]["all_inputs_ready"] = True
        plan["runner"]["execution_enabled"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory)
            plan_path = temporary / "e2_plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            output = temporary / "already_exists"
            output.mkdir()
            claim_path = e2_runner._task_claim_path(output)
            stderr = io.StringIO()
            with (
                patch.object(e2_runner, "_verify_plan", return_value=plan),
                patch.object(
                    e2_runner,
                    "_code_matches",
                    return_value=(True, {"git_revision": "x", "git_dirty": False}),
                ),
                patch.object(e2_runner, "_task_output_path", return_value=output),
                patch.object(e2_runner, "_run_task_subprocess") as launch,
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                e2_runner.main(
                    [
                        "--plan",
                        str(plan_path.relative_to(ROOT)),
                        "--index",
                        "0",
                    ]
                )
            claim_remained = claim_path.exists()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("refusing to overwrite", stderr.getvalue())
        self.assertFalse(claim_remained)
        launch.assert_not_called()

    def test_allow_unversioned_never_consumes_canonical_output_or_writes_handoff(
        self,
    ) -> None:
        plan = build_plan()
        plan["status"] = READY_PLAN_STATUS
        plan["profile_handoff"]["all_inputs_ready"] = True
        plan["runner"]["execution_enabled"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory)
            plan_path = temporary / "e2_plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            output = temporary / "not_consumed"
            stderr = io.StringIO()
            with (
                patch.object(e2_runner, "_verify_plan", return_value=plan),
                patch.object(e2_runner, "_task_output_path", return_value=output),
                patch.object(e2_runner, "_run_task_subprocess") as launch,
                patch.object(
                    e2_runner, "write_preconfirmation_task_manifest"
                ) as write_handoff,
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                e2_runner.main(
                    [
                        "--plan",
                        str(plan_path.relative_to(ROOT)),
                        "--index",
                        "0",
                        "--allow-unversioned",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("single-use E2 output", stderr.getvalue())
        self.assertFalse(output.exists())
        launch.assert_not_called()
        write_handoff.assert_not_called()

    def test_production_device_override_must_equal_frozen_task_device(self) -> None:
        plan = build_plan()
        plan["status"] = READY_PLAN_STATUS
        plan["profile_handoff"]["all_inputs_ready"] = True
        plan["runner"]["execution_enabled"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory)
            plan_path = temporary / "e2_plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            stderr = io.StringIO()
            with (
                patch.object(e2_runner, "_verify_plan", return_value=plan),
                patch.object(e2_runner, "_run_task_subprocess") as launch,
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                e2_runner.main(
                    [
                        "--plan",
                        str(plan_path.relative_to(ROOT)),
                        "--index",
                        "0",
                        "--device",
                        "cpu",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("plan-declared device 'cuda'", stderr.getvalue())
        launch.assert_not_called()

    def test_profile_cli_accepts_only_exact_authenticated_e2_task_output(self) -> None:
        plan = build_plan()
        source = next(task for task in plan["tasks"] if task["safety_arm"] == PROFILE_ARM)
        task = dict(source)
        task.update(
            {
                "task_id": 0,
                "task_sha256": "c" * 64,
                "profile_teacher_plan_sha256": "b" * 64,
            }
        )
        e2_plan = {
            "plan_sha256": "a" * 64,
            "status": READY_PLAN_STATUS,
            "profile_handoff": {"all_inputs_ready": True},
            "runner": {"execution_enabled": True},
            "tasks": [task],
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory)
            plan_path = temporary / "e2_plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            planned_output = temporary / "planned_output"
            output_value = planned_output.relative_to(ROOT)
            teacher = (ROOT / str(task["profile_teacher_plan"])).resolve()
            assembly = (ROOT / str(task["profile_label_manifest"])).resolve()
            coverage = (ROOT / str(task["profile_coverage_manifest"])).resolve()
            common = {
                "e2_plan_value": plan_path.relative_to(ROOT),
                "e2_task_id": 0,
                "e2_task_sha256": "c" * 64,
                "orchestration_plan_sha256": "a" * 64,
                "teacher_plan_path": teacher,
                "teacher_plan_sha256": "b" * 64,
                "domain": task["domain"],
                "model_family": task["model_family"],
                "data_seed": task["seed"],
                "safety_weight": task["safety_weight"],
                "device": task["device"],
                "assembly_path": assembly,
                "coverage_path": coverage,
            }
            with (
                patch("run_e2_preconfirmation_task._verify_plan", return_value=e2_plan),
                patch("run_e2_preconfirmation_task._verify_pinned_profile_inputs"),
                patch(
                    "run_e2_preconfirmation_task._task_output_path",
                    return_value=planned_output,
                ),
            ):
                resolved, observed_plan = _resolve_e2_task_binding(
                    **common,
                    requested_output=output_value,
                )
                self.assertEqual(resolved, planned_output)
                self.assertEqual(observed_plan, e2_plan)
                with self.assertRaisesRegex(ValueError, "output disagrees"):
                    _resolve_e2_task_binding(
                        **common,
                        requested_output=Path("runs/e2_frontier/wrong"),
                    )
                with self.assertRaisesRegex(ValueError, "task-sha256 disagrees"):
                    _resolve_e2_task_binding(
                        **{**common, "e2_task_sha256": "d" * 64},
                        requested_output=output_value,
                    )
                with self.assertRaisesRegex(ValueError, "identity disagrees"):
                    _resolve_e2_task_binding(
                        **{**common, "device": "cpu"},
                        requested_output=output_value,
                    )
                e2_plan["status"] = "blocked_on_production_profile_inputs"
                with self.assertRaisesRegex(ValueError, "not ready"):
                    _resolve_e2_task_binding(
                        **common,
                        requested_output=output_value,
                    )

    def test_managed_output_guard_rejects_absolute_escape_and_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            _task_output_path(Path("/tmp/escaped").as_posix())
        with self.assertRaisesRegex(ValueError, "must not equal"):
            _task_output_path(OUTPUT_ROOT.as_posix())
        with self.assertRaisesRegex(ValueError, "must resolve below"):
            _task_output_path("runs/e2_frontier/not_preconfirmation/task")


if __name__ == "__main__":
    unittest.main()
