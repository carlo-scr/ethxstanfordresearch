from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plan_profile_teachers import build_plan  # noqa: E402
from latent_safety.learning.config import canonical_config_json, load_config  # noqa: E402
from latent_safety.learning.profile_artifacts import (  # noqa: E402
    PREDICTED_PROFILE_ARM,
    PredictedProfileDataset,
    ProfileArtifactError,
    assemble_crossfit_label_shards,
    file_sha256,
    ids_sha256,
    load_crossfit_label_index,
)
from latent_safety.learning.profile_protocol import build_crossfit_manifest  # noqa: E402
from latent_safety.learning.profile_teacher import (  # noqa: E402
    build_teacher_split_spec,
    resolve_teacher_config,
)
from latent_safety.manifest import write_json_atomic  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fake_teacher_shard(
    directory: Path,
    task: dict[str, object],
    plan: dict[str, object],
    base: object,
    *,
    omit_last_label: bool = False,
) -> None:
    directory.mkdir()
    resolved = resolve_teacher_config(
        base,
        task,
        device="cpu",
        engineering_smoke=True,
    )
    split = build_teacher_split_spec(
        resolved,
        data_seed=int(task["data_seed"]),
        fold_index=int(task["fold_index"]),
    )
    crossfit, crossfit_sha = build_crossfit_manifest(
        task=resolved.data.task,
        model_family=resolved.model.family,
        data_seed=int(task["data_seed"]),
        training_trajectory_ids=split.all_training_ids,
        action_grid=resolved.data.actions,
        horizon=resolved.data.action_profile_horizon,
        margin_scale=resolved.objective.margin_scale,
        inherited_config_sha256=str(task["base_config_sha256"]),
    )
    crossfit_path = directory / "crossfit_protocol_manifest.json"
    write_json_atomic(crossfit_path, crossfit)
    dataset = {
        "schema_version": 1,
        "protocol_version": "predicted_profile_v1",
        "generator": "controlled_observed_margin_teacher_v1",
        "data_seed": task["data_seed"],
        "task": task["domain"],
        "fold_index": task["fold_index"],
        "split_access": {
            "materialized": ["train", "validation"],
            "not_materialized": ["calibration", "test"],
        },
        "signals": {
            "permitted_sample_fields": [
                "sample_id",
                "trajectory_id",
                "timestep",
                "history",
                "next_history",
                "action",
                "safety_margin",
            ],
            "oracle_action_profiles_materialized": False,
            "hidden_state_exposed_to_model": False,
        },
        "trajectory_sets": {
            "all_training": {
                "count": len(split.all_training_ids),
                "sha256": split.all_training_ids_sha256,
            },
            "fitting": {
                "count": len(split.fitting_ids),
                "sha256": split.fitting_ids_sha256,
            },
            "held_out": {
                "count": len(split.held_out_ids),
                "sha256": split.held_out_ids_sha256,
            },
            "ordinary_validation": {
                "count": len(split.validation_ids),
                "sha256": split.validation_ids_sha256,
            },
        },
        "sample_counts": {
            "train": len(split.fitting_ids) * resolved.data.horizon,
            "held_out": len(split.held_out_ids) * resolved.data.horizon,
            "validation": len(split.validation_ids) * resolved.data.horizon,
        },
        "crossfit_protocol_manifest_sha256": crossfit_sha,
        "engineering_smoke": True,
        "production_plan_split_hashes_enforced": False,
    }
    dataset_path = directory / "dataset_manifest.json"
    write_json_atomic(dataset_path, dataset)
    (directory / "history.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / "checkpoint_best.pt").write_bytes(
        f"best-{task['task_id']}".encode()
    )
    (directory / "checkpoint_last.pt").write_bytes(
        f"last-{task['task_id']}".encode()
    )
    resolved_sha = hashlib.sha256(
        canonical_config_json(resolved).encode("utf-8")
    ).hexdigest()
    checkpoint = {
        "schema_version": 1,
        "protocol_version": "predicted_profile_v1",
        "status": "success",
        "plan_sha256": plan["plan_sha256"],
        "task_sha256": task["task_sha256"],
        "resolved_config_sha256": resolved_sha,
        "dataset_manifest_sha256": "d" * 64,
        "crossfit_protocol_manifest_sha256": crossfit_sha,
        "architecture": {},
        "selection": {
            "metric": "ordinary_validation.world_model_utility",
            "mode": "min",
            "tie_break": "earliest_epoch",
            "selected_epoch": 1,
            "selected_value": 1.0,
            "held_out_fold_used": False,
            "calibration_or_test_used": False,
        },
        "checkpoints": {
            "best": {
                "path": "checkpoint_best.pt",
                "sha256": file_sha256(directory / "checkpoint_best.pt"),
            },
            "last": {
                "path": "checkpoint_last.pt",
                "sha256": file_sha256(directory / "checkpoint_last.pt"),
            },
        },
        "state_hashes": {
            "selected_model_state": "a" * 64,
            "selected_optimizer_state": "b" * 64,
        },
    }
    checkpoint_path = directory / "checkpoint_manifest.json"
    write_json_atomic(checkpoint_path, checkpoint)
    rows = [
        {
            "sample_id": f"{trajectory_id}:{timestep:04d}",
            "trajectory_id": trajectory_id,
            "timestep": timestep,
            "predicted_action_profile": [
                0.01 * int(task["fold_index"]) + 0.001 * action_index
                for action_index in range(len(resolved.data.actions))
            ],
        }
        for trajectory_id in split.held_out_ids
        for timestep in range(resolved.data.horizon)
    ]
    if omit_last_label:
        rows.pop()
    labels_path = directory / "heldout_profile_predictions.jsonl"
    _write_jsonl(labels_path, rows)
    label_manifest = {
        "schema_version": 1,
        "protocol_version": "predicted_profile_v1",
        "status": "unvalidated_crossfit_label_shard",
        "plan_sha256": plan["plan_sha256"],
        "task_sha256": task["task_sha256"],
        "checkpoint_manifest": {
            "path": checkpoint_path.name,
            "sha256": file_sha256(checkpoint_path),
        },
        "label_shard": {
            "path": labels_path.name,
            "sha256": file_sha256(labels_path),
            "record_count": len(rows),
            "sample_ids_sha256": ids_sha256(
                [str(row["sample_id"]) for row in rows]
            ),
        },
        "held_out_trajectory_ids_sha256": split.held_out_ids_sha256,
        "action_grid": list(resolved.data.actions),
        "horizon": resolved.data.action_profile_horizon,
        "inference": {
            "history_mode": "stack_h4",
            "latent": "posterior_mean",
            "transition": "deterministic",
            "profile": "minimum_predicted_margin_over_t_0_through_H",
            "ensemble": False,
        },
        "eligibility": {
            "five_fold_assembly_complete": False,
            "coverage_gate_complete": False,
            "paper_evidence": False,
        },
    }
    label_manifest_path = directory / "label_shard_manifest.json"
    write_json_atomic(label_manifest_path, label_manifest)
    artifacts = {
        "checkpoint_manifest": checkpoint_path,
        "crossfit_protocol_manifest": crossfit_path,
        "dataset_manifest": dataset_path,
        "history": directory / "history.jsonl",
        "label_shard_manifest": label_manifest_path,
    }
    handoff = {
        "schema_version": 1,
        "protocol_version": "predicted_profile_v1",
        "status": "single_teacher_success",
        "evidence_eligible": False,
        "engineering_smoke": True,
        "plan_sha256": plan["plan_sha256"],
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "artifacts": {
            name: {"path": path.name, "sha256": file_sha256(path)}
            for name, path in artifacts.items()
        },
        "remaining_blockers": [],
    }
    handoff_path = directory / "teacher_handoff_manifest.json"
    write_json_atomic(handoff_path, handoff)
    run_manifest = {
        "status": "success",
        "engineering_smoke": True,
        "plan": {
            "sha256": plan["plan_sha256"],
            "task_sha256": task["task_sha256"],
        },
        "resolved_config": {
            "sha256": resolved_sha,
            "values": resolved.to_dict(),
        },
        "handoff_manifest": {
            "path": handoff_path.name,
            "sha256": file_sha256(handoff_path),
        },
    }
    write_json_atomic(directory / "run_manifest.json", run_manifest)


class _FakeHistory:
    def new_tensor(self, values: object) -> tuple[str, tuple[float, ...]]:
        return "tensor", tuple(float(value) for value in values)  # type: ignore[arg-type]


class _ObservedDataset:
    def __init__(self, sample_ids: tuple[str, ...], *, forbidden: bool = False) -> None:
        self.sample_ids = sample_ids
        self.forbidden = forbidden

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = {
            "sample_id": self.sample_ids[index],
            "history": _FakeHistory(),
        }
        if self.forbidden:
            sample["action_safety_margins"] = (999.0,)
        return sample


class ProfileLabelAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan()
        cls.tasks = cls.plan["tasks"][:5]
        cls.base = load_config(ROOT / cls.tasks[0]["base_config"])

    def _cell(self, root: Path, *, omit_fold: int | None = None) -> list[Path]:
        shards = []
        for task in self.tasks:
            directory = root / f"fold_{task['fold_index']}"
            _fake_teacher_shard(
                directory,
                task,
                self.plan,
                self.base,
                omit_last_label=task["fold_index"] == omit_fold,
            )
            shards.append(directory)
        return shards

    def _expected(self) -> tuple[object, list[str]]:
        config = resolve_teacher_config(
            self.base,
            self.tasks[0],
            device="cpu",
            engineering_smoke=True,
        )
        split = build_teacher_split_spec(config, data_seed=0, fold_index=0)
        expected = [
            f"{trajectory_id}:{timestep:04d}"
            for trajectory_id in split.all_training_ids
            for timestep in range(config.data.horizon)
        ]
        return config, expected

    def test_five_shards_assemble_and_ingest_exact_training_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = self._cell(root)
            result = assemble_crossfit_label_shards(
                self.tasks,
                shards,
                plan_sha256=str(self.plan["plan_sha256"]),
                base_config=self.base,
                output_dir=root / "assembled",
                producing_command=("python", "scripts/assemble_profile_labels.py"),
                engineering_smoke=True,
                execution_code={"git_revision": None, "git_dirty": True},
                execution_evidence_eligible=False,
            )
            self.assertEqual(result["label_count"], 104)
            self.assertFalse(result["evidence_eligible"])
            config, expected = self._expected()
            index = load_crossfit_label_index(
                Path(result["manifest_path"]),
                expected_domain=config.data.task,
                expected_model_family=config.model.family,
                expected_data_seed=0,
                expected_plan_sha256=str(self.plan["plan_sha256"]),
                expected_base_config_sha256=str(self.tasks[0]["base_config_sha256"]),
                expected_action_grid=config.data.actions,
                expected_profile_horizon=config.data.action_profile_horizon,
                expected_margin_scale=config.objective.margin_scale,
                expected_sample_ids=expected,
                allow_engineering_smoke=True,
            )
            self.assertEqual(index.semantic_arm, PREDICTED_PROFILE_ARM)
            self.assertEqual(len(index.profiles), 104)
            self.assertEqual(set(index.teacher_folds.values()), set(range(5)))

            dataset = PredictedProfileDataset(_ObservedDataset(tuple(expected)), index)
            sample = dataset[0]
            self.assertIn("predicted_action_profile_target", sample)
            self.assertNotIn("action_safety_margins", sample)
            self.assertEqual(sample["profile_label_source"], PREDICTED_PROFILE_ARM)

    def test_missing_label_and_duplicate_shard_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = self._cell(root, omit_fold=2)
            with self.assertRaisesRegex(ProfileArtifactError, "sample set mismatch"):
                assemble_crossfit_label_shards(
                    self.tasks,
                    missing,
                    plan_sha256=str(self.plan["plan_sha256"]),
                    base_config=self.base,
                    output_dir=root / "missing-output",
                    producing_command=("test",),
                    engineering_smoke=True,
                    execution_code={},
                    execution_evidence_eligible=False,
                )
            self.assertFalse((root / "missing-output").exists())
            duplicate = list(missing)
            duplicate[-1] = duplicate[0]
            with self.assertRaisesRegex(ProfileArtifactError, "directories must be unique"):
                assemble_crossfit_label_shards(
                    self.tasks,
                    duplicate,
                    plan_sha256=str(self.plan["plan_sha256"]),
                    base_config=self.base,
                    output_dir=root / "duplicate-output",
                    producing_command=("test",),
                    engineering_smoke=True,
                    execution_code={},
                    execution_evidence_eligible=False,
                )

    def test_tampered_shard_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = self._cell(root)
            with (shards[3] / "heldout_profile_predictions.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ProfileArtifactError, "SHA-256 mismatch"):
                assemble_crossfit_label_shards(
                    self.tasks,
                    shards,
                    plan_sha256=str(self.plan["plan_sha256"]),
                    base_config=self.base,
                    output_dir=root / "tampered-output",
                    producing_command=("test",),
                    engineering_smoke=True,
                    execution_code={},
                    execution_evidence_eligible=False,
                )

    def test_ingestion_rejects_smoke_by_default_and_wrong_sample_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = self._cell(root)
            result = assemble_crossfit_label_shards(
                self.tasks,
                shards,
                plan_sha256=str(self.plan["plan_sha256"]),
                base_config=self.base,
                output_dir=root / "assembled",
                producing_command=("test",),
                engineering_smoke=True,
                execution_code={},
                execution_evidence_eligible=False,
            )
            config, expected = self._expected()
            kwargs = {
                "expected_domain": config.data.task,
                "expected_model_family": config.model.family,
                "expected_data_seed": 0,
                "expected_plan_sha256": str(self.plan["plan_sha256"]),
                "expected_base_config_sha256": str(
                    self.tasks[0]["base_config_sha256"]
                ),
                "expected_action_grid": config.data.actions,
                "expected_profile_horizon": config.data.action_profile_horizon,
                "expected_margin_scale": config.objective.margin_scale,
                "expected_sample_ids": expected,
            }
            with self.assertRaisesRegex(ProfileArtifactError, "forbidden"):
                load_crossfit_label_index(Path(result["manifest_path"]), **kwargs)
            kwargs["allow_engineering_smoke"] = True
            kwargs["expected_sample_ids"] = expected[:-1]
            with self.assertRaisesRegex(ProfileArtifactError, "training samples"):
                load_crossfit_label_index(Path(result["manifest_path"]), **kwargs)

    def test_adapter_rejects_privileged_oracle_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = self._cell(root)
            result = assemble_crossfit_label_shards(
                self.tasks,
                shards,
                plan_sha256=str(self.plan["plan_sha256"]),
                base_config=self.base,
                output_dir=root / "assembled",
                producing_command=("test",),
                engineering_smoke=True,
                execution_code={},
                execution_evidence_eligible=False,
            )
            config, expected = self._expected()
            index = load_crossfit_label_index(
                Path(result["manifest_path"]),
                expected_domain=config.data.task,
                expected_model_family=config.model.family,
                expected_data_seed=0,
                expected_plan_sha256=str(self.plan["plan_sha256"]),
                expected_base_config_sha256=str(self.tasks[0]["base_config_sha256"]),
                expected_action_grid=config.data.actions,
                expected_profile_horizon=config.data.action_profile_horizon,
                expected_margin_scale=config.objective.margin_scale,
                expected_sample_ids=expected,
                allow_engineering_smoke=True,
            )
            dataset = PredictedProfileDataset(
                _ObservedDataset(tuple(expected), forbidden=True), index
            )
            with self.assertRaisesRegex(ProfileArtifactError, "forbidden supervision"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
