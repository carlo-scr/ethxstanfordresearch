from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.analysis.preconfirmation import (  # noqa: E402
    CONTROL_ARM,
    DOMAINS,
    LEARNED_ARMS,
    MODEL_FAMILIES,
    PILOT_SEEDS,
    POSITIVE_WEIGHTS,
    PROFILE_ARM,
)
from latent_safety.analysis.preconfirmation_artifacts import (  # noqa: E402
    _validate_profile_teacher_plan,
    canonical_sha256,
    file_sha256,
    write_preconfirmation_task_manifest,
)
from latent_safety.learning.profile_artifacts import ids_sha256  # noqa: E402
from latent_safety.learning.profile_protocol import (  # noqa: E402
    FOLD_COUNT,
    FOLD_SEED,
    PROFILE_PROTOCOL_VERSION,
    REQUIRED_NORMALIZED_P95,
    assign_trajectory_folds,
    evaluate_profile_gate,
    teacher_seed,
)
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.learning.data import trajectory_split_assignment  # noqa: E402
from latent_safety.records import AuditRecord, write_jsonl  # noqa: E402


SCRIPT_PATH = ROOT / "scripts" / "aggregate_e2_preconfirmation.py"
SPEC = importlib.util.spec_from_file_location("aggregate_e2_preconfirmation", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load aggregate_e2_preconfirmation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SELECT_PATH = ROOT / "scripts" / "select_preconfirmation_weights.py"
SELECT_SPEC = importlib.util.spec_from_file_location(
    "select_preconfirmation_weights_for_aggregate_test", SELECT_PATH
)
if SELECT_SPEC is None or SELECT_SPEC.loader is None:
    raise RuntimeError("could not load select_preconfirmation_weights.py")
SELECT_MODULE = importlib.util.module_from_spec(SELECT_SPEC)
SELECT_SPEC.loader.exec_module(SELECT_MODULE)


PROFILES = (
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -0.5, -1.0),
    (-0.5, 1.0, -1.0),
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _signed(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = canonical_sha256(result)
    return result


def _weight_slug(weight: float) -> str:
    return format(weight, ".12g").replace(".", "p")


def _output_dir(
    domain: str, family: str, arm: str, seed: int, weight: float | None
) -> str:
    path = Path("runs/e2_frontier/preconfirmation_288") / domain / family / arm
    if weight is not None:
        path /= f"weight_{_weight_slug(weight)}"
    return (path / f"seed_{seed}").as_posix()


def _specs() -> list[tuple[str, str, str, int, float | None]]:
    values = [
        (domain, family, CONTROL_ARM, seed, None)
        for domain, family, seed in product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS)
    ]
    values.extend(
        (domain, family, arm, seed, float(weight))
        for domain, family, arm, weight, seed in product(
            DOMAINS, MODEL_FAMILIES, LEARNED_ARMS, POSITIVE_WEIGHTS, PILOT_SEEDS
        )
    )
    return values


def _records(domain: str, family: str, seed: int, *, control: bool) -> tuple[AuditRecord, ...]:
    latents = (0.0, 0.01, 1.0, 1.01) if control else (0.0, 1.0, 0.01, 1.01)
    prefix = f"{domain}|{family}|{seed}"
    return tuple(
        AuditRecord(
            sample_id=f"{prefix}|sample-{index}",
            trajectory_id=f"{prefix}|trajectory-{'a' if index < 2 else 'b'}",
            split="validation",
            safety_margin=0.1,
            latent=(latent,),
            action_safety_margins=PROFILES[index],
        )
        for index, latent in enumerate(latents)
    )


def _resolved_values(
    root: Path, task: dict[str, object]
) -> dict[str, object]:
    with (root / str(task["base_config"])).open("rb") as stream:
        values: dict[str, object] = tomllib.load(stream)
    run = values["run"]
    data = values["data"]
    model = values["model"]
    objective = values["objective"]
    evaluation = values["evaluation"]
    assert isinstance(run, dict)
    assert isinstance(data, dict)
    assert isinstance(model, dict)
    assert isinstance(objective, dict)
    assert isinstance(evaluation, dict)
    run.update({"seed": task["seed"], "device": task["device"]})
    data.update(
        {"task": task["domain"], "history_length": task["history_length"]}
    )
    model.update(
        {
            "family": task["model_family"],
            "history_encoder": task["history_encoder"],
            "latent_dim": task["latent_dim"],
        }
    )
    objective.update(
        {
            "safety_arm": task["safety_arm"],
            "safety_weight": task["safety_weight"],
            "kl_weight": task["kl_weight"],
            "fcsrl_head_hidden_dim": task["fcsrl_head_hidden_dim"],
        }
    )
    if task["safety_arm"] == PROFILE_ARM:
        weight_token = _weight_slug(float(task["safety_weight"]))
        values["experiment"] = (
            f"e2_{PROFILE_ARM}_{task['domain']}_{task['model_family']}_"
            f"data_seed_{task['seed']}_weight_{weight_token}"
        )
        values["status"] = "preconfirmation_or_confirmatory"
        run["output_dir"] = (
            "runs/e2_frontier/predicted_profile_arm/"
            f"{task['domain']}/{task['model_family']}/"
            f"data_seed_{task['seed']}/weight_{weight_token}"
        )
        evaluation["emit_audit_records"] = False
    return values


def _ordinary_scope() -> dict[str, object]:
    return {
        "name": "train_validation_only",
        "materialized_splits": ["train", "validation"],
        "evaluated_splits": ["train", "validation"],
        "rollout_splits": ["validation"],
        "oracle_action_profiles_materialized": False,
        "calibration_access": False,
        "final_test_access": False,
    }


def _create_run(root: Path, plan: dict[str, object], task: dict[str, object]) -> None:
    task_id = int(task["task_id"])
    run_dir = root / str(task["output_dir"])
    run_dir.mkdir(parents=True)
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    best_path.write_bytes(f"best-checkpoint-{task_id}".encode())
    last_path.write_bytes(f"last-checkpoint-{task_id}".encode())
    best_sha = file_sha256(best_path)
    last_sha = file_sha256(last_path)
    records = _records(
        str(task["domain"]),
        str(task["model_family"]),
        int(task["seed"]),
        control=task["safety_arm"] == CONTROL_ARM,
    )
    audit_path = run_dir / "audit_validation.jsonl"
    write_jsonl(audit_path, records)

    profile = task["safety_arm"] == PROFILE_ARM
    if profile:
        dataset: dict[str, object] = {
            "schema_version": 1,
            "semantic_arm": PROFILE_ARM,
            "not_materialized": ["calibration", "test"],
            "oracle_action_profiles_materialized": False,
        }
    else:
        dataset = {
            "schema_version": 1,
            "task": task["domain"],
            "access_scope": {
                "materialized_splits": ["train", "validation"],
                "not_materialized_splits": ["calibration", "test"],
                "oracle_action_profiles_materialized": False,
            },
        }
    dataset_sha = canonical_sha256(dataset)
    _write_json(run_dir / "dataset_manifest.json", dataset)
    resolved_values = _resolved_values(root, task)
    resolved_data = resolved_values["data"]
    resolved_evaluation = resolved_values["evaluation"]
    assert isinstance(resolved_data, dict)
    assert isinstance(resolved_evaluation, dict)
    maximum_horizon = max(resolved_evaluation["rollout_horizons"])
    utility = {
        "reconstruction_mse": 1.0 if task["safety_arm"] == CONTROL_ARM else 1.01,
        "maximum_horizon": maximum_horizon,
        "maximum_horizon_rollout_pixel_mse": (
            2.0 if task["safety_arm"] == CONTROL_ARM else 2.02
        ),
    }
    postfit: dict[str, object] = {
        "schema_version": 1,
        "status": "validation_postfit_audit_success",
        "semantic_arm": task["safety_arm"],
        "split_access": {
            "materialized": ["validation"],
            "not_materialized": ["calibration", "test"],
            "checkpoint_selection_complete_before_profile_targets": True,
        },
        "checkpoint_sha256": best_sha,
        "ordinary_validation_metrics": {"reconstruction": utility["reconstruction_mse"]},
        "e2_utility_summary": utility,
        "rollout_metrics": {
            str(maximum_horizon): {
                "pixel_mse": utility["maximum_horizon_rollout_pixel_mse"]
            }
        },
        "trajectory_count": 2,
        "sample_count": len(records),
        "physical_profile_target": {
            "source": "explicit_observed_physical_rollouts",
            "constant_action_grid": resolved_data["actions"],
            "profile_rollout_horizon": resolved_data["action_profile_horizon"],
            "action_safety_profile_helper_called": False,
            "used_for_fitting_or_checkpoint_selection": False,
        },
        "audit_records": {
            "path": "audit_validation.jsonl",
            "sha256": file_sha256(audit_path),
            "record_count": len(records),
        },
    }
    postfit = _signed(postfit, "manifest_sha256")
    postfit_path = run_dir / "validation_postfit_manifest.json"
    _write_json(postfit_path, postfit)
    resolved_sha = canonical_sha256(resolved_values)
    checkpoint: dict[str, object] = {
        "schema_version": 1,
        "status": "success",
        "orchestration_plan_sha256": plan["plan_sha256"],
        "resolved_config_sha256": resolved_sha,
        "dataset_manifest_sha256": dataset_sha,
        "postfit_validation_audit": {
            "path": "validation_postfit_manifest.json",
            "sha256": file_sha256(postfit_path),
            "manifest_sha256": postfit["manifest_sha256"],
            "used_for_fitting_or_checkpoint_selection": False,
        },
        "checkpoints": {
            "best": {"path": "checkpoint_best.pt", "sha256": best_sha},
            "last": {"path": "checkpoint_last.pt", "sha256": last_sha},
        },
    }
    run: dict[str, object] = {
        "schema_version": 1,
        "status": "success",
        "failure": None,
        "orchestration_plan_sha256": plan["plan_sha256"],
        "resolved_config": {"sha256": resolved_sha, "values": resolved_values},
    }
    if profile:
        checkpoint.update(
            {
                "semantic_arm": PROFILE_ARM,
                "teacher_plan_sha256": task["profile_teacher_plan_sha256"],
                "label_manifest_sha256": task["profile_label_manifest_sha256"],
                "coverage_manifest_sha256": task["profile_coverage_manifest_sha256"],
            }
        )
        run.update(
            {
                "semantic_arm": PROFILE_ARM,
                "evidence_eligible": True,
                "execution_code": plan["code"],
                "teacher_plan_sha256": task["profile_teacher_plan_sha256"],
                "label_manifest": {
                    "manifest_sha256": task["profile_label_manifest_sha256"]
                },
                "coverage_gate": {
                    "manifest_sha256": task["profile_coverage_manifest_sha256"]
                },
                "dataset_manifest_sha256": dataset_sha,
                "isolation": {
                    "oracle_action_safety_margins_forbidden": True,
                    "calibration_materialized": False,
                    "final_test_materialized": False,
                },
            }
        )
    else:
        scope = _ordinary_scope()
        config_provenance = {
            "path": str(task["base_config"]),
            "sha256": task["base_config_sha256"],
        }
        checkpoint["code"] = plan["code"]
        checkpoint["config"] = config_provenance
        run.update(
            {
                "code": plan["code"],
                "config": config_provenance,
                "training_arm": task["safety_arm"],
                "data_access_scope": scope,
                "dataset": {"manifest_sha256": dataset_sha},
            }
        )
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    _write_json(checkpoint_path, checkpoint)
    if not profile:
        evaluation = {
            "schema_version": 1,
            "status": "success",
            "code": plan["code"],
            "config": config_provenance,
            "orchestration_plan_sha256": plan["plan_sha256"],
            "resolved_config_sha256": resolved_sha,
            "dataset_manifest_sha256": dataset_sha,
            "checkpoint_sha256": best_sha,
            "checkpoint_manifest": "checkpoint_manifest.json",
            "data_access_scope": _ordinary_scope(),
            "postfit_validation_audit": {
                "path": "validation_postfit_manifest.json",
                "sha256": file_sha256(postfit_path),
                "manifest_sha256": postfit["manifest_sha256"],
            },
            "split_metrics": {"train": {}, "validation": {}},
            "rollout_metrics": {"validation": {}},
            "audit_records": {"validation": "audit_validation.jsonl"},
        }
        _write_json(run_dir / "evaluation_manifest.json", evaluation)
    _write_json(run_dir / "run_manifest.json", run)
    write_preconfirmation_task_manifest(
        repo_root=root,
        plan_path=root / "runs/e2_frontier/preconfirmation_plan.json",
        plan_sha256=str(plan["plan_sha256"]),
        task=task,
    )


def _build_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    intervention_path = root / "configs/e2_frontier/intervention.toml"
    intervention_path.parent.mkdir(parents=True)
    intervention_path.write_bytes(
        (ROOT / "configs/e2_frontier/intervention.toml").read_bytes()
    )
    canonical_bases = {
        "controlled_cart_video": "configs/e1_world_models/torch_pilot.toml",
        "controlled_pendulum_video": (
            "configs/e1_world_models/torch_pendulum_pilot.toml"
        ),
        "controlled_dubins_navigation_pixels": (
            "configs/e1_world_models/torch_dubins_pilot.toml"
        ),
    }
    base_configs: dict[str, dict[str, str]] = {}
    for index, domain in enumerate(DOMAINS):
        relative = f"configs/e1_world_models/base_{index}.toml"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / canonical_bases[domain]).read_bytes())
        base_configs[domain] = {"path": relative, "sha256": file_sha256(path)}

    teacher_data_seeds = (*PILOT_SEEDS, *range(100, 108))
    teacher_splits: dict[
        tuple[str, int, int], tuple[tuple[str, ...], tuple[str, ...]]
    ] = {}
    for domain in DOMAINS:
        base_config = load_config(root / base_configs[domain]["path"])
        for data_seed in teacher_data_seeds:
            split_assignment = trajectory_split_assignment(
                base_config.data, seed=data_seed
            )
            all_training = tuple(
                sorted(
                    trajectory_id
                    for trajectory_id, split in split_assignment.items()
                    if split == "train"
                )
            )
            assignments = assign_trajectory_folds(
                all_training, fold_count=FOLD_COUNT, seed=FOLD_SEED
            )
            for fold in range(FOLD_COUNT):
                held_out = tuple(
                    sorted(
                        trajectory_id
                        for trajectory_id, assigned in assignments.items()
                        if assigned == fold
                    )
                )
                held_out_set = set(held_out)
                fitting = tuple(
                    trajectory_id
                    for trajectory_id in all_training
                    if trajectory_id not in held_out_set
                )
                teacher_splits[(domain, data_seed, fold)] = (fitting, held_out)
    teacher_tasks: list[dict[str, object]] = []
    for task_id, (domain, family, data_seed, fold) in enumerate(
        product(DOMAINS, MODEL_FAMILIES, teacher_data_seeds, range(FOLD_COUNT))
    ):
        fitting_ids, held_out_ids = teacher_splits[(domain, data_seed, fold)]
        teacher_task: dict[str, object] = {
            "task_id": task_id,
            "domain": domain,
            "model_family": family,
            "data_seed": data_seed,
            "fold_index": fold,
            "teacher_seed": teacher_seed(data_seed, fold),
            "base_config": base_configs[domain]["path"],
            "base_config_sha256": base_configs[domain]["sha256"],
            "training_trajectory_count": len(fitting_ids),
            "training_trajectory_ids_sha256": ids_sha256(fitting_ids),
            "held_out_trajectory_count": len(held_out_ids),
            "held_out_trajectory_ids_sha256": ids_sha256(held_out_ids),
            "history_mode": "stack_h4",
            "latent_dim": 8,
            "hidden_dim": 128,
            "transition_hidden_dim": 192,
            "objective": "h_prediction",
            "safety_weight": 0.5,
            "epochs": 30,
        }
        teacher_task["task_sha256"] = canonical_sha256(teacher_task)
        teacher_tasks.append(teacher_task)
    teacher_hashes = {
        (
            str(row["domain"]),
            str(row["model_family"]),
            int(row["data_seed"]),
            int(row["fold_index"]),
        ): str(row["task_sha256"])
        for row in teacher_tasks
    }
    teacher_path = root / "runs/e2_frontier/profile_teacher_plan.json"
    teacher = _signed(
        {
            "schema_version": 2,
            "experiment": "predicted_profile_crossfit_teacher_preflight",
            "status": "pipeline_components_ready_production_cells_and_gates_unexecuted",
            "task_formula": "3 domains x 2 families x 11 data seeds x 5 folds",
            "task_count": len(teacher_tasks),
            "domains": list(DOMAINS),
            "model_families": list(MODEL_FAMILIES),
            "data_seeds": list(teacher_data_seeds),
            "folds": list(range(FOLD_COUNT)),
            "code": {"git_revision": "clean", "git_dirty": False},
            "base_configs": base_configs,
            "tasks": teacher_tasks,
        },
        "plan_sha256",
    )
    _write_json(teacher_path, teacher)
    profile_cells: dict[tuple[str, str, int], dict[str, str]] = {}
    for domain, family, seed in product(DOMAINS, MODEL_FAMILIES, PILOT_SEEDS):
        with (root / base_configs[domain]["path"]).open("rb") as stream:
            base_payload = tomllib.load(stream)
        action_grid = list(base_payload["data"]["actions"])
        profile_horizon = int(base_payload["data"]["action_profile_horizon"])
        margin_scale = float(base_payload["objective"]["margin_scale"])
        label_path = (
            root
            / "runs/e2_frontier/profile_labels"
            / domain
            / family
            / f"data_seed_{seed}"
            / "crossfit_label_manifest.json"
        )
        labels = _signed(
            {
                "schema_version": 1,
                "protocol_version": PROFILE_PROTOCOL_VERSION,
                "status": "complete_crossfit_label_manifest",
                "semantic_arm": PROFILE_ARM,
                "plan_sha256": teacher["plan_sha256"],
                "domain": domain,
                "model_family": family,
                "data_seed": seed,
                "base_config_sha256": base_configs[domain]["sha256"],
                "engineering_smoke": False,
                "evidence_eligible": True,
                "teachers": [
                    {
                        "fold_index": fold,
                        "task_sha256": teacher_hashes[(domain, family, seed, fold)],
                    }
                    for fold in range(5)
                ],
            },
            "manifest_sha256",
        )
        _write_json(label_path, labels)
        coverage_path = (
            root
            / "runs/e2_frontier/profile_coverage"
            / domain
            / family
            / f"data_seed_{seed}"
            / "profile_coverage_gate.json"
        )
        bundle_ids = [
            f"coverage-{domain}-seed-{seed:03d}-bundle-{index:04d}"
            for index in range(200)
        ]
        prediction = tuple(
            margin_scale * 0.08 * (-1.0 if index % 2 else 1.0)
            for index in range(len(action_grid))
        )
        target = tuple(0.0 for _ in action_grid)
        predictions = {bundle_id: prediction for bundle_id in bundle_ids}
        targets = {bundle_id: target for bundle_id in bundle_ids}
        gate = evaluate_profile_gate(
            predictions,
            targets,
            margin_scale=margin_scale,
            required_max=REQUIRED_NORMALIZED_P95,
        )
        coverage = _signed(
            {
                "schema_version": 1,
                "protocol_version": PROFILE_PROTOCOL_VERSION,
                "status": "gate_passed",
                "scientific_gate_passed": True,
                "engineering_smoke": False,
                "evidence_eligible": True,
                "plan_sha256": teacher["plan_sha256"],
                "domain": domain,
                "model_family": family,
                "data_seed": seed,
                "action_grid": action_grid,
                "profile_rollout_horizon": profile_horizon,
                "margin_scale": margin_scale,
                "required_normalized_p95": REQUIRED_NORMALIZED_P95,
                "assembly": {
                    "manifest_sha256": labels["manifest_sha256"],
                    "file_sha256": file_sha256(label_path),
                },
                "bundle_count": 200,
                "bundle_ids_sha256": ids_sha256(bundle_ids),
                "gate": json.loads(json.dumps(gate.to_dict())),
                "bundles": [
                    {
                        "bundle_id": bundle_id,
                        "teacher_fold": index % 5,
                        "teacher_task_sha256": labels["teachers"][index % 5][
                            "task_sha256"
                        ],
                        "prediction": list(predictions[bundle_id]),
                        "observed_target": list(targets[bundle_id]),
                        "normalized_max_error": 0.08,
                    }
                    for index, bundle_id in enumerate(bundle_ids)
                ],
            },
            "manifest_sha256",
        )
        _write_json(coverage_path, coverage)
        profile_cells[(domain, family, seed)] = {
            "profile_teacher_plan": teacher_path.relative_to(root).as_posix(),
            "profile_teacher_plan_file_sha256": file_sha256(teacher_path),
            "profile_teacher_plan_sha256": str(teacher["plan_sha256"]),
            "profile_label_manifest": label_path.relative_to(root).as_posix(),
            "profile_label_manifest_file_sha256": file_sha256(label_path),
            "profile_label_manifest_sha256": str(labels["manifest_sha256"]),
            "profile_coverage_manifest": coverage_path.relative_to(root).as_posix(),
            "profile_coverage_manifest_file_sha256": file_sha256(coverage_path),
            "profile_coverage_manifest_sha256": str(coverage["manifest_sha256"]),
        }

    tasks: list[dict[str, object]] = []
    for task_id, (domain, family, arm, seed, weight) in enumerate(_specs()):
        task: dict[str, object] = {
            "task_id": task_id,
            "domain": domain,
            "model_family": family,
            "history_mode": "stack_h4",
            "history_encoder": "stack",
            "history_length": 4,
            "latent_dim": 8,
            "safety_arm": arm,
            "seed": seed,
            "regularization_weight": weight,
            "safety_weight": 0.0 if weight is None else weight,
            "kl_weight": 0.0 if family == "ae" else 0.0005,
            "fcsrl_head_hidden_dim": 64,
            "device": "cuda",
            "selection_split": "validation_only",
            "calibration_or_final_test_selection_access": False,
            "base_config": base_configs[domain]["path"],
            "base_config_sha256": base_configs[domain]["sha256"],
            "output_dir": _output_dir(domain, family, arm, seed, weight),
        }
        profile_fields = (
            profile_cells[(domain, family, seed)]
            if arm == PROFILE_ARM
            else {
                field: None
                for field in (
                    "profile_teacher_plan",
                    "profile_teacher_plan_file_sha256",
                    "profile_teacher_plan_sha256",
                    "profile_label_manifest",
                    "profile_label_manifest_file_sha256",
                    "profile_label_manifest_sha256",
                    "profile_coverage_manifest",
                    "profile_coverage_manifest_file_sha256",
                    "profile_coverage_manifest_sha256",
                )
            }
        )
        task.update(profile_fields)
        task["task_sha256"] = canonical_sha256(task)
        tasks.append(task)
    plan: dict[str, object] = {
        "schema_version": 1,
        "experiment": "e2_preconfirmation_validation_weight_freeze_288",
        "status": "indexed_validation_only_execution_ready",
        "code": {"git_revision": "clean", "git_dirty": False},
        "intervention_config": {
            "path": intervention_path.relative_to(root).as_posix(),
            "sha256": file_sha256(intervention_path),
        },
        "base_configs": base_configs,
        "scope": {
            "selection_split": "validation_only",
            "calibration_access": False,
            "final_test_access": False,
            "selection_entrypoint": "scripts/select_preconfirmation_weights.py",
        },
        "axes": {
            "domains": list(DOMAINS),
            "model_families": list(MODEL_FAMILIES),
            "history_modes": ["stack_h4"],
            "control_arms": [CONTROL_ARM],
            "learned_arms": list(LEARNED_ARMS),
            "positive_weights": list(POSITIVE_WEIGHTS),
            "seeds": list(PILOT_SEEDS),
        },
        "fixed": {
            "history_encoder": "stack",
            "history_length": 4,
            "latent_dim": 8,
            "fcsrl_head_hidden_dim": 64,
        },
        "counts": {"controls": 18, "learned": 270, "total": 288},
        "profile_handoff": {
            "all_inputs_ready": True,
            "cell_count": 18,
            "weights_per_cell": 5,
        },
        "runner": {
            "path": "scripts/run_e2_preconfirmation_task.py",
            "task_manifest": "preconfirmation_task_manifest.json",
            "evidence_requires_clean_matching_revision": True,
            "evidence_requires_validation_only_trainer": True,
            "validation_postfit_audit_implemented": True,
            "authenticated_task_handoff_implemented": True,
            "execution_enabled": True,
            "generic_execution_enabled": True,
            "profile_execution_enabled": True,
            "execution_blocker": None,
        },
        "task_count": 288,
        "tasks": tasks,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = root / "runs/e2_frontier/preconfirmation_plan.json"
    _write_json(plan_path, plan)
    for task in tasks:
        _create_run(root, plan, task)
    return plan_path, plan


def _resign_postfit_chain(root: Path, plan: dict[str, object], task_id: int) -> None:
    task = plan["tasks"][task_id]
    assert isinstance(task, dict)
    run_dir = root / str(task["output_dir"])
    postfit_path = run_dir / "validation_postfit_manifest.json"
    postfit = json.loads(postfit_path.read_text())
    audit_path = run_dir / "audit_validation.jsonl"
    postfit["audit_records"]["sha256"] = file_sha256(audit_path)
    postfit.pop("manifest_sha256", None)
    postfit["manifest_sha256"] = canonical_sha256(postfit)
    _write_json(postfit_path, postfit)
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["postfit_validation_audit"]["sha256"] = file_sha256(postfit_path)
    checkpoint["postfit_validation_audit"]["manifest_sha256"] = postfit["manifest_sha256"]
    _write_json(checkpoint_path, checkpoint)
    evaluation_path = run_dir / "evaluation_manifest.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["postfit_validation_audit"]["sha256"] = file_sha256(postfit_path)
        evaluation["postfit_validation_audit"]["manifest_sha256"] = postfit[
            "manifest_sha256"
        ]
        _write_json(evaluation_path, evaluation)
    manifest_path = run_dir / "preconfirmation_task_manifest.json"
    manifest_path.unlink()
    write_preconfirmation_task_manifest(
        repo_root=root,
        plan_path=root / "runs/e2_frontier/preconfirmation_plan.json",
        plan_sha256=str(plan["plan_sha256"]),
        task=task,
    )


class AggregateE2PreconfirmationTests(unittest.TestCase):
    def test_selector_fields_report_incomplete_trajectory_coverage(self) -> None:
        fields = MODULE._selector_fields(
            SimpleNamespace(
                trajectory_balanced_p95_required_violation=0.2,
                median_nonself_neighborhood_mass=0.4,
                eligible_center_coverage=0.75,
                eligible_given_viable=0.75,
                empty_trajectory_count=1,
            ),
            task_id=7,
        )

        self.assertEqual(fields["eligible_center_coverage"], 0.75)
        self.assertEqual(fields["empty_trajectory_count"], 1)
        self.assertIs(fields["coverage_complete"], False)

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls._temporary.name) / "fixture"
        cls.plan_path, cls.plan = _build_fixture(cls.fixture_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def setUp(self) -> None:
        clean = {"git_revision": "clean", "git_dirty": False}
        self._code_patches = [
            patch.object(MODULE, "_git_code_state", return_value=clean),
            patch.object(
                SELECT_MODULE.AGGREGATOR_MODULE,
                "_git_code_state",
                return_value=clean,
            ),
        ]
        for code_patch in self._code_patches:
            code_patch.start()

    def tearDown(self) -> None:
        for code_patch in reversed(self._code_patches):
            code_patch.stop()

    def _copy(self) -> tuple[Path, Path, dict[str, object]]:
        root = Path(tempfile.mkdtemp()) / "case"
        shutil.copytree(self.fixture_root, root)
        plan_path = root / self.plan_path.relative_to(self.fixture_root)
        return root, plan_path, json.loads(plan_path.read_text())

    def test_complete_288_fixture_and_selector_preserve_radius_freezes(self) -> None:
        with patch.object(MODULE, "ROOT", self.fixture_root):
            aggregate = MODULE.aggregate(self.plan_path)
        self.assertEqual(aggregate["observation_count"], 288)
        self.assertEqual(aggregate["row_provenance_count"], 288)
        self.assertEqual(len(aggregate["observations"]), 288)
        self.assertEqual(len(aggregate["row_provenance"]), 288)
        self.assertEqual(len(aggregate["implementation"]), 9)
        self.assertTrue(
            all(len(entry["sha256"]) == 64 for entry in aggregate["implementation"])
        )
        self.assertTrue(
            all(
                len(row["radius_freeze_sha256"]) == 64
                for row in aggregate["row_provenance"]
            )
        )
        selected_input = self.fixture_root / "selector_input.json"
        _write_json(selected_input, aggregate)
        try:
            with patch.object(
                SELECT_MODULE, "AGGREGATION_ROOT", self.fixture_root
            ), patch.object(
                SELECT_MODULE.AGGREGATOR_MODULE, "ROOT", self.fixture_root
            ):
                selected = SELECT_MODULE.select(selected_input)
        finally:
            selected_input.unlink()
        freezes = selected["result"]["selected_radius_freezes"]
        self.assertEqual(selected["result"]["selected_radius_freeze_count"], 18)
        self.assertEqual(len(freezes), 18)
        self.assertTrue(all(len(row["seed_radius_freezes"]) == 3 for row in freezes))

    def test_missing_and_tampered_task_manifests_fail_closed(self) -> None:
        root, plan_path, plan = self._copy()
        task = plan["tasks"][0]
        path = root / task["output_dir"] / "preconfirmation_task_manifest.json"
        path.unlink()
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "task 0 artifact authentication failed"
        ):
            MODULE.aggregate(plan_path)

    def test_selector_rejects_a_resigned_fabricated_aggregate(self) -> None:
        with patch.object(MODULE, "ROOT", self.fixture_root):
            aggregate = MODULE.aggregate(self.plan_path)
        aggregate["scope"]["paired_none_control"] = "fabricated but re-signed"
        aggregate.pop("aggregate_sha256")
        aggregate["aggregate_sha256"] = canonical_sha256(aggregate)
        input_path = self.fixture_root / "fabricated_selector_input.json"
        _write_json(input_path, aggregate)
        try:
            with patch.object(
                SELECT_MODULE, "AGGREGATION_ROOT", self.fixture_root
            ), patch.object(
                SELECT_MODULE.AGGREGATOR_MODULE, "ROOT", self.fixture_root
            ), self.assertRaisesRegex(ValueError, "does not exactly match reconstruction"):
                SELECT_MODULE.select(input_path)
        finally:
            input_path.unlink()

        root, plan_path, plan = self._copy()
        task = plan["tasks"][0]
        path = root / task["output_dir"] / "preconfirmation_task_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["utility_summary"]["reconstruction"] = 99.0
        _write_json(path, manifest)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "self-hash mismatch"
        ):
            MODULE.aggregate(plan_path)

    def test_plan_hash_and_duplicate_cell_drift_are_rejected(self) -> None:
        root, plan_path, plan = self._copy()
        plan["status"] = "tampered"
        _write_json(plan_path, plan)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "plan self-hash mismatch"
        ):
            MODULE.aggregate(plan_path)

        root, plan_path, plan = self._copy()
        tasks = plan["tasks"]
        tasks[1]["domain"] = tasks[0]["domain"]
        tasks[1]["model_family"] = tasks[0]["model_family"]
        tasks[1]["seed"] = tasks[0]["seed"]
        tasks[1].pop("task_sha256")
        tasks[1]["task_sha256"] = canonical_sha256(tasks[1])
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(plan)
        _write_json(plan_path, plan)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "identity mismatch"
        ):
            MODULE.aggregate(plan_path)

    def test_resigned_intervention_control_radius_drift_is_rejected(self) -> None:
        root, plan_path, plan = self._copy()
        intervention_path = root / str(plan["intervention_config"]["path"])
        original = intervention_path.read_text(encoding="utf-8")
        drifted = original.replace(
            "control_reference_relative_radius = 0.05",
            "control_reference_relative_radius = 0.10",
        )
        self.assertNotEqual(drifted, original)
        intervention_path.write_text(drifted, encoding="utf-8")
        plan["intervention_config"]["sha256"] = file_sha256(intervention_path)
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(plan)
        _write_json(plan_path, plan)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "intervention protocol contract drifted"
        ):
            MODULE.aggregate(plan_path)

    def test_resolved_action_grid_and_utility_substitution_are_rejected(self) -> None:
        root, plan_path, plan = self._copy()
        task = plan["tasks"][0]
        run_path = root / task["output_dir"] / "run_manifest.json"
        run = json.loads(run_path.read_text())
        run["resolved_config"]["values"]["data"]["actions"] = [-1.0, 1.0]
        run["resolved_config"]["sha256"] = canonical_sha256(
            run["resolved_config"]["values"]
        )
        _write_json(run_path, run)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "resolved config disagrees with planned task"
        ):
            MODULE.aggregate(plan_path)

        root, _plan_path, plan = self._copy()
        task = plan["tasks"][0]
        postfit_path = root / task["output_dir"] / "validation_postfit_manifest.json"
        postfit = json.loads(postfit_path.read_text())
        postfit["e2_utility_summary"]["reconstruction_mse"] = 99.0
        _write_json(postfit_path, postfit)
        with self.assertRaisesRegex(
            ValueError, "utility summary disagrees with authenticated validation metrics"
        ):
            _resign_postfit_chain(root, plan, 0)

    def test_profile_teacher_numeric_contract_drift_is_rejected(self) -> None:
        teacher_path = self.fixture_root / "runs/e2_frontier/profile_teacher_plan.json"
        canonical = json.loads(teacher_path.read_text())
        teacher = copy.deepcopy(canonical)
        teacher["tasks"][0]["epochs"] = 31
        teacher["tasks"][0].pop("task_sha256")
        teacher["tasks"][0]["task_sha256"] = canonical_sha256(teacher["tasks"][0])
        teacher.pop("plan_sha256")
        teacher["plan_sha256"] = canonical_sha256(teacher)
        with self.assertRaisesRegex(ValueError, "numeric/identity contract drifted"):
            _validate_profile_teacher_plan(self.fixture_root, teacher)

        membership = copy.deepcopy(canonical)
        membership["tasks"][0]["held_out_trajectory_ids_sha256"] = "f" * 64
        membership["tasks"][0].pop("task_sha256")
        membership["tasks"][0]["task_sha256"] = canonical_sha256(
            membership["tasks"][0]
        )
        membership.pop("plan_sha256")
        membership["plan_sha256"] = canonical_sha256(membership)
        with self.assertRaisesRegex(ValueError, "fold membership disagrees"):
            _validate_profile_teacher_plan(self.fixture_root, membership)

        alternate_fold = copy.deepcopy(canonical)
        for field in (
            "training_trajectory_count",
            "training_trajectory_ids_sha256",
            "held_out_trajectory_count",
            "held_out_trajectory_ids_sha256",
        ):
            alternate_fold["tasks"][0][field] = canonical["tasks"][1][field]
        alternate_fold["tasks"][0].pop("task_sha256")
        alternate_fold["tasks"][0]["task_sha256"] = canonical_sha256(
            alternate_fold["tasks"][0]
        )
        alternate_fold.pop("plan_sha256")
        alternate_fold["plan_sha256"] = canonical_sha256(alternate_fold)
        with self.assertRaisesRegex(ValueError, "fold membership disagrees"):
            _validate_profile_teacher_plan(self.fixture_root, alternate_fold)

    def test_checkpoint_hash_mismatch_is_rejected(self) -> None:
        root, plan_path, plan = self._copy()
        task = plan["tasks"][0]
        checkpoint = root / task["output_dir"] / "checkpoint_best.pt"
        checkpoint.write_bytes(b"tampered-checkpoint")
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "checkpoint best file SHA mismatch"
        ):
            MODULE.aggregate(plan_path)

    def test_resigned_dirty_plan_and_kl_drift_are_rejected(self) -> None:
        root, plan_path, plan = self._copy()
        plan["code"]["git_dirty"] = True
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(plan)
        _write_json(plan_path, plan)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "clean committed revision"
        ):
            MODULE.aggregate(plan_path)

        root, plan_path, plan = self._copy()
        plan["status"] = "blocked_on_production_profile_inputs"
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(plan)
        _write_json(plan_path, plan)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "fully profile-ready"
        ):
            MODULE.aggregate(plan_path)

        root, plan_path, plan = self._copy()
        task = plan["tasks"][3]
        self.assertEqual(task["model_family"], "beta_vae")
        task["kl_weight"] = 0.001
        task.pop("task_sha256")
        task["task_sha256"] = canonical_sha256(task)
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(plan)
        _write_json(plan_path, plan)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "identity mismatch"
        ):
            MODULE.aggregate(plan_path)

    def test_split_isolation_is_rejected_even_with_resigned_hash_chain(self) -> None:
        root, plan_path, plan = self._copy()
        task = plan["tasks"][0]
        audit_path = root / task["output_dir"] / "audit_validation.jsonl"
        rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        rows[0]["split"] = "test"
        rows[1]["split"] = "test"
        audit_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "validation-only"):
            _resign_postfit_chain(root, plan, 0)

    def test_paired_physical_record_mismatch_is_rejected(self) -> None:
        root, plan_path, plan = self._copy()
        task = plan["tasks"][18]
        audit_path = root / task["output_dir"] / "audit_validation.jsonl"
        rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        rows[0]["safety_margin"] = 0.2
        audit_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        _resign_postfit_chain(root, plan, 18)
        with patch.object(MODULE, "ROOT", root), self.assertRaisesRegex(
            ValueError, "not physically paired"
        ):
            MODULE.aggregate(plan_path)

    def test_cli_refuses_overwrite(self) -> None:
        output = self.fixture_root / "aggregate_output.json"
        try:
            with patch.object(MODULE, "ROOT", self.fixture_root):
                self.assertEqual(
                    MODULE.main(["--plan", str(self.plan_path), "--output", str(output)]),
                    0,
                )
                with self.assertRaises(SystemExit):
                    MODULE.main(
                        ["--plan", str(self.plan_path), "--output", str(output)]
                    )
        finally:
            if output.exists():
                output.unlink()


if __name__ == "__main__":
    unittest.main()
