#!/usr/bin/env python3
"""Build the exact 330-task cross-fitted predicted-profile teacher preflight plan."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.learning import (  # noqa: E402
    assign_trajectory_folds,
    load_config,
    teacher_seed,
    trajectory_split_assignment,
)
from latent_safety.manifest import config_sha256, write_json_atomic  # noqa: E402

DOMAINS = {
    "controlled_cart_video": "configs/e1_world_models/torch_pilot.toml",
    "controlled_pendulum_video": (
        "configs/e1_world_models/torch_pendulum_pilot.toml"
    ),
    "controlled_dubins_navigation_pixels": (
        "configs/e1_world_models/torch_dubins_pilot.toml"
    ),
}
MODEL_FAMILIES = ("ae", "beta_vae")
DATA_SEEDS = (0, 1, 2, 100, 101, 102, 103, 104, 105, 106, 107)
FOLDS = tuple(range(5))
EXPECTED_TASKS = 330


def _ids_sha256(ids: list[str]) -> str:
    canonical = json.dumps(sorted(ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_code_state() -> dict[str, str | bool | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_revision": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def task_sha256(task: dict[str, Any]) -> str:
    """Hash a task without allowing its digest field to self-attest."""

    unsigned = dict(task)
    unsigned.pop("task_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_plan() -> dict[str, Any]:
    domain_data: dict[tuple[str, int], dict[str, Any]] = {}
    base_configs: dict[str, dict[str, str]] = {}
    for domain, relative in DOMAINS.items():
        config_path = ROOT / relative
        config = load_config(config_path)
        if config.data.task != domain:
            raise ValueError(f"{relative} declares {config.data.task!r}, expected {domain!r}")
        if config.data.history_length != 4:
            raise ValueError(f"{relative} must freeze four-frame teacher histories")
        base_configs[domain] = {
            "path": relative,
            "sha256": config_sha256(config_path),
        }
        for data_seed in DATA_SEEDS:
            split_assignment = trajectory_split_assignment(
                config.data, seed=data_seed
            )
            training_ids = sorted(
                trajectory_id
                for trajectory_id, split in split_assignment.items()
                if split == "train"
            )
            assignments = assign_trajectory_folds(training_ids)
            folds = {
                fold: sorted(
                    trajectory_id
                    for trajectory_id, assigned in assignments.items()
                    if assigned == fold
                )
                for fold in FOLDS
            }
            if any(not ids for ids in folds.values()):
                raise ValueError(f"empty teacher fold for {domain}/seed-{data_seed}")
            domain_data[(domain, data_seed)] = {
                "training_trajectory_count": len(training_ids),
                "training_trajectory_ids_sha256": _ids_sha256(training_ids),
                "folds": folds,
            }

    tasks = []
    for task_id, (domain, family, data_seed, fold) in enumerate(
        itertools.product(DOMAINS, MODEL_FAMILIES, DATA_SEEDS, FOLDS)
    ):
        data = domain_data[(domain, data_seed)]
        held_out_ids = data["folds"][fold]
        training_ids = [
            trajectory_id
            for candidate_fold, ids in data["folds"].items()
            if candidate_fold != fold
            for trajectory_id in ids
        ]
        task: dict[str, Any] = {
            "task_id": task_id,
            "domain": domain,
            "model_family": family,
            "data_seed": data_seed,
            "fold_index": fold,
            "teacher_seed": teacher_seed(data_seed, fold),
            "base_config": base_configs[domain]["path"],
            "base_config_sha256": base_configs[domain]["sha256"],
            "training_trajectory_count": len(training_ids),
            "training_trajectory_ids_sha256": _ids_sha256(training_ids),
            "held_out_trajectory_count": len(held_out_ids),
            "held_out_trajectory_ids_sha256": _ids_sha256(held_out_ids),
            "history_mode": "stack_h4",
            "latent_dim": 8,
            "hidden_dim": 128,
            "transition_hidden_dim": 192,
            "objective": "h_prediction",
            "safety_weight": 0.5,
            "epochs": 30,
            "permitted_training_signals": [
                "pixel_history",
                "next_pixel_history",
                "behavior_action",
                "observed_safety_margin",
            ],
            "forbidden_training_signals": [
                "hidden_physical_state",
                "known_dynamics",
                "oracle_counterfactual_action_profile",
                "calibration_split",
                "final_test_split",
            ],
            "output_dir": (
                "runs/e2_frontier/profile_teachers/"
                f"{domain}/{family}/data_seed_{data_seed}/fold_{fold}"
            ),
            "runner_status": (
                "teacher runner, five-shard assembly, strict downstream ingestion, and "
                "observed-rollout coverage gate implemented; production cell unexecuted"
            ),
        }
        task["task_sha256"] = task_sha256(task)
        tasks.append(task)
    if len(tasks) != EXPECTED_TASKS:
        raise AssertionError(f"teacher plan has {len(tasks)} tasks, expected {EXPECTED_TASKS}")
    if len({task["output_dir"] for task in tasks}) != EXPECTED_TASKS:
        raise AssertionError("teacher plan contains duplicate output directories")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "experiment": "predicted_profile_crossfit_teacher_preflight",
        "status": "pipeline_components_ready_production_cells_and_gates_unexecuted",
        "task_formula": "3 domains x 2 families x 11 data seeds x 5 folds",
        "task_count": len(tasks),
        "domains": list(DOMAINS),
        "model_families": list(MODEL_FAMILIES),
        "data_seeds": list(DATA_SEEDS),
        "folds": list(FOLDS),
        "code": _git_code_state(),
        "runner": {
            "path": "scripts/run_profile_teacher.py",
            "assembly_path": "scripts/assemble_profile_labels.py",
            "coverage_path": "scripts/run_profile_coverage_gate.py",
            "downstream_path": "scripts/run_predicted_profile_arm.py",
            "handoff_manifest": "teacher_handoff_manifest.json",
            "held_out_label_shard": "heldout_profile_predictions.jsonl",
        },
        "base_configs": base_configs,
        "tasks": tasks,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    plan = build_plan()
    write_json_atomic(output.resolve(), plan)
    print(
        f"wrote {plan['task_count']} component-ready teacher tasks "
        "(production cells and gates unexecuted) "
        f"to {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
