#!/usr/bin/env python3
"""Authenticate and assemble one five-teacher cross-fitted profile-label cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_profile_teacher import (  # noqa: E402
    _current_code_state,
    _repository_path,
    _smoke_output_path,
    _task_output_path,
    _verify_plan,
)
from latent_safety.learning.config import load_config  # noqa: E402
from latent_safety.learning.profile_artifacts import (  # noqa: E402
    assemble_crossfit_label_shards,
)


def _production_output_path(value: Path) -> Path:
    if value.is_absolute():
        raise ValueError("production assembly output must be repository-relative")
    resolved = (ROOT / value).resolve()
    managed = (ROOT / "runs" / "e2_frontier" / "profile_labels").resolve()
    try:
        resolved.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            "production assembly output must resolve below runs/e2_frontier/profile_labels"
        ) from error
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-family", choices=("ae", "beta_vae"), required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        help="repeat exactly five times for an engineering-smoke assembly",
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--allow-unversioned", action="store_true")
    args = parser.parse_args(argv)

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    try:
        with plan_path.open(encoding="utf-8") as stream:
            plan = _verify_plan(json.load(stream))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    tasks = sorted(
        (
            task
            for task in plan["tasks"]
            if task["domain"] == args.domain
            and task["model_family"] == args.model_family
            and task["data_seed"] == args.data_seed
        ),
        key=lambda task: int(task["fold_index"]),
    )
    if len(tasks) != 5 or [task["fold_index"] for task in tasks] != list(range(5)):
        parser.error("selected plan cell must contain exactly folds 0 through 4")
    config_entry = plan["base_configs"].get(args.domain)
    if not isinstance(config_entry, dict):
        parser.error(f"plan has no base config for domain {args.domain!r}")
    try:
        base_path = _repository_path(
            config_entry["path"], label=f"base_configs.{args.domain}.path"
        )
        base_config = load_config(base_path)
        if args.engineering_smoke:
            if args.shard_dir is None or len(args.shard_dir) != 5:
                raise ValueError("engineering-smoke assembly requires exactly five --shard-dir")
            if args.output is None:
                raise ValueError("engineering-smoke assembly requires --output")
            shard_dirs = [path.resolve() for path in args.shard_dir]
            output_dir = _smoke_output_path(args.output)
        else:
            if args.shard_dir is not None:
                raise ValueError("--shard-dir is valid only with --engineering-smoke")
            shard_dirs = [_task_output_path(task["output_dir"]) for task in tasks]
            default_output = Path(
                "runs/e2_frontier/profile_labels/"
                f"{args.domain}/{args.model_family}/data_seed_{args.data_seed}"
            )
            output_dir = _production_output_path(args.output or default_output)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    current_code = _current_code_state()
    planned_code = plan["code"]
    version_matches = (
        planned_code.get("git_revision") is not None
        and planned_code.get("git_dirty") is False
        and current_code["git_revision"] == planned_code.get("git_revision")
        and current_code["git_dirty"] is False
    )
    if not args.engineering_smoke and not version_matches and not args.allow_unversioned:
        parser.error(
            "production assembly requires the same clean revision frozen in the plan; use "
            "--allow-unversioned only for a non-evidence engineering assembly"
        )
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing assembly output: {output_dir}")
    command = [
        "python",
        "scripts/assemble_profile_labels.py",
        "--plan",
        "<verified-plan>",
        "--domain",
        args.domain,
        "--model-family",
        args.model_family,
        "--data-seed",
        str(args.data_seed),
    ]
    if args.engineering_smoke:
        command.extend(["--engineering-smoke", "--output", "<engineering-output>"])
    if args.allow_unversioned:
        command.append("--allow-unversioned")
    try:
        result = assemble_crossfit_label_shards(
            tasks,
            shard_dirs,
            plan_sha256=str(plan["plan_sha256"]),
            base_config=base_config,
            output_dir=output_dir,
            producing_command=command,
            engineering_smoke=args.engineering_smoke,
            execution_code=current_code,
            execution_evidence_eligible=version_matches,
        )
    except Exception as error:
        print(f"profile-label assembly failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
