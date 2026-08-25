#!/usr/bin/env python3
"""Run one registered observed-rollout predicted-profile coverage gate."""

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
from latent_safety.learning.profile_coverage import (  # noqa: E402
    COVERAGE_BUNDLES_PER_DOMAIN_SEED,
    run_profile_coverage_gate,
)


def _managed_path(value: Path, *, root_name: str, label: str) -> Path:
    if value.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    resolved = (ROOT / value).resolve()
    managed = (ROOT / "runs" / "e2_frontier" / root_name).resolve()
    try:
        resolved.relative_to(managed)
    except ValueError as error:
        raise ValueError(
            f"{label} must resolve below runs/e2_frontier/{root_name}"
        ) from error
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-family", choices=("ae", "beta_vae"), required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--shard-dir", type=Path, action="append")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--bundle-count", type=int)
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
                raise ValueError("engineering smoke requires exactly five --shard-dir")
            if args.assembly is None or args.output is None:
                raise ValueError("engineering smoke requires --assembly and --output")
            shard_dirs = [path.resolve() for path in args.shard_dir]
            assembly_path = args.assembly.resolve()
            output_dir = _smoke_output_path(args.output)
            bundle_count = args.bundle_count or 20
        else:
            if args.shard_dir is not None or args.bundle_count is not None:
                raise ValueError(
                    "--shard-dir and --bundle-count are engineering-smoke options only"
                )
            shard_dirs = [_task_output_path(task["output_dir"]) for task in tasks]
            default_assembly = Path(
                "runs/e2_frontier/profile_labels/"
                f"{args.domain}/{args.model_family}/data_seed_{args.data_seed}/"
                "crossfit_label_manifest.json"
            )
            assembly_path = _managed_path(
                args.assembly or default_assembly,
                root_name="profile_labels",
                label="assembly",
            )
            default_output = Path(
                "runs/e2_frontier/profile_coverage/"
                f"{args.domain}/{args.model_family}/data_seed_{args.data_seed}"
            )
            output_dir = _managed_path(
                args.output or default_output,
                root_name="profile_coverage",
                label="output",
            )
            bundle_count = COVERAGE_BUNDLES_PER_DOMAIN_SEED
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if not assembly_path.is_file():
        parser.error(f"assembled label manifest is missing: {assembly_path}")

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
            "production coverage requires the same clean revision frozen in the plan; use "
            "--allow-unversioned only for a non-evidence engineering gate"
        )
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing coverage output: {output_dir}")
    command = [
        "python",
        "scripts/run_profile_coverage_gate.py",
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
        command.extend(
            [
                "--engineering-smoke",
                "--assembly",
                "<engineering-assembly>",
                "--output",
                "<engineering-output>",
                "--bundle-count",
                str(bundle_count),
            ]
        )
    if args.allow_unversioned:
        command.append("--allow-unversioned")
    try:
        result = run_profile_coverage_gate(
            tasks,
            shard_dirs,
            plan_sha256=str(plan["plan_sha256"]),
            base_config=base_config,
            assembled_manifest_path=assembly_path,
            output_dir=output_dir,
            device=args.device,
            producing_command=command,
            execution_code=current_code,
            execution_evidence_eligible=version_matches,
            engineering_smoke=args.engineering_smoke,
            bundle_count=bundle_count,
        )
    except Exception as error:
        print(f"profile coverage failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.engineering_smoke and not bool(result["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
