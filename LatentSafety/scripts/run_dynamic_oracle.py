#!/usr/bin/env python3
"""Run the exhaustive registered finite-tree oracle on both controlled domains."""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.benchmarks.dynamic_oracle import (  # noqa: E402
    audit_controlled_dynamic_oracle,
)
from latent_safety.learning import load_config  # noqa: E402

CONFIG_PATHS = (
    ROOT / "configs/e1_world_models/torch_pilot.toml",
    ROOT / "configs/e1_world_models/torch_pendulum_pilot.toml",
)

DOMAIN_DISPLAY_NAMES = {
    "controlled_cart_video": "Cart",
    "controlled_pendulum_video": "Pendulum",
}
REPRESENTATION_DISPLAY_NAMES = {
    "privileged_kinematic_state_exact": "privileged state",
    "memoryless_rendered_frame_exact": "current frame",
    "two_frame_rendered_history_exact": "causal two-frame history",
}


def _full_payload(reports: list[object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "finite_nominal_theorem_oracle_not_empirical_evidence",
        "reports": [dataclasses.asdict(report) for report in reports],
    }


def _claim_evidence_payload(reports: list[object]) -> dict[str, object]:
    """Build the compact, deterministic evidence record behind the manuscript table."""

    table_rows: list[dict[str, object]] = []
    evidence_reports: list[dict[str, object]] = []
    for report in reports:
        domain_display = DOMAIN_DISPLAY_NAMES[report.domain]
        representation_records: list[dict[str, object]] = []
        for summary in report.representations:
            root_stages = [
                stage
                for stage in summary.stages
                if stage.remaining_steps == report.horizon
            ]
            if len(root_stages) != 1:
                raise AssertionError(
                    f"expected one root stage for {report.domain}/{summary.representation}"
                )
            root_kappa = root_stages[0].kappa_s
            display_kappa = "0" if root_kappa == 0.0 else f"{root_kappa:.6f}"
            global_policy = summary.exact_viability_preserving_code_policy_exists
            table_rows.append(
                {
                    "domain": domain_display,
                    "representation": REPRESENTATION_DISPLAY_NAMES[
                        summary.representation
                    ],
                    "root_kappa_3_display": display_kappa,
                    "root_kappa_3_exact": root_kappa,
                    "global_policy_display": "yes" if global_policy else "no",
                    "global_policy_exact": global_policy,
                }
            )
            root_stage = root_stages[0]
            representation_records.append(
                {
                    "exact_viability_preserving_code_policy_exists": global_policy,
                    "global_sign_characterization_holds": (
                        summary.global_sign_characterization_holds
                    ),
                    "representation": summary.representation,
                    "root_code_count": summary.root_code_count,
                    "root_fiber_safety": [
                        dataclasses.asdict(fiber)
                        for fiber in summary.root_fiber_safety
                    ],
                    "root_outcomes": [
                        dataclasses.asdict(outcome)
                        for outcome in summary.root_outcomes
                    ],
                    "root_stage": {
                        "delta_s_star": root_stage.delta_s_star,
                        "fibers_without_common_safe_action": (
                            root_stage.fibers_without_common_safe_action
                        ),
                        "full_history_viable_count": (
                            root_stage.full_history_viable_count
                        ),
                        "kappa_s": root_stage.kappa_s,
                        "remaining_steps": root_stage.remaining_steps,
                        "rho_s": root_stage.rho_s,
                        "selected_policy_retained_viable_count": (
                            root_stage.selected_policy_retained_viable_count
                        ),
                        "selected_policy_retained_viable_fraction": (
                            root_stage.selected_policy_retained_viable_fraction
                        ),
                    },
                    "stagewise_first_action_feasible_within_tolerance": (
                        summary.stagewise_first_action_feasible_within_tolerance
                    ),
                }
            )
        evidence_reports.append(
            {
                "actions": report.actions,
                "config_sha256": report.config_sha256,
                "domain": report.domain,
                "fixture_roots": [
                    dataclasses.asdict(root) for root in report.fixture_roots
                ],
                "fixture_version": report.fixture_version,
                "horizon": report.horizon,
                "node_count_by_remaining": report.node_count_by_remaining,
                "population_certificate": report.population_certificate,
                "representations": representation_records,
                "tolerance": report.tolerance,
            }
        )
    return {
        "claim_id": "D1-controlled-dynamic-oracle",
        "generator": "scripts/run_dynamic_oracle.py --format claim-evidence",
        "manuscript_label": "tab:dynamic-controls",
        "manuscript_table_rows": table_rows,
        "reports": evidence_reports,
        "schema_version": 1,
        "scope": (
            "complete floating-point enumeration of the registered roots, finite action "
            "grid, three-step noise-free nominal trees, exact code equality, and recorded "
            "tolerance; not learned-model or population evidence"
        ),
        "source_paths": [
            "configs/e1_world_models/torch_pilot.toml",
            "configs/e1_world_models/torch_pendulum_pilot.toml",
            "src/latent_safety/benchmarks/dynamic_oracle.py",
            "src/latent_safety/metrics/dynamic.py",
        ],
        "status": "verified_internal_finite_control",
    }


def _resolved_output(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(3,),
        default=3,
        help="Registered finite horizon (currently fixed at 3).",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON path. Existing files are never overwritten.",
    )
    destination.add_argument(
        "--check",
        type=Path,
        help="Fail unless this existing JSON file exactly matches regenerated output.",
    )
    parser.add_argument(
        "--format",
        choices=("full", "claim-evidence"),
        default="full",
        help="Full oracle report or compact manuscript-claim evidence.",
    )
    args = parser.parse_args()

    reports = [
        audit_controlled_dynamic_oracle(
            load_config(config_path).data, horizon=args.horizon
        )
        for config_path in CONFIG_PATHS
    ]
    payload = (
        _full_payload(reports)
        if args.format == "full"
        else _claim_evidence_payload(reports)
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check is not None:
        expected_path = _resolved_output(args.check)
        expected = expected_path.read_text(encoding="utf-8")
        if expected != serialized:
            print(
                "".join(
                    difflib.unified_diff(
                        expected.splitlines(keepends=True),
                        serialized.splitlines(keepends=True),
                        fromfile=str(expected_path),
                        tofile="regenerated",
                    )
                ),
                file=sys.stderr,
                end="",
            )
            return 1
        print(f"verified {expected_path}")
        return 0
    if args.output is None:
        print(serialized, end="")
        return 0

    output = _resolved_output(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
