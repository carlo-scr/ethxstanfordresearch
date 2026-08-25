#!/usr/bin/env python3
"""Build the deterministic D1 controlled-alias figure and provenance sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.benchmarks.dynamic_oracle import (  # noqa: E402
    PRIVILEGED_STATE,
    build_controlled_dynamic_tree,
)
from latent_safety.learning import (  # noqa: E402
    CartState,
    PendulumState,
    load_config,
    render_state,
)
from latent_safety.metrics.dynamic import (  # noqa: E402
    audit_finite_dynamic_sufficiency,
)

CONFIG_PATHS = {
    "controlled_cart_video": ROOT / "configs/e1_world_models/torch_pilot.toml",
    "controlled_pendulum_video": (
        ROOT / "configs/e1_world_models/torch_pendulum_pilot.toml"
    ),
}
EVIDENCE_PATH = ROOT / "evidence/controlled_dynamic_oracle_v1.json"
DYNAMIC_ORACLE_PATH = ROOT / "src/latent_safety/benchmarks/dynamic_oracle.py"
DYNAMIC_METRICS_PATH = ROOT / "src/latent_safety/metrics/dynamic.py"
RENDERER_PATH = ROOT / "src/latent_safety/learning/data.py"
DEFAULT_PDF = ROOT / "paper/figures/d1_controlled_aliases.pdf"
DEFAULT_PNG = ROOT / "paper/figures/d1_controlled_aliases.png"
DEFAULT_SIDECAR = ROOT / "paper/figures/d1_controlled_aliases.json"

DOMAIN_DISPLAY = {
    "controlled_cart_video": "Controlled cart",
    "controlled_pendulum_video": "Controlled pendulum",
}
ROOT_DISPLAY = {
    "cart_incoming_left": "incoming left",
    "cart_incoming_right": "incoming right",
    "pendulum_fast_clockwise": "clockwise",
    "pendulum_fast_counterclockwise": "counterclockwise",
}
VELOCITY_DISPLAY = {
    "controlled_cart_video": "hidden v",
    "controlled_pendulum_video": "hidden omega",
}
FIGURE_TITLE = "D1: identical current pixels hide incompatible viable actions"
FIGURE_SUBTITLE = "Previous frames reveal motion; a memoryless current frame merges the roots."
FIXED_TIMESTAMP = datetime(2026, 8, 22, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RootVisualData:
    """One registered root and its rendered two-frame history."""

    history_id: str
    root_display: str
    previous_kinematics: tuple[float, float]
    current_kinematics: tuple[float, float]
    hidden_velocity: float
    viable_first_actions: tuple[float, ...]
    q_values: tuple[tuple[float, float], ...]
    previous_frame: tuple[float, ...]
    current_frame: tuple[float, ...]


@dataclass(frozen=True)
class DomainVisualData:
    """The pair of aliased roots for one registered controlled domain."""

    domain: str
    display_name: str
    image_size: int
    channels: int
    config_sha256: str
    roots: tuple[RootVisualData, RootVisualData]
    current_frames_pixel_identical: bool
    previous_frames_pixel_different: bool
    viable_first_action_intersection: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: tuple[float, ...]) -> str:
    canonical = json.dumps(frame, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _kinematics(state: object) -> tuple[float, float]:
    if isinstance(state, CartState):
        return state.position, state.velocity
    if isinstance(state, PendulumState):
        return state.angle, state.angular_velocity
    raise TypeError(f"unsupported D1 root state: {type(state).__name__}")


def _evidence_report(
    evidence: dict[str, Any], domain: str
) -> dict[str, Any]:
    reports = evidence.get("reports")
    if not isinstance(reports, list):
        raise ValueError("D1 evidence reports must be a list")
    matches = [report for report in reports if report.get("domain") == domain]
    if len(matches) != 1:
        raise ValueError(f"D1 evidence must contain exactly one report for {domain}")
    return matches[0]


def _validate_evidence_header(evidence: dict[str, Any]) -> None:
    if evidence.get("claim_id") != "D1-controlled-dynamic-oracle":
        raise ValueError("unexpected D1 evidence claim_id")
    if evidence.get("schema_version") != 1:
        raise ValueError("unexpected D1 evidence schema version")
    if evidence.get("status") != "verified_internal_finite_control":
        raise ValueError("D1 evidence is not the verified internal finite control")


def collect_figure_data(
    *, evidence_path: Path = EVIDENCE_PATH
) -> tuple[DomainVisualData, DomainVisualData]:
    """Load and cross-check the live registered roots against the frozen D1 evidence."""

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("D1 evidence must be a JSON object")
    _validate_evidence_header(evidence)
    domains: list[DomainVisualData] = []
    for domain, config_path in CONFIG_PATHS.items():
        config = load_config(config_path).data
        tree = build_controlled_dynamic_tree(config, horizon=3)
        report = _evidence_report(evidence, domain)
        if report.get("config_sha256") != tree.config_sha256:
            raise ValueError(f"D1 evidence config hash mismatch for {domain}")
        if report.get("fixture_version") != tree.fixture_version:
            raise ValueError(f"D1 evidence fixture version mismatch for {domain}")
        if report.get("horizon") != tree.horizon:
            raise ValueError(f"D1 evidence horizon mismatch for {domain}")
        if tuple(report.get("actions", ())) != tree.actions:
            raise ValueError(f"D1 evidence action grid mismatch for {domain}")

        fixture_roots = report.get("fixture_roots")
        if not isinstance(fixture_roots, list):
            raise ValueError(f"D1 evidence fixture roots missing for {domain}")
        fixture_by_id = {root.get("history_id"): root for root in fixture_roots}
        if set(fixture_by_id) != set(tree.root_histories):
            raise ValueError(f"D1 evidence root IDs mismatch for {domain}")

        audit = audit_finite_dynamic_sufficiency(
            actions=tree.actions,
            histories_by_remaining=tree.histories_by_remaining,
            margins=tree.margins,
            successors=tree.successors,
            codes=tree.codes_by_representation[PRIVILEGED_STATE],
            tolerance=1e-12,
        )
        roots: list[RootVisualData] = []
        for history_id in tree.root_histories:
            node = tree.nodes_by_history[history_id]
            current_kinematics = _kinematics(node.state)
            previous_kinematics = _kinematics(node.previous_state)
            frozen_root = fixture_by_id[history_id]
            if tuple(frozen_root.get("current_kinematics", ())) != current_kinematics:
                raise ValueError(f"D1 current-root mismatch for {history_id}")
            if tuple(frozen_root.get("previous_kinematics", ())) != previous_kinematics:
                raise ValueError(f"D1 predecessor mismatch for {history_id}")
            q_values = tuple(
                (
                    action,
                    float(audit.q_values[tree.horizon][(history_id, action)]),
                )
                for action in tree.actions
            )
            viable_actions = tuple(
                action for action, value in q_values if value >= 0.0
            )
            if not viable_actions:
                raise ValueError(f"registered D1 root has no viable first action: {history_id}")
            roots.append(
                RootVisualData(
                    history_id=history_id,
                    root_display=ROOT_DISPLAY[history_id],
                    previous_kinematics=previous_kinematics,
                    current_kinematics=current_kinematics,
                    hidden_velocity=current_kinematics[1],
                    viable_first_actions=viable_actions,
                    q_values=q_values,
                    previous_frame=tuple(render_state(node.previous_state, config)),
                    current_frame=tuple(render_state(node.state, config)),
                )
            )
        if len(roots) != 2:
            raise AssertionError(f"registered D1 domain must have two roots: {domain}")

        current_identical = roots[0].current_frame == roots[1].current_frame
        previous_different = roots[0].previous_frame != roots[1].previous_frame
        if not current_identical:
            raise AssertionError(f"D1 current frames are not pixel-identical for {domain}")
        if not previous_different:
            raise AssertionError(f"D1 previous frames do not separate roots for {domain}")
        viable_intersection = tuple(
            sorted(set(roots[0].viable_first_actions) & set(roots[1].viable_first_actions))
        )
        if viable_intersection:
            raise AssertionError(f"D1 roots unexpectedly share a viable first action for {domain}")

        representations = report.get("representations")
        if not isinstance(representations, list):
            raise ValueError(f"D1 evidence representations missing for {domain}")
        memoryless = [
            item
            for item in representations
            if item.get("representation") == "memoryless_rendered_frame_exact"
        ]
        if len(memoryless) != 1:
            raise ValueError(f"D1 evidence memoryless record missing for {domain}")
        fiber_records = memoryless[0].get("root_fiber_safety")
        if not isinstance(fiber_records, list) or len(fiber_records) != 1:
            raise ValueError(f"D1 evidence root fiber mismatch for {domain}")
        if tuple(fiber_records[0].get("histories", ())) != tree.root_histories:
            raise ValueError(f"D1 evidence fiber histories mismatch for {domain}")
        if fiber_records[0].get("common_safe_actions") != []:
            raise ValueError(f"D1 evidence no-common-action claim mismatch for {domain}")

        domains.append(
            DomainVisualData(
                domain=domain,
                display_name=DOMAIN_DISPLAY[domain],
                image_size=config.image_size,
                channels=config.channels,
                config_sha256=tree.config_sha256,
                roots=(roots[0], roots[1]),
                current_frames_pixel_identical=current_identical,
                previous_frames_pixel_different=previous_different,
                viable_first_action_intersection=viable_intersection,
            )
        )
    if len(domains) != 2:
        raise AssertionError("D1 figure requires exactly the cart and pendulum domains")
    return domains[0], domains[1]


def _action_display(actions: tuple[float, ...]) -> str:
    def token(action: float) -> str:
        if action == 0.0:
            return "0"
        if action == float(int(action)):
            return f"{int(action):+d}"
        return f"{action:+g}"

    return "{" + ", ".join(token(action) for action in actions) + "}"


def _frame_image(root: RootVisualData, domain: DomainVisualData, *, previous: bool) -> Any:
    import numpy as np

    flat = root.previous_frame if previous else root.current_frame
    channel_first = np.asarray(flat, dtype=float).reshape(
        domain.channels,
        domain.image_size,
        domain.image_size,
    )
    return np.moveaxis(channel_first, 0, -1)


def _configure_matplotlib() -> Any:
    cache = Path(tempfile.gettempdir()) / "latentsafety-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "semibold",
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    return matplotlib


def _render_figure(domains: tuple[DomainVisualData, DomainVisualData]) -> Any:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    previous_color = "#B45309"
    current_color = "#0F766E"
    ink = "#172033"
    muted = "#526075"
    figure, axes = plt.subplots(2, 4, figsize=(7.15, 4.55), dpi=100)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        FIGURE_TITLE,
        x=0.515,
        y=0.975,
        fontsize=12.2,
        fontweight="bold",
        color=ink,
    )
    figure.text(
        0.515,
        0.925,
        FIGURE_SUBTITLE,
        ha="center",
        va="center",
        fontsize=8.4,
        color=muted,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.992,
        top=0.835,
        bottom=0.085,
        wspace=0.10,
        hspace=0.58,
    )
    for row, domain in enumerate(domains):
        figure.text(
            0.015,
            0.695 if row == 0 else 0.295,
            domain.display_name.replace(" ", "\n", 1),
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=ink,
        )
        for root_index, root in enumerate(domain.roots):
            velocity_name = VELOCITY_DISPLAY[domain.domain]
            group_label = (
                f"Root {root_index + 1}: {root.root_display}\n"
                f"{velocity_name} = {root.hidden_velocity:+.2f}  |  "
                f"viable a = {_action_display(root.viable_first_actions)}"
            )
            first_axis = axes[row, 2 * root_index]
            second_axis = axes[row, 2 * root_index + 1]
            left = first_axis.get_position().x0
            right = second_axis.get_position().x1
            y = first_axis.get_position().y1 + 0.039
            figure.text(
                (left + right) / 2.0,
                y,
                group_label,
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=ink,
                fontweight="semibold",
                linespacing=0.95,
            )
            for offset, previous in enumerate((True, False)):
                axis = axes[row, 2 * root_index + offset]
                color = previous_color if previous else current_color
                axis.imshow(
                    _frame_image(root, domain, previous=previous),
                    interpolation="nearest",
                    vmin=0.0,
                    vmax=1.0,
                    rasterized=True,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(
                    "previous frame" if previous else "current frame",
                    color=color,
                    pad=3.0,
                )
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.8)
                    spine.set_color(color)
                if not previous:
                    axis.text(
                        0.96,
                        0.05,
                        "PIXEL-IDENTICAL",
                        transform=axis.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=5.6,
                        fontweight="bold",
                        color="white",
                        bbox={
                            "boxstyle": "round,pad=0.23",
                            "facecolor": current_color,
                            "edgecolor": "none",
                            "alpha": 0.95,
                        },
                    )
        row_axes = axes[row]
        row_y = min(axis.get_position().y0 for axis in row_axes) - 0.027
        figure.text(
            0.515,
            row_y,
            "Current pixels: one shared code   |   Previous pixels: two distinct histories   "
            "|   Common viable first actions: empty",
            ha="center",
            va="top",
            fontsize=7.0,
            color=muted,
        )
    figure.text(
        0.515,
        0.018,
        "Registered three-step, noise-free finite-action trees; hidden velocities are shown "
        "for interpretation only.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=muted,
    )
    return figure


def _save_figure(figure: Any, *, pdf_path: Path, png_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_temporary = pdf_path.with_name(pdf_path.name + ".tmp")
    png_temporary = png_path.with_name(png_path.name + ".tmp")
    metadata = {
        "Title": FIGURE_TITLE,
        "Author": "LatentSafety collaborators",
        "Subject": "Registered D1 controlled current-frame aliases",
        "Keywords": "latent safety, partial observability, controlled alias",
        "Creator": "scripts/build_d1_alias_figure.py",
        "Producer": "Matplotlib",
        "CreationDate": FIXED_TIMESTAMP,
        "ModDate": FIXED_TIMESTAMP,
    }
    figure.savefig(
        pdf_temporary,
        format="pdf",
        dpi=300,
        metadata=metadata,
    )
    figure.savefig(
        png_temporary,
        format="png",
        dpi=300,
        metadata={"Software": "Matplotlib", "Title": FIGURE_TITLE},
    )
    os.replace(pdf_temporary, pdf_path)
    os.replace(png_temporary, png_path)


def _domain_payload(domain: DomainVisualData) -> dict[str, Any]:
    return {
        "domain": domain.domain,
        "display_name": domain.display_name,
        "frame_shape_hwc": [domain.image_size, domain.image_size, domain.channels],
        "oracle_config_sha256": domain.config_sha256,
        "current_frames_pixel_identical": domain.current_frames_pixel_identical,
        "previous_frames_pixel_different": domain.previous_frames_pixel_different,
        "viable_first_action_intersection": domain.viable_first_action_intersection,
        "roots": [
            {
                "history_id": root.history_id,
                "display_name": root.root_display,
                "previous_kinematics": root.previous_kinematics,
                "current_kinematics": root.current_kinematics,
                "hidden_velocity": root.hidden_velocity,
                "viable_first_actions": root.viable_first_actions,
                "q_values": [
                    {"action": action, "value": value}
                    for action, value in root.q_values
                ],
                "previous_frame_sha256": _frame_sha256(root.previous_frame),
                "current_frame_sha256": _frame_sha256(root.current_frame),
            }
            for root in domain.roots
        ],
    }


def build_artifacts(
    *,
    pdf_path: Path = DEFAULT_PDF,
    png_path: Path = DEFAULT_PNG,
    sidecar_path: Path = DEFAULT_SIDECAR,
    evidence_path: Path = EVIDENCE_PATH,
) -> dict[str, Any]:
    """Build all three deterministic artifacts and return the sidecar payload."""

    resolved_pdf = _resolve_output(pdf_path)
    resolved_png = _resolve_output(png_path)
    resolved_sidecar = _resolve_output(sidecar_path)
    resolved_evidence = _resolve_output(evidence_path)
    if len({resolved_pdf, resolved_png, resolved_sidecar}) != 3:
        raise ValueError("PDF, PNG, and sidecar output paths must be distinct")
    if resolved_pdf.suffix.lower() != ".pdf":
        raise ValueError("PDF output path must end in .pdf")
    if resolved_png.suffix.lower() != ".png":
        raise ValueError("PNG output path must end in .png")
    if resolved_sidecar.suffix.lower() != ".json":
        raise ValueError("sidecar output path must end in .json")

    domains = collect_figure_data(evidence_path=resolved_evidence)
    figure = _render_figure(domains)
    try:
        _save_figure(figure, pdf_path=resolved_pdf, png_path=resolved_png)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

    source_paths = (
        Path(__file__).resolve(),
        DYNAMIC_ORACLE_PATH,
        DYNAMIC_METRICS_PATH,
        RENDERER_PATH,
    )
    config_paths = tuple(CONFIG_PATHS.values())
    payload = {
        "schema_version": 1,
        "artifact": "D1 controlled current-frame aliases",
        "generator": "scripts/build_d1_alias_figure.py",
        "claim_id": "D1-controlled-dynamic-oracle",
        "scope": (
            "registered two-root, three-step, finite-action, noise-free nominal controls; "
            "not learned-model or population evidence"
        ),
        "assertions": {
            "all_current_frame_pairs_pixel_identical": all(
                domain.current_frames_pixel_identical for domain in domains
            ),
            "all_previous_frame_pairs_pixel_different": all(
                domain.previous_frames_pixel_different for domain in domains
            ),
            "all_viable_first_action_intersections_empty": all(
                not domain.viable_first_action_intersection for domain in domains
            ),
        },
        "domains": [_domain_payload(domain) for domain in domains],
        "provenance": {
            "evidence": {
                "path": _display_path(resolved_evidence),
                "sha256": _sha256(resolved_evidence),
            },
            "source_files": {
                _display_path(path): _sha256(path) for path in source_paths
            },
            "config_files": {
                _display_path(path): _sha256(path) for path in config_paths
            },
        },
        "outputs": {
            "pdf": {
                "path": _display_path(resolved_pdf),
                "sha256": _sha256(resolved_pdf),
            },
            "png": {
                "path": _display_path(resolved_png),
                "sha256": _sha256(resolved_png),
            },
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    resolved_sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_sidecar.with_name(resolved_sidecar.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, resolved_sidecar)
    normalized = json.loads(serialized)
    if not isinstance(normalized, dict):  # pragma: no cover - serialization invariant
        raise AssertionError("sidecar serialization did not produce a JSON object")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    arguments = parser.parse_args()
    payload = build_artifacts(
        pdf_path=arguments.pdf,
        png_path=arguments.png,
        sidecar_path=arguments.sidecar,
        evidence_path=arguments.evidence,
    )
    print(json.dumps(payload["outputs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
