#!/usr/bin/env python3
"""Validate claim scope, manuscript inclusion, and evidence provenance."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 2
ALLOWED_STATUSES = {
    "hypothesis",
    "theorem_needs_proof",
    "pilot_unreproduced",
    "verified_internal",
    "verified_independent",
    "rejected",
}
ALLOWED_SCOPES = {"included", "prospective", "excluded", "historical"}
ALLOWED_PROMOTION_STATES = {
    "provisional_internal",
    "independently_reproduced",
}
FORMAL_ENVIRONMENTS = ("theorem", "proposition", "lemma", "corollary")
EMPIRICAL_ENVIRONMENTS = ("table", "table*", "figure", "figure*")


class ValidationCounts(NamedTuple):
    claims: int
    formal_labels: int
    empirical_labels: int
    provisional_empirical_claims: int


def _included_tex_paths(root: Path) -> tuple[Path, ...]:
    """Return transitive section inputs in manuscript order, rejecting cycles."""

    paper_root = root / "paper"
    ordered: list[Path] = []
    visited: set[Path] = set()
    active: set[Path] = set()

    def resolve(relative: str) -> Path:
        candidate = paper_root / relative
        return candidate if candidate.suffix == ".tex" else candidate.with_suffix(".tex")

    def visit(path: Path) -> None:
        if path in active:
            raise ValueError(f"cyclic manuscript input: {path.relative_to(root)}")
        if path in visited:
            return
        visited.add(path)
        active.add(path)
        ordered.append(path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for relative in re.findall(r"\\input\{([^}]+)\}", text):
                visit(resolve(relative))
        active.remove(path)

    main_tex = (paper_root / "main.tex").read_text(encoding="utf-8")
    for relative in re.findall(r"\\input\{([^}]+)\}", main_tex):
        visit(resolve(relative))
    return tuple(ordered)


def _environment_labels(
    path: Path,
    environments: tuple[str, ...],
    *,
    result_kind: str,
) -> tuple[str, ...]:
    """Extract the first label from each selected environment, failing closed."""

    text = path.read_text(encoding="utf-8")
    alternatives = "|".join(re.escape(environment) for environment in environments)
    labels: list[str] = []
    for match in re.finditer(rf"\\begin\{{({alternatives})\}}", text):
        environment = match.group(1)
        end_token = f"\\end{{{environment}}}"
        end = text.find(end_token, match.end())
        if end < 0:
            raise ValueError(f"{path}: unterminated {environment} environment")
        body = text[match.end() : end]
        label_match = re.search(r"\\label\{([^}]+)\}", body)
        if label_match is None:
            raise ValueError(
                f"{path}: unlabeled {result_kind} {environment} environment"
            )
        labels.append(label_match.group(1))
    return tuple(labels)


def _formal_result_labels(path: Path) -> tuple[str, ...]:
    """Extract labels for all included formal-result environments."""

    return _environment_labels(
        path,
        FORMAL_ENVIRONMENTS,
        result_kind="formal-result",
    )


def _empirical_result_labels(path: Path) -> tuple[str, ...]:
    """Extract labels for every included table and figure."""

    return _environment_labels(
        path,
        EMPIRICAL_ENVIRONMENTS,
        result_kind="empirical-result",
    )


def _normalize_latex_cell(value: object) -> str:
    text = str(value).strip().replace("$", "")
    return re.sub(r"\s+", " ", text)


def _latex_table_rows_for_label(
    paths: tuple[Path, ...], label: str
) -> tuple[tuple[str, ...], ...]:
    """Extract normalized data cells between midrule and bottomrule for one table."""

    label_token = f"\\label{{{label}}}"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\begin\{(table\*?)\}", text):
            environment = match.group(1)
            end = text.find(f"\\end{{{environment}}}", match.end())
            if end < 0:
                raise ValueError(f"{path}: unterminated {environment} environment")
            body = text[match.end() : end]
            if label_token not in body:
                continue
            tabular_start = body.find(r"\begin{tabular}")
            tabular_end = body.find(r"\end{tabular}", tabular_start)
            if tabular_start < 0 or tabular_end < 0:
                raise ValueError(f"{path}: {label!r} has no complete tabular environment")
            tabular = body[tabular_start:tabular_end]
            if r"\midrule" not in tabular or r"\bottomrule" not in tabular:
                raise ValueError(
                    f"{path}: {label!r} must delimit data with midrule and bottomrule"
                )
            data = tabular.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
            rows: list[tuple[str, ...]] = []
            for raw_row in re.split(r"\\\\", data):
                without_comments = re.sub(r"(?m)%.*$", "", raw_row).strip()
                if not without_comments:
                    continue
                cells = tuple(
                    _normalize_latex_cell(cell)
                    for cell in without_comments.split("&")
                )
                rows.append(cells)
            return tuple(rows)
    raise ValueError(f"included table label not found: {label!r}")


def _validate_table_value_check(
    claim: dict[str, Any],
    manifest: Path,
    included_paths: tuple[Path, ...],
    errors: list[str],
) -> None:
    claim_id = claim.get("id")
    if claim.get("value_check") != "latex_tabular_rows_v1":
        errors.append(
            f"{claim_id}: included empirical claim requires "
            "value_check = 'latex_tabular_rows_v1'"
        )
        return
    labels = claim.get("empirical_labels", [])
    if not isinstance(labels, list):
        errors.append(f"{claim_id}: empirical_labels must be a list")
        return
    raw_value_label = claim.get("value_label")
    if raw_value_label is None:
        if len(labels) != 1:
            errors.append(
                f"{claim_id}: table value check requires value_label when the "
                "claim registers multiple empirical labels"
            )
            return
        value_label = labels[0]
    elif not isinstance(raw_value_label, str) or not raw_value_label.strip():
        errors.append(f"{claim_id}: value_label must be a non-empty string")
        return
    elif raw_value_label not in labels:
        errors.append(
            f"{claim_id}: value_label must name one of the registered empirical labels"
        )
        return
    else:
        value_label = raw_value_label
    columns = claim.get("value_columns")
    if not isinstance(columns, list) or not columns or any(
        not isinstance(column, str) or not column.strip() for column in columns
    ):
        errors.append(f"{claim_id}: value_columns must be non-empty strings")
        return
    manifest_key = claim.get("value_manifest_key")
    if not isinstance(manifest_key, str) or not manifest_key.strip():
        errors.append(f"{claim_id}: missing value_manifest_key")
        return
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload[manifest_key]
        if not isinstance(records, list):
            raise TypeError("manifest value is not a list")
        expected = tuple(
            tuple(_normalize_latex_cell(record[column]) for column in columns)
            for record in records
        )
        actual = _latex_table_rows_for_label(included_paths, value_label)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{claim_id}: table value check failed to load: {error}")
        return
    if actual != expected:
        errors.append(
            f"{claim_id}: TeX table rows differ from manifest: "
            f"expected {expected!r}, got {actual!r}"
        )


def _string_list(
    claim: dict[str, Any],
    field: str,
    claim_id: object,
    errors: list[str],
) -> tuple[str, ...]:
    raw = claim.get(field, [])
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        errors.append(f"{claim_id}: {field} must be a list of non-empty strings")
        return ()
    return tuple(raw)


def _relative_file(
    root: Path,
    raw_path: object,
    *,
    claim_id: object,
    field: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"{claim_id}: missing {field}")
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{claim_id}: {field} must be a repository-relative path")
        return None
    resolved = root / relative
    if not resolved.is_file():
        errors.append(f"{claim_id}: {field} does not exist: {raw_path!r}")
        return None
    return resolved


def _register_labels(
    claim_id: object,
    labels: tuple[str, ...],
    destination: dict[str, str],
    *,
    kind: str,
    errors: list[str],
) -> None:
    for label in labels:
        if label in destination:
            errors.append(
                f"{claim_id}: {kind} label {label!r} already registered by "
                f"{destination[label]}"
            )
        destination[label] = str(claim_id)


def validate_registry(
    root: Path, registry: dict[str, Any]
) -> tuple[list[str], ValidationCounts]:
    """Return all validation errors and summary counts without mutating the repository."""

    errors: list[str] = []
    if registry.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"unsupported claim schema version: {registry.get('schema_version')!r}"
        )

    claims = registry.get("claims", [])
    if not isinstance(claims, list):
        errors.append("claims must be an array of tables")
        claims = []

    seen: set[str] = set()
    registered_formal_labels: dict[str, str] = {}
    registered_empirical_labels: dict[str, str] = {}
    provisional_empirical_claims = 0
    table_value_checks: list[tuple[dict[str, Any], Path]] = []

    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("every claim must be a table")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip() or claim_id in seen:
            errors.append(f"invalid or duplicate claim id: {claim_id!r}")
        else:
            seen.add(claim_id)

        status = claim.get("status")
        scope = claim.get("scope")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{claim_id}: invalid status {status!r}")
        if scope not in ALLOWED_SCOPES:
            errors.append(f"{claim_id}: invalid or missing scope {scope!r}")
        for field in ("statement", "status", "scope", "evidence"):
            if not str(claim.get(field, "")).strip():
                errors.append(f"{claim_id}: missing {field}")

        if status == "hypothesis" and scope != "prospective":
            errors.append(f"{claim_id}: hypotheses must have prospective scope")
        if scope == "prospective" and status != "hypothesis":
            errors.append(f"{claim_id}: prospective scope requires hypothesis status")
        if status == "pilot_unreproduced" and scope != "historical":
            errors.append(f"{claim_id}: unreproduced pilots must have historical scope")
        if scope == "historical" and status not in {"pilot_unreproduced", "rejected"}:
            errors.append(
                f"{claim_id}: historical scope requires pilot_unreproduced or rejected status"
            )

        tex_labels = _string_list(claim, "tex_labels", claim_id, errors)
        empirical_labels = _string_list(
            claim, "empirical_labels", claim_id, errors
        )
        if "manuscript_labels" in claim:
            errors.append(
                f"{claim_id}: manuscript_labels is obsolete; use empirical_labels"
            )
        if scope == "included" and not tex_labels and not empirical_labels:
            errors.append(
                f"{claim_id}: included claims require a formal or empirical manuscript label"
            )
        if scope in {"prospective", "excluded", "historical"} and (
            tex_labels or empirical_labels
        ):
            errors.append(
                f"{claim_id}: {scope} claims cannot register included manuscript labels"
            )
        _register_labels(
            claim_id,
            tex_labels,
            registered_formal_labels,
            kind="formal-result",
            errors=errors,
        )
        _register_labels(
            claim_id,
            empirical_labels,
            registered_empirical_labels,
            kind="empirical-result",
            errors=errors,
        )

        artifact_paths: tuple[str, ...] = ()
        if isinstance(status, str) and status.startswith("verified"):
            if not str(claim.get("command", "")).strip():
                errors.append(f"{claim_id}: verified claims require a producing command")
            artifact_paths = _string_list(
                claim, "artifact_paths", claim_id, errors
            )
            if not artifact_paths:
                errors.append(
                    f"{claim_id}: verified claims require non-empty artifact_paths"
                )
            for artifact_path in artifact_paths:
                _relative_file(
                    root,
                    artifact_path,
                    claim_id=claim_id,
                    field="evidence artifact",
                    errors=errors,
                )

        if empirical_labels and isinstance(status, str) and status.startswith("verified"):
            manifest_path = claim.get("manifest")
            manifest = _relative_file(
                root,
                manifest_path,
                claim_id=claim_id,
                field="manifest",
                errors=errors,
            )
            if isinstance(manifest_path, str) and manifest_path not in artifact_paths:
                errors.append(f"{claim_id}: manifest must also appear in artifact_paths")
            expected_digest = claim.get("manifest_sha256")
            if not isinstance(expected_digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_digest
            ):
                errors.append(f"{claim_id}: manifest_sha256 must be 64 lowercase hex digits")
            elif manifest is not None:
                actual_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
                if actual_digest != expected_digest:
                    errors.append(
                        f"{claim_id}: manifest_sha256 mismatch: expected "
                        f"{expected_digest}, got {actual_digest}"
                    )
            if not str(claim.get("reviewer", "")).strip():
                errors.append(f"{claim_id}: included empirical claim requires reviewer")
            promotion_state = claim.get("promotion_state")
            if promotion_state not in ALLOWED_PROMOTION_STATES:
                errors.append(
                    f"{claim_id}: invalid or missing promotion_state {promotion_state!r}"
                )
            if status == "verified_internal":
                if promotion_state != "provisional_internal":
                    errors.append(
                        f"{claim_id}: verified_internal empirical evidence must remain "
                        "provisional_internal"
                    )
                else:
                    provisional_empirical_claims += 1
            if status == "verified_independent" and promotion_state != "independently_reproduced":
                errors.append(
                    f"{claim_id}: verified_independent empirical evidence requires "
                    "independently_reproduced promotion_state"
                )
            if manifest is not None:
                table_value_checks.append((claim, manifest))

        if status == "verified_independent":
            for field in ("manifest", "independent_evidence", "reviewer"):
                if not str(claim.get(field, "")).strip():
                    errors.append(f"{claim_id}: verified_independent requires {field}")

    included_formal_labels: set[str] = set()
    included_empirical_labels: set[str] = set()
    included_paths: tuple[Path, ...] = ()
    try:
        included_paths = _included_tex_paths(root)
        for tex_path in included_paths:
            if not tex_path.is_file():
                errors.append(
                    f"manuscript input does not exist: {tex_path.relative_to(root)}"
                )
                continue
            for label in _formal_result_labels(tex_path):
                if label in included_formal_labels:
                    errors.append(f"duplicate formal-result TeX label: {label!r}")
                included_formal_labels.add(label)
            for label in _empirical_result_labels(tex_path):
                if label in included_empirical_labels:
                    errors.append(f"duplicate empirical-result TeX label: {label!r}")
                included_empirical_labels.add(label)
    except ValueError as error:
        errors.append(str(error))

    for claim, manifest in table_value_checks:
        _validate_table_value_check(claim, manifest, included_paths, errors)

    for label in sorted(included_formal_labels - registered_formal_labels.keys()):
        errors.append(f"formal manuscript result is not claim-registered: {label!r}")
    for label in sorted(registered_formal_labels.keys() - included_formal_labels):
        errors.append(f"registered formal label is not an included result: {label!r}")
    for label in sorted(included_empirical_labels - registered_empirical_labels.keys()):
        errors.append(f"empirical manuscript result is not claim-registered: {label!r}")
    for label in sorted(registered_empirical_labels.keys() - included_empirical_labels):
        errors.append(f"registered empirical label is not an included result: {label!r}")
    if not seen:
        errors.append("claim registry is empty")

    return errors, ValidationCounts(
        claims=len(seen),
        formal_labels=len(included_formal_labels),
        empirical_labels=len(included_empirical_labels),
        provisional_empirical_claims=provisional_empirical_claims,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "paper" / "claims.toml"
    with path.open("rb") as stream:
        registry = tomllib.load(stream)
    errors, counts = validate_registry(root, registry)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    empirical_noun = "label" if counts.empirical_labels == 1 else "labels"
    print(
        f"validated {counts.claims} paper claims, "
        f"{counts.formal_labels} formal-result labels, "
        f"{counts.empirical_labels} empirical manuscript {empirical_noun}, and "
        f"{counts.provisional_empirical_claims} provisional internal empirical claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
