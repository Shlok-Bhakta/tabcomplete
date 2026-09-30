#!/usr/bin/env python3
"""Export already-qualified public typed-prefix rows and portable role receipts.

The output is a private candidate bundle, not a trainer manifest. This script
does no provider, sandbox, compilation, or training work.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION
from tinycomplete.one_line.contract import EditAction, EditState, apply_action
from tinycomplete.one_line.fixed_state_role_receipts import (
    export_fixed_state_role_execution_proof,
)
from tinycomplete.one_line.pilot_data import (
    LICENSE_MIXED_HISTORY,
    PUBLIC_PREFIX_ROLE_BUNDLE_SCHEMA,
    REVIEWED_PUBLIC_HISTORY_SOURCE,
    TYPED_RETURN_PREFIX_TRANSFORM,
    _canonical_bytes,
    _sha256,
    validate_public_prefix_role_bundle,
)

DEFAULT_ORACLE_PACKAGE = Path(
    "/mnt/ssd/tabcomplete-product-r2/commitpackft/python-prefix-objective-oracle-v4"
)
DEFAULT_ROLE_RUN = Path("/mnt/ssd/tabcomplete-product-r2/fixed-state-role-phase-r1")
DEFAULT_USAGE_LEDGER = Path(
    "/home/crabcake/Projects/tabcomplete-product-r2/reports/prototype/product_r2/teacher_usage.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/mnt/ssd/tabcomplete-product-r2/public-prefix-history-role-bundle-v1"
)
SOURCE_PACKAGE_DIR = "source_oracle_v4"
MAX_PACKAGE_BYTES = 64 * 1024 * 1024


class BundleExportError(ValueError):
    """Raised when an immutable source or role artifact does not qualify."""


def _json_unique(payload: bytes, *, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise BundleExportError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeError):
        raise BundleExportError(f"{label} is not strict JSON") from None


def _jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        payload = path.read_bytes()
        for line in payload.splitlines():
            row = _json_unique(line, label=label)
            if not isinstance(row, dict):
                raise BundleExportError(f"{label} contains a nonobject row")
            rows.append(row)
    except OSError:
        raise BundleExportError(f"{label} is unavailable") from None
    return rows


def _index(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in indexed:
            raise BundleExportError(f"{label} has duplicate or missing candidate IDs")
        indexed[candidate_id] = row
    return indexed


def _safe_package_path(path_value: object, *, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise BundleExportError(f"{label} path is missing")
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise BundleExportError(f"{label} path escapes its package")
    return path


def _read_manifest_file(
    source_root: Path, files: Mapping[str, Any], relative: str, *, label: str
) -> bytes:
    descriptor = files.get(relative)
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256", "bytes"}
        or descriptor.get("path") != relative
        or type(descriptor.get("bytes")) is not int
        or descriptor["bytes"] < 0
    ):
        raise BundleExportError(f"{label} is not pinned by the source manifest")
    path = source_root / relative
    try:
        current = source_root
        for part in Path(relative).parts:
            current = current / part
            if current.is_symlink():
                raise BundleExportError(f"{label} path contains a symlink")
        if not path.is_file() or path.is_symlink():
            raise BundleExportError(f"{label} is not a regular file")
        payload = path.read_bytes()
    except OSError:
        raise BundleExportError(f"{label} is unavailable") from None
    if len(payload) != descriptor["bytes"] or _sha256(payload) != descriptor["sha256"]:
        raise BundleExportError(f"{label} does not match its frozen digest")
    return payload


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_frozen_source_package(source: Path, destination: Path) -> dict[str, Any]:
    manifest_path = source / "manifest.json"
    try:
        if source.is_symlink() or not source.is_dir() or manifest_path.is_symlink():
            raise BundleExportError("source oracle package root is not a regular directory")
        manifest_payload = manifest_path.read_bytes()
    except OSError:
        raise BundleExportError("source oracle manifest is unavailable") from None
    source_manifest = _json_unique(manifest_payload, label="source oracle manifest")
    files = source_manifest.get("files") if isinstance(source_manifest, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise BundleExportError("source oracle manifest has no file inventory")
    total_bytes = 0
    for relative in sorted(files):
        relative_path = _safe_package_path(relative, label="source inventory")
        payload = _read_manifest_file(source, files, str(relative), label="source inventory file")
        total_bytes += len(payload)
        if total_bytes > MAX_PACKAGE_BYTES:
            raise BundleExportError("source oracle package exceeds the export byte limit")
        _write_private(destination / relative_path, payload)
    _write_private(destination / "manifest.json", manifest_payload)
    os.chmod(destination, 0o700)
    return {
        "manifest": source_manifest,
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": _sha256(manifest_payload),
        "plan": _json_unique((destination / "plan.json").read_bytes(), label="source plan"),
        "plan_bytes": (destination / "plan.json").stat().st_size,
        "plan_sha256": _sha256((destination / "plan.json").read_bytes()),
        "inventory_bytes": total_bytes,
    }


def _license_id(scope: Mapping[str, Any]) -> str:
    path_scope = scope.get("path_scope")
    spdx = path_scope.get("spdx") if isinstance(path_scope, Mapping) else None
    if not isinstance(spdx, list) or len(spdx) != 1 or not isinstance(spdx[0], str):
        raise BundleExportError("source file does not have one pinned SPDX license")
    labels = {
        "mit": "MIT",
        "apache-2.0": "Apache-2.0",
        "bsd-2-clause": "BSD-2-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "isc": "ISC",
    }
    try:
        return labels[spdx[0].casefold()]
    except KeyError:
        raise BundleExportError("source file license is outside the pilot allowlist") from None


def _artifact_reference(
    source_root: Path,
    source_package_root: str,
    metadata: Mapping[str, Any],
    stem: str,
) -> dict[str, Any]:
    relative = _safe_package_path(metadata.get(stem + "_path"), label=stem)
    payload = source_root / relative
    try:
        if payload.is_symlink() or not payload.is_file():
            raise BundleExportError(f"{stem} artifact is not a regular file")
        content = payload.read_bytes()
    except OSError:
        raise BundleExportError(f"{stem} artifact is unavailable") from None
    expected_hash = metadata.get(stem + "_sha256")
    expected_bytes = metadata.get(stem + "_bytes")
    if (
        type(expected_bytes) is not int
        or expected_bytes != len(content)
        or expected_hash != _sha256(content)
    ):
        raise BundleExportError(f"{stem} artifact does not match its authoring metadata")
    return {
        "path": f"{source_package_root}/{relative.as_posix()}",
        "sha256": expected_hash,
        "bytes": expected_bytes,
    }


def _build_candidate_row(
    *,
    input_row: Mapping[str, Any],
    answer: Mapping[str, Any],
    provenance: Mapping[str, Any],
    fixture_binding: Mapping[str, Any],
    oracle: Mapping[str, Any],
    parent_tree: Mapping[str, Any],
    source_root: Path,
    source_package_root: str,
    source_manifest_sha256: str,
    tokenizer: Any,
) -> dict[str, Any]:
    candidate_id = str(input_row["candidate_id"])
    raw_metadata = provenance.get("authoring_metadata")
    if not isinstance(raw_metadata, Mapping):
        raise BundleExportError("candidate authoring metadata is incomplete")
    scope_ref = _artifact_reference(
        source_root, source_package_root, raw_metadata, "license_scope_artifact"
    )
    scope = _json_unique(
        (source_root / str(raw_metadata["license_scope_artifact_path"])).read_bytes(),
        label="source file license scope",
    )
    if not isinstance(scope, Mapping):
        raise BundleExportError("source file license scope is not an object")
    license_id = _license_id(scope)
    source_ref = _artifact_reference(
        source_root, source_package_root, raw_metadata, "source_artifact"
    )
    license_ref = _artifact_reference(
        source_root, source_package_root, raw_metadata, "root_license_artifact"
    )
    path_scope = scope.get("path_scope")
    root_license = scope.get("root_license")
    if not isinstance(path_scope, Mapping) or not isinstance(root_license, Mapping):
        raise BundleExportError("source license scope lacks its exact root-license identity")

    state = EditState.from_mapping(input_row["state"])
    action = EditAction(**answer["action"])
    after_source = apply_action(state, action)
    transform = raw_metadata.get("transform")
    if not isinstance(transform, Mapping) or transform.get("kind") != TYPED_RETURN_PREFIX_TRANSFORM:
        raise BundleExportError("source candidate is not an exact typed return prefix")
    selected_sha256 = _sha256(state.source.encode("utf-8"))
    metadata: dict[str, Any] = {
        "source_repo": parent_tree["repository"],
        "source_revision": parent_tree["parent_commit"],
        "origin_transition_revision": raw_metadata["source_revision"],
        "parent_commit": parent_tree["parent_commit"],
        "source_tree_sha": parent_tree["source_tree_sha"],
        "source_path": parent_tree["source_path"],
        "source_sha256": parent_tree["source_sha256"],
        "source_artifact_path": source_ref["path"],
        "source_artifact_sha256": source_ref["sha256"],
        "source_artifact_bytes": source_ref["bytes"],
        "source_license": license_id,
        "path_license": license_id,
        "path_license_spdx": path_scope["spdx"],
        "path_license_scope": path_scope["scope"],
        "path_license_sha256": path_scope["sha256"],
        "path_license_git_blob_sha": path_scope["git_blob_sha"],
        "license_path": path_scope["license_path"],
        "license_scope_candidate_id": raw_metadata["license_scope_candidate_id"],
        "license_scope_status": scope["status"],
        "path_license_artifact_path": license_ref["path"],
        "path_license_artifact_sha256": license_ref["sha256"],
        "path_license_artifact_bytes": license_ref["bytes"],
        "license_scope_artifact_path": scope_ref["path"],
        "license_scope_sha256": scope_ref["sha256"],
        "license_scope_artifact_bytes": scope_ref["bytes"],
        "selected_source_sha256": selected_sha256,
        "source_group_id": provenance["source_group_id"],
        "session_or_commit": parent_tree["parent_commit"],
        "task_family_id": raw_metadata["task_family_id"],
        "template_id": raw_metadata["template_id"],
        "history_origin": raw_metadata["history_origin"],
        "visible_intent_basis": raw_metadata["visible_intent_basis"],
        "visible_request_location": raw_metadata["visible_request_location"],
        "license_spdx": raw_metadata["license_spdx"],
        "transform": dict(transform),
        "transform_sha256": raw_metadata["transform_sha256"],
    }
    state_context = __import__(
        "tinycomplete.one_line.context", fromlist=["serialize_state_bounded"]
    ).serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
    if (
        input_row.get("prompt") != state_context.text
        or input_row.get("context_sha256") != _sha256(state_context.text.encode("utf-8"))
    ):
        raise BundleExportError("answer-free model context does not rebuild from the exact state")
    row: dict[str, Any] = {
        "id": candidate_id,
        "candidate_id": candidate_id,
        "seed_id": input_row["seed_id"],
        "split": answer["split"],
        "source_type": REVIEWED_PUBLIC_HISTORY_SOURCE,
        "objective_source_type": provenance["source_type"],
        "source_group_id": provenance["source_group_id"],
        "session_or_commit": parent_tree["parent_commit"],
        "task_family_id": raw_metadata["task_family_id"],
        "template_id": raw_metadata["template_id"],
        # The role runner canonicalizes EditState to a dataclass mapping, where
        # tuple-valued history/relevant fields compare equal only before JSON
        # converts them to arrays. Keep that canonical in-memory form here;
        # canonical JSON hashes and the staged JSONL remain unchanged.
        "state": asdict(state),
        "prompt": input_row["prompt"],
        "context_sha256": input_row["context_sha256"],
        "context_policy": CONTEXT_POLICY_VERSION,
        "action": answer["action"],
        "after_source": after_source,
        "source_license": license_id,
        "source_license_sha256": license_ref["sha256"],
        "source_materialization_provenance_sha256": _sha256(
            _canonical_bytes(provenance)
        ),
        "history_order": "synthetic_fixed_before_provider",
        "history_sha256": answer["history_before_sha256"],
        "human_chronology_observed": False,
        "accepted_training": True,
        "inferability_reviewed": True,
        "objective_verified": True,
        "wrong_action_controls_rejected": True,
        "history_leakage_check": True,
        "quality_evidence": False,
        "authoring_metadata": metadata,
        "objective_fixture_binding": {
            "schema": "python-prefix-objective-fixture-binding-v1",
            "path": oracle["fixture_artifact_path"],
            "sha256": oracle["fixture_sha256"],
            "bytes": oracle["fixture_artifact_bytes"],
        },
        "objective_evidence_binding": {
            "schema": "python-prefix-objective-candidate-binding-v1",
            "source_manifest_sha256": source_manifest_sha256,
            "source_input_sha256": _sha256(_canonical_bytes(input_row)),
            "answer_sha256": _sha256(_canonical_bytes(answer)),
            "source_provenance_sha256": _sha256(_canonical_bytes(provenance)),
            "fixture_binding_sha256": _sha256(_canonical_bytes(fixture_binding)),
            "oracle_result_sha256": _sha256(_canonical_bytes(oracle)),
            "oracle_diagnostic_sha256": {},
            "fixture_artifact_path": oracle["fixture_artifact_path"],
            "fixture_artifact_sha256": oracle["fixture_sha256"],
            "fixture_artifact_bytes": oracle["fixture_artifact_bytes"],
            "evaluator_sha256": oracle["evaluator_sha256"],
            "runtime_sha256": oracle["runtime_sha256"],
        },
    }
    # The validator recomputes this small diagnostic inventory from its pinned source bundle.
    row["objective_evidence_binding"]["oracle_diagnostic_sha256"] = {
        name: None for name in ("gold", "before", "behavior_breaking")
    }
    return row


def _load_tokenizer() -> Any:
    runner_ns = runpy.run_path(
        str(Path(__file__).resolve().with_name("run_fixed_state_role_pilot.py")),
        run_name="public_prefix_role_export",
    )
    loader = runner_ns.get("_tokenizer")
    if not callable(loader):
        raise BundleExportError("pinned fixed-state Q25 tokenizer loader is unavailable")
    return loader()


def export_bundle(
    *,
    oracle_package: Path,
    role_run: Path,
    release_path: Path,
    release_sha256: str,
    usage_ledger: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.expanduser().resolve(strict=False)
    allowed_root = Path("/mnt/ssd/tabcomplete-product-r2").resolve(strict=True)
    if (
        output == allowed_root
        or not output.is_relative_to(allowed_root)
        or output.exists()
        or output.is_symlink()
    ):
        raise BundleExportError("output must be a new private SSD package directory")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        if stat.S_IMODE(output.parent.stat().st_mode) & 0o077:
            raise BundleExportError("output parent is not owner-only")
    except OSError:
        raise BundleExportError("output parent is unavailable") from None
    if release_path.is_symlink() or _sha256(release_path.read_bytes()) != release_sha256:
        raise BundleExportError("root release does not match the pinned digest")
    role_plan = _json_unique((role_run / "plan.json").read_bytes(), label="role plan")
    role_summary = _json_unique((role_run / "summary.json").read_bytes(), label="role summary")
    role_cases = role_summary.get("cases") if isinstance(role_summary, Mapping) else None
    if (
        not isinstance(role_plan, Mapping)
        or not isinstance(role_summary, Mapping)
        or not isinstance(role_cases, list)
        or role_summary.get("phase_status") != "complete"
        or len(role_cases) != role_summary.get("case_count")
        or any(
            not isinstance(case, Mapping)
            or case.get("role_evidence_accepted") is not True
            or case.get("author_solver_action_agreement") is not True
            or case.get("reviewer_retained") is not True
            for case in role_cases
        )
        or role_summary.get("training_accepted") is not False
        or role_summary.get("training_started") is not False
        or role_plan.get("plan_sha256") != role_summary.get("plan_sha256")
        or not isinstance(role_plan.get("plan_sha256"), str)
    ):
        raise BundleExportError("fixed-state role phase is not complete and accepted")

    source_manifest_bytes = (oracle_package / "manifest.json").read_bytes()
    source_manifest = _json_unique(source_manifest_bytes, label="source oracle manifest")
    source_files = source_manifest.get("files") if isinstance(source_manifest, Mapping) else None
    if not isinstance(source_files, Mapping):
        raise BundleExportError("source oracle package has no frozen inventory")
    candidate_rows = {
        filename: _index(_jsonl(oracle_package / filename, label=filename), label=filename)
        for filename in (
            "source_only_inputs.jsonl",
            "answers_private.jsonl",
            "provenance_private.jsonl",
            "fixture_bindings_private.jsonl",
            "oracle_results.jsonl",
            "parent_trees.jsonl",
        )
    }
    diagnostics = _jsonl(oracle_package / "oracle_diagnostics.jsonl", label="oracle diagnostics")
    diagnostics_by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for diagnostic in diagnostics:
        key = (str(diagnostic.get("candidate_id", "")), str(diagnostic.get("variant", "")))
        if not all(key) or key in diagnostics_by_candidate:
            raise BundleExportError("source oracle diagnostics have duplicate identities")
        diagnostics_by_candidate[key] = diagnostic

    tokenizer = _load_tokenizer()
    temporary_parent = output.parent
    temporary = Path(tempfile.mkdtemp(prefix=".public-prefix-bundle-", dir=temporary_parent))
    os.chmod(temporary, 0o700)
    try:
        source_root = temporary / SOURCE_PACKAGE_DIR
        source = _copy_frozen_source_package(oracle_package, source_root)
        source_package_manifest_sha256 = source["manifest_sha256"]
        selected_ids = sorted(candidate_rows["source_only_inputs.jsonl"])
        if selected_ids != sorted(candidate_rows["answers_private.jsonl"]):
            raise BundleExportError("source model inputs and private answers disagree")
        if len(selected_ids) != source["manifest"].get("candidate_count"):
            raise BundleExportError("source oracle candidate count differs from its input rows")

        accepted: list[dict[str, Any]] = []
        for candidate_id in selected_ids:
            values = {
                name: rows[candidate_id]
                for name, rows in candidate_rows.items()
            }
            row = _build_candidate_row(
                input_row=values["source_only_inputs.jsonl"],
                answer=values["answers_private.jsonl"],
                provenance=values["provenance_private.jsonl"],
                fixture_binding=values["fixture_bindings_private.jsonl"],
                oracle=values["oracle_results.jsonl"],
                parent_tree=values["parent_trees.jsonl"],
                source_root=oracle_package,
                source_package_root=SOURCE_PACKAGE_DIR,
                source_manifest_sha256=source_package_manifest_sha256,
                tokenizer=tokenizer,
            )
            evidence_binding = row["objective_evidence_binding"]
            evidence_binding["oracle_diagnostic_sha256"] = {
                name: _sha256(_canonical_bytes(diagnostics_by_candidate[(candidate_id, name)]))
                for name in ("gold", "before", "behavior_breaking")
            }
            role_ref = export_fixed_state_role_execution_proof(
                row,
                run_dir=role_run,
                release_path=release_path,
                release_sha256=release_sha256,
                ledger_path=usage_ledger,
                package_root=temporary,
                tokenizer=tokenizer,
            )
            row["role_execution_proof_ref"] = role_ref
            row["role_evidence_artifact_path"] = (
                str(role_ref["root"]) + "/" + str(role_ref["role_evidence_path"])
            )
            row["role_evidence_sha256"] = role_ref["role_evidence_sha256"]
            row["role_evidence_bytes"] = role_ref["role_evidence_bytes"]
            accepted.append(row)

        rows_payload = b"".join(_canonical_bytes(row) + b"\n" for row in accepted)
        _write_private(temporary / "accepted_rows.jsonl", rows_payload)
        candidate_split_counts = {"train": len(accepted)}
        manifest: dict[str, Any] = {
            "schema": PUBLIC_PREFIX_ROLE_BUNDLE_SCHEMA,
            "data_policy_schema": LICENSE_MIXED_HISTORY.data_schema,
            "source_package_root": SOURCE_PACKAGE_DIR,
            "source_package_manifest_path": f"{SOURCE_PACKAGE_DIR}/manifest.json",
            "source_package_manifest_sha256": source["manifest_sha256"],
            "source_package_manifest_bytes": source["manifest_bytes"],
            "source_package_plan_path": f"{SOURCE_PACKAGE_DIR}/plan.json",
            "source_package_plan_sha256": source["plan_sha256"],
            "source_package_plan_bytes": source["plan_bytes"],
            "accepted_rows_path": "accepted_rows.jsonl",
            "accepted_rows_sha256": _sha256(rows_payload),
            "accepted_rows_bytes": len(rows_payload),
            "candidate_count": len(accepted),
            "candidate_split_counts": candidate_split_counts,
            "train_count": len(accepted),
            "development_count": 0,
            "source_group_count": len({row["source_group_id"] for row in accepted}),
            "role_plan_sha256": role_plan["plan_sha256"],
            "root_release_sha256": release_sha256,
            "accepted_training_count": len(accepted),
            "training_ready": False,
            "quality_evidence": False,
        }
        _write_private(temporary / "manifest.json", _canonical_bytes(manifest) + b"\n")
        audit = validate_public_prefix_role_bundle(
            manifest, package_root=temporary, tokenizer=tokenizer
        )
        if audit["candidate_count"] != len(accepted) or audit["training_ready"] is not False:
            raise BundleExportError("candidate bundle failed its own portable policy check")
        os.replace(temporary, output)
        os.chmod(output, 0o700)
        return {
            "manifest": manifest,
            "audit": audit,
            "output": str(output),
            "manifest_sha256": _sha256((output / "manifest.json").read_bytes()),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-package", type=Path, default=DEFAULT_ORACLE_PACKAGE)
    parser.add_argument("--role-run", type=Path, default=DEFAULT_ROLE_RUN)
    parser.add_argument("--release", type=Path, default=DEFAULT_ROLE_RUN / "release.json")
    parser.add_argument(
        "--release-sha256",
        required=True,
        help="SHA-256 of the already-frozen local role-phase release file",
    )
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_USAGE_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = export_bundle(
        oracle_package=args.oracle_package,
        role_run=args.role_run,
        release_path=args.release,
        release_sha256=args.release_sha256,
        usage_ledger=args.usage_ledger,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": "candidate_bundle_verified",
                "output": result["output"],
                "manifest_sha256": result["manifest_sha256"],
                "candidate_count": result["audit"]["candidate_count"],
                "accepted_training_count": result["audit"]["accepted_training_count"],
                "training_ready": result["audit"]["training_ready"],
                "source_group_count": result["audit"]["source_group_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
