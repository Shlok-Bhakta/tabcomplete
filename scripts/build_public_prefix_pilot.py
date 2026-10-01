#!/usr/bin/env python3
"""Package exact public parent-line completions for a later balanced pilot.

This CPU-only builder verifies a frozen 266-row public prefix pool, reopens its
parent-source and path-license artifacts, and produces a portable R-only
candidate bank. It does not mark the bank training-ready: a separate N/I/D
functional corpus and a combined split/action audit are required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION
from tinycomplete.one_line.contract import EditAction, EditState, encode_action
from tinycomplete.one_line.public_prefix_pilot import (
    PUBLIC_PREFIX_HISTORY_ORDER,
    PUBLIC_PREFIX_PILOT_SCHEMA,
    PUBLIC_PREFIX_SOURCE_TYPE,
    canonical_bytes,
    canonical_sha256,
    public_prefix_template_key,
    sha256_bytes,
    validate_public_prefix_row,
    validate_public_prefix_splits,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ROOT = Path("/mnt/ssd/tabcomplete-product-r2/commitpackft")
DEFAULT_INPUT = DEFAULT_PACKAGE_ROOT / "verified-parent-prefix-materialized-v3"
DEFAULT_EXPANSION = DEFAULT_PACKAGE_ROOT / "source-verification-expansion-v3"
DEFAULT_OUTPUT = Path("/mnt/ssd/tabcomplete-product-r2/public-prefix-typing-pilot-v1")
EXPECTED_MATERIALIZATION_SCHEMA = "verified-parent-prefix-materialization-result-v3"
EXPECTED_EXPANSION_SCHEMA = "commit-sequence-parent-source-expansion-result-v1"
EXPECTED_EXPANSION_PLAN_IDENTITY = (
    "d823b2ab136179ebcac49033e73facca9697cb535fe318724fadf651b4faa31c"
)
EXPECTED_ROW_COUNT = 266
MAX_PACKAGE_BYTES = 512 * 1024**2
MAX_ARTIFACT_BYTES = 512 * 1024**2
_LICENSE_NAMES = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
}


def _strict_json(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(label + " is invalid JSON") from None


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
        rows = [_strict_json(line, label) for line in raw.splitlines() if line]
    except OSError:
        raise ValueError(label + " is unavailable") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(label + " rows must be objects")
    return rows


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def _check_manifest_file(directory: Path, manifest: dict[str, Any], name: str) -> Path:
    files = manifest.get("files")
    item = files.get(name) if isinstance(files, dict) else None
    if not isinstance(item, dict):
        raise ValueError("materialized prefix manifest lacks " + name)
    relative = item.get("path")
    expected_bytes = item.get("bytes")
    expected_sha = item.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or type(expected_bytes) is not int
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
    ):
        raise ValueError("materialized prefix file identity is invalid: " + name)
    path = (directory / relative).resolve()
    if directory.resolve() not in path.parents or not path.is_file():
        raise ValueError("materialized prefix file escapes or is missing: " + name)
    identity = _file_identity(path)
    if identity != {"bytes": expected_bytes, "sha256": expected_sha}:
        raise ValueError("materialized prefix file hash/size mismatch: " + name)
    return path


def _manifest_child(root: Path, relative: Any, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError(label + " path is invalid")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(label + " path escapes or is missing")
    return path


def _load_pool(
    materialized_dir: Path, package_root: Path, expansion_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = materialized_dir / "manifest.json"
    materialized_manifest = _strict_json(manifest_path.read_bytes(), "materialization manifest")
    if (
        not isinstance(materialized_manifest, dict)
        or materialized_manifest.get("schema") != EXPECTED_MATERIALIZATION_SCHEMA
        or materialized_manifest.get("status") != "complete"
        or materialized_manifest.get("selected_rows") != EXPECTED_ROW_COUNT
        or materialized_manifest.get("split_counts") != {"development": 49, "train": 217}
        or materialized_manifest.get("accepted_training") != 0
        or materialized_manifest.get("training_ready") is not False
        or materialized_manifest.get("quality_evidence") is not False
        or materialized_manifest.get("human_chronology_observed") is not False
        or materialized_manifest.get("child_source_or_diff_bytes_read") != 0
        or materialized_manifest.get("model_runs") != 0
        or materialized_manifest.get("provider_calls") != 0
    ):
        raise ValueError("frozen R-only public-prefix candidate pool is not the expected v3 result")
    review_path = _check_manifest_file(
        materialized_dir, materialized_manifest, "review_inputs.jsonl"
    )
    answer_path = _check_manifest_file(
        materialized_dir, materialized_manifest, "answer_key_private.jsonl"
    )
    provenance_path = _check_manifest_file(
        materialized_dir, materialized_manifest, "provenance_private.jsonl"
    )
    review_rows = _read_jsonl(review_path, "public-prefix review inputs")
    answer_rows = _read_jsonl(answer_path, "public-prefix private answers")
    provenance_rows = _read_jsonl(provenance_path, "public-prefix provenance")
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in (
        ("review", review_rows),
        ("answer", answer_rows),
        ("provenance", provenance_rows),
    ):
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            identifier = row.get("candidate_id")
            if not isinstance(identifier, str) or not identifier or identifier in mapping:
                raise ValueError("public-prefix " + name + " IDs are missing or duplicated")
            mapping[identifier] = row
        records[name] = mapping
    ids = set(records["review"])
    if ids != set(records["answer"]) or ids != set(records["provenance"]):
        raise ValueError("public-prefix inputs, answers, and provenance do not align")
    if len(ids) != EXPECTED_ROW_COUNT:
        raise ValueError("public-prefix row count differs from its frozen manifest")

    expansion_manifest_path = expansion_dir / "manifest.json"
    expansion_manifest = _strict_json(
        expansion_manifest_path.read_bytes(), "source expansion manifest"
    )
    if (
        not isinstance(expansion_manifest, dict)
        or expansion_manifest.get("schema") != EXPECTED_EXPANSION_SCHEMA
        or expansion_manifest.get("status") != "complete"
        or expansion_manifest.get("plan_identity_sha256") != EXPECTED_EXPANSION_PLAN_IDENTITY
        or expansion_manifest.get("processed_rows") != 581
        or expansion_manifest.get("reserved_evaluation_access") is not False
        or expansion_manifest.get("child_source_or_diff_bytes_requested") != 0
        or expansion_manifest.get("provider_calls") != 0
        or expansion_manifest.get("model_runs") != 0
        or expansion_manifest.get("accepted_training") != 0
        or expansion_manifest.get("quality_evidence") is not False
    ):
        raise ValueError("verified-parent source expansion is not the frozen v3 result")
    expansion_candidates = _manifest_child(
        expansion_dir, expansion_manifest.get("candidate_results_path"), "source expansion results"
    )
    expansion_scopes = _manifest_child(
        expansion_dir,
        expansion_manifest.get("license_scope_results_path"),
        "source expansion path scopes",
    )
    for path, bytes_field, sha_field in (
        (expansion_candidates, "candidate_results_bytes", "candidate_results_sha256"),
        (expansion_scopes, "license_scope_results_bytes", "license_scope_results_sha256"),
    ):
        if not path.is_file() or _file_identity(path) != {
            "bytes": expansion_manifest.get(bytes_field),
            "sha256": expansion_manifest.get(sha_field),
        }:
            raise ValueError("parent-source expansion artifact hash/size mismatch")
    source_records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(expansion_candidates, "verified parent-source rows"):
        if row.get("status") != "parent_source_and_path_scope_verified":
            continue
        identifier = row.get("candidate_id")
        if not isinstance(identifier, str) or not identifier or identifier in source_records:
            raise ValueError("verified parent-source IDs are missing or duplicated")
        source_records[identifier] = row
    scope_rows = _read_jsonl(expansion_scopes, "verified parent path-scope rows")
    scope_records: dict[str, dict[str, Any]] = {}
    for row in scope_rows:
        identifier = row.get("candidate_id")
        if not isinstance(identifier, str) or not identifier or identifier in scope_records:
            raise ValueError("parent path-scope proof IDs are missing or duplicated")
        scope_records[identifier] = row
    if (
        len(source_records) != 423
        or len(scope_records) > EXPECTED_ROW_COUNT * 2
        or not set(source_records).issubset(scope_records)
    ):
        raise ValueError(
            "parent-source expansion row counts differ from the frozen proof inventory"
        )

    package_root = package_root.resolve()
    converted: list[dict[str, Any]] = []
    referenced_artifacts: dict[str, tuple[Path, int]] = {}
    for identifier in sorted(ids):
        review = records["review"][identifier]
        answer = records["answer"][identifier]
        provenance = records["provenance"][identifier]
        candidate_id = provenance.get("license_scope_candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in source_records:
            raise ValueError("public-prefix row lacks its verified parent-source record")
        source_record = source_records[candidate_id]
        scope_record = scope_records.get(candidate_id)
        if not isinstance(scope_record, dict):
            raise ValueError("public-prefix row lacks a path-scope result")
        provenance_scope_path = Path(str(provenance.get("license_scope_artifact_path", "")))
        scope_record_path = Path(str(scope_record.get("scope_path", "")))
        if (
            source_record.get("repository") != provenance.get("source_repo")
            or source_record.get("parent_commit") != provenance.get("parent_revision")
            or source_record.get("source_path") != provenance.get("source_path")
            or source_record.get("source_sha256") != provenance.get("source_artifact_sha256")
            or source_record.get("source_bytes") != provenance.get("source_artifact_bytes")
            or source_record.get("source_revision") != provenance.get("source_revision")
            or source_record.get("split") != provenance.get("split")
            or source_record.get("source_group_id") != provenance.get("source_group_id")
            or source_record.get("license_scope_sha256")
            != provenance.get("license_scope_artifact_sha256")
            or source_record.get("root_license_sha256")
            != provenance.get("root_license_artifact_sha256")
            or scope_record.get("scope_status") != "parent_source_and_path_scope_verified"
            or scope_record.get("scope_sha256") != provenance.get("license_scope_artifact_sha256")
            or source_record.get("license_scope_path") != scope_record.get("scope_path")
            or not scope_record_path.parts
            or provenance_scope_path.parts[-len(scope_record_path.parts) :]
            != scope_record_path.parts
            or scope_record.get("split") != provenance.get("split")
        ):
            raise ValueError("public-prefix source provenance differs from verified expansion")
        state = EditState.from_mapping(review["state"])
        action = EditAction(**answer["action"])
        if answer.get("candidate_id") != identifier or provenance.get("candidate_id") != identifier:
            raise ValueError("public-prefix row IDs differ across source artifacts")
        licenses = provenance.get("license_spdx")
        if not isinstance(licenses, list) or len(licenses) != 1:
            raise ValueError("public-prefix source needs one unambiguous path license")
        license_name = _LICENSE_NAMES.get(str(licenses[0]).casefold())
        if license_name is None:
            raise ValueError("public-prefix source license is not supported")

        artifacts: dict[str, dict[str, Any]] = {}
        for prefix in (
            "source_artifact",
            "path_license_artifact",
            "license_scope_artifact",
            "root_license_artifact",
        ):
            original = provenance.get(prefix + "_path")
            expected_sha = provenance.get(prefix + "_sha256")
            expected_bytes = provenance.get(prefix + "_bytes")
            if (
                not isinstance(original, str)
                or Path(original).is_absolute()
                or ".." in Path(original).parts
                or not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or type(expected_bytes) is not int
                or expected_bytes < 0
            ):
                raise ValueError("public-prefix source artifact identity is invalid")
            source_path = (package_root / original).resolve()
            if package_root not in source_path.parents or not source_path.is_file():
                raise ValueError("public-prefix source artifact is missing")
            payload = source_path.read_bytes()
            if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha:
                raise ValueError("public-prefix source artifact hash/size mismatch")
            artifacts[prefix] = {
                "original_path": original,
                "source_path": source_path,
                "bytes": len(payload),
                "sha256": expected_sha,
            }
            referenced_artifacts[expected_sha] = (source_path, len(payload))

        scope_path = artifacts["license_scope_artifact"]["source_path"]
        scope_payload = scope_path.read_bytes()
        scope = _strict_json(scope_payload, "parent path-license scope proof")
        path_scope = scope.get("path_scope") if isinstance(scope, dict) else None
        root_license = scope.get("root_license") if isinstance(scope, dict) else None
        if not isinstance(path_scope, dict) or not isinstance(root_license, dict):
            raise ValueError("public-prefix scope proof has no file/root license records")
        if (
            scope.get("schema") != "exact-parent-license-scope-v2"
            or scope.get("status") != "verified_path_scope"
            or scope.get("repository") != provenance.get("source_repo")
            or scope.get("parent_commit") != provenance.get("parent_revision")
            or scope.get("source_path") != provenance.get("source_path")
            or scope.get("source_sha256") != provenance.get("source_artifact_sha256")
            or path_scope.get("scope") != "root"
            or path_scope.get("spdx") != [str(licenses[0]).casefold()]
            or path_scope.get("spdx_ambiguous") is not False
            or path_scope.get("spdx_expression_count") != 0
            or scope.get("source_header_spdx_ambiguous") is not False
            or scope.get("source_header_spdx_expression_count") != 0
            or scope.get("additional_license_references") != []
            or scope.get("reuse_dep5_references") != []
            or path_scope.get("sha256") != provenance.get("root_license_artifact_sha256")
            or root_license.get("sha256") != provenance.get("root_license_artifact_sha256")
            or root_license.get("path") != path_scope.get("license_path")
            or root_license.get("git_blob_sha") != path_scope.get("git_blob_sha")
        ):
            raise ValueError("public-prefix parent file/path/root license proof is inconsistent")

        source_payload = artifacts["source_artifact"]["source_path"].read_bytes()
        history_sha = canonical_sha256(
            [
                {"row": edit.row, "old_text": edit.old_text, "new_text": edit.new_text}
                for edit in state.history
            ]
        )
        transform = {
            "kind": PUBLIC_PREFIX_HISTORY_ORDER,
            "target_physical_row": state.target_row,
            "cursor_byte_column": state.cursor_col,
            "parent_source_sha256": provenance["source_artifact_sha256"],
            "history_sha256": history_sha,
            "history_origin": "synthetic_editor_typing",
            "exact_prefix": True,
            "no_suffix_in_input": True,
        }
        authoring_metadata = {
            "source_repo": provenance["source_repo"],
            "source_revision": provenance["parent_revision"],
            "origin_transition_revision": provenance["source_revision"],
            "source_path": provenance["source_path"],
            "source_sha256": provenance["source_artifact_sha256"],
            "source_license": license_name,
            "path_license": license_name,
            "license_scope_status": "verified_path_scope",
            "license_path": path_scope["license_path"],
            "path_license_sha256": path_scope["sha256"],
            "path_license_git_blob_sha": path_scope["git_blob_sha"],
            "root_license_sha256": provenance["root_license_artifact_sha256"],
            "source_artifact_path": "artifacts/sha256/" + artifacts["source_artifact"]["sha256"],
            "source_artifact_sha256": artifacts["source_artifact"]["sha256"],
            "source_artifact_bytes": artifacts["source_artifact"]["bytes"],
            "path_license_artifact_path": "artifacts/sha256/"
            + artifacts["path_license_artifact"]["sha256"],
            "path_license_artifact_sha256": artifacts["path_license_artifact"]["sha256"],
            "path_license_artifact_bytes": artifacts["path_license_artifact"]["bytes"],
            "license_scope_artifact_path": "artifacts/sha256/"
            + artifacts["license_scope_artifact"]["sha256"],
            "license_scope_artifact_sha256": artifacts["license_scope_artifact"]["sha256"],
            "license_scope_artifact_bytes": artifacts["license_scope_artifact"]["bytes"],
            "root_license_artifact_path": "artifacts/sha256/"
            + artifacts["root_license_artifact"]["sha256"],
            "root_license_artifact_sha256": artifacts["root_license_artifact"]["sha256"],
            "root_license_artifact_bytes": artifacts["root_license_artifact"]["bytes"],
            "transform": transform,
            "transform_sha256": canonical_sha256(transform),
        }
        row = {
            "id": identifier,
            "candidate_id": identifier,
            "seed_id": identifier,
            "split": provenance["split"],
            "source_type": PUBLIC_PREFIX_SOURCE_TYPE,
            "source_license": license_name,
            "human_chronology_observed": False,
            "history_order": PUBLIC_PREFIX_HISTORY_ORDER,
            "context_policy": CONTEXT_POLICY_VERSION,
            "state": {
                "file_id": state.file_id,
                "filetype": state.filetype,
                "source": state.source,
                "target_row": state.target_row,
                "cursor_col": state.cursor_col,
                "history": [
                    {
                        "row": edit.row,
                        "old_text": edit.old_text,
                        "new_text": edit.new_text,
                    }
                    for edit in state.history
                ],
                "relevant": list(state.relevant),
            },
            "prompt": review["prompt"],
            "context_sha256": review["context_sha256"],
            "history_sha256": history_sha,
            "history_before_sha256": answer["history_before_sha256"],
            "action": {"kind": action.kind, "text": action.text},
            "action_wire": encode_action(action),
            "after_source": answer["after_source"],
            "after_source_sha256": answer["after_source_sha256"],
            "target_source_line_sha256": answer["target_source_line_sha256"],
            "source_repo": provenance["source_repo"],
            "source_revision": provenance["parent_revision"],
            "origin_transition_revision": provenance["source_revision"],
            "source_path": provenance["source_path"],
            "source_sha256": provenance["source_artifact_sha256"],
            "source_group_id": provenance["source_group_id"],
            "session_or_commit": provenance["parent_revision"],
            "task_family_id": provenance["task_family_id"],
            "template_id": provenance["template_id"],
            "near_duplicate_sha256": "",
            "authoring_metadata": authoring_metadata,
            "accepted_training": False,
            "prefix_supervision_verified": True,
            "objective_verified": False,
            "inferability_reviewed": False,
            "quality_evidence": False,
        }
        row["near_duplicate_sha256"] = str(provenance.get("near_duplicate_key", ""))
        if (
            not isinstance(provenance.get("template_id"), str)
            or provenance.get("near_duplicate_key") != provenance["template_id"].rsplit("/", 1)[-1]
            or provenance.get("near_duplicate_key") != public_prefix_template_key(state)
        ):
            raise ValueError("public-prefix materialization template identity is inconsistent")
        # The split audit hashes the exact normalized model-visible state.
        from tinycomplete.one_line.public_prefix_pilot import validate_public_prefix_transition

        validate_public_prefix_transition(row, parent_source=source_payload)
        if row["split"] not in {"train", "development"}:
            raise ValueError("public-prefix pool includes an unapproved split")
        converted.append(row)

    split_audit = validate_public_prefix_splits(converted)
    if split_audit["splits"] != {"development": 49, "train": 217}:
        raise ValueError("public-prefix source group split counts changed")
    input_tokens = materialized_manifest.get("input_and_history_token_counts")
    response_tokens = materialized_manifest.get("supervised_response_and_eos_token_counts")
    if (
        not isinstance(input_tokens, dict)
        or input_tokens.get("train_input_tokens") != 81817
        or input_tokens.get("development_input_tokens") != 19009
        or not isinstance(response_tokens, dict)
        or response_tokens.get("train_response_eos_tokens") != 2152
        or response_tokens.get("development_response_eos_tokens") != 585
    ):
        raise ValueError("frozen public-prefix token exposure counts changed")
    metadata = {
        "materialization_manifest": _file_identity(manifest_path),
        "materialization_plan_sha256": materialized_manifest["plan_sha256"],
        "materialized_inputs": {
            name: materialized_manifest["files"][name]
            for name in (
                "review_inputs.jsonl",
                "answer_key_private.jsonl",
                "provenance_private.jsonl",
            )
        },
        "source_expansion_manifest": _file_identity(expansion_manifest_path),
        "source_expansion_candidates": _file_identity(expansion_candidates),
        "source_expansion_path_scopes": _file_identity(expansion_scopes),
        "source_expansion_plan_identity_sha256": EXPECTED_EXPANSION_PLAN_IDENTITY,
        "token_exposure": {
            "input": input_tokens,
            "response_including_eos": response_tokens,
            "tokenizer_and_context_policy": "frozen v3 materialization identity",
        },
        "source_groups_by_split": {
            split: len({str(row["source_group_id"]) for row in converted if row["split"] == split})
            for split in ("train", "development")
        },
        "task_families_by_split": {
            split: len({str(row["task_family_id"]) for row in converted if row["split"] == split})
            for split in ("train", "development")
        },
        "license_counts": dict(
            sorted(Counter(str(row["source_license"]) for row in converted).items())
        ),
        "action_counts": dict(
            sorted(Counter(str(row["action_wire"][0]) for row in converted).items())
        ),
        "rows": len(converted),
        "split_counts": split_audit["splits"],
        "split_audit": split_audit,
        "artifact_sources": referenced_artifacts,
    }
    return converted, metadata


def build_candidate_bank(
    *,
    materialized_dir: Path,
    package_root: Path,
    expansion_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("public-prefix output directory already exists; preserve it")
    requested_output = output_dir.resolve()
    for protected in (materialized_dir.resolve(), package_root.resolve(), expansion_dir.resolve()):
        if (
            requested_output == protected
            or requested_output in protected.parents
            or protected in requested_output.parents
        ):
            raise ValueError("public-prefix output overlaps a frozen source tree")
    rows, source_metadata = _load_pool(materialized_dir, package_root, expansion_dir)
    source_artifacts = source_metadata.pop("artifact_sources")
    total_bytes = sum(size for _, size in source_artifacts.values())
    if total_bytes > MAX_ARTIFACT_BYTES:
        raise ValueError("public-prefix source-proof artifacts exceed the 512 MiB cap")
    train = [row for row in rows if row["split"] == "train"]
    development = [row for row in rows if row["split"] == "development"]
    train_payload = b"".join(canonical_bytes(row) + b"\n" for row in train)
    development_payload = b"".join(canonical_bytes(row) + b"\n" for row in development)
    output_bytes = len(train_payload) + len(development_payload) + total_bytes
    if output_bytes > MAX_PACKAGE_BYTES:
        raise ValueError("public-prefix candidate bank exceeds the 512 MiB cap")

    plan = {
        "schema": "one-line-public-prefix-typing-plan-v1",
        "data_schema": PUBLIC_PREFIX_PILOT_SCHEMA,
        "source_policy": {
            "description": "exact licensed parent-line continuation after a synthetic typed prefix",
            "human_chronology_observed": False,
            "source_tree_claim": (
                "none; parent commit/path/source bytes and license scope are pinned"
            ),
            "no_suffix_in_model_input": True,
            "line_completion_target": "exact original parent physical line bytes",
            "action_family": "R",
        },
        "training_contract": {
            "context_policy": CONTEXT_POLICY_VERSION,
            "input_rows": 1024,
            "total_rows": 2048,
            "response_tokens_including_eos": 64,
            "objective_loss": "full response including eos only",
            "initial_loss_scale": 128,
        },
        "limits": {
            "max_candidate_rows": EXPECTED_ROW_COUNT,
            "max_new_artifact_bytes": MAX_ARTIFACT_BYTES,
            "max_output_bytes": MAX_PACKAGE_BYTES,
            "provider_calls": 0,
            "model_runs": 0,
            "network_bytes": 0,
            "no_training": True,
        },
        "inputs": {key: value for key, value in source_metadata.items() if key != "split_audit"},
        "selected_rows": [
            {
                "candidate_id": row["candidate_id"],
                "split": row["split"],
                "state_sha256": canonical_sha256(row["state"]),
                "action_sha256": canonical_sha256(row["action"]),
                "source_group_id": row["source_group_id"],
                "task_family_id": row["task_family_id"],
            }
            for row in rows
        ],
        "candidate_pool_sha256": canonical_sha256(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "state_sha256": canonical_sha256(row["state"]),
                    "action_sha256": canonical_sha256(row["action"]),
                }
                for row in rows
            ]
        ),
        "plan_sha256": "",
    }
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(mode=0o700)
    os.chmod(output_dir, 0o700)
    _write_private(output_dir / "plan.json", canonical_bytes(plan) + b"\n")
    artifact_dir = output_dir / "artifacts" / "sha256"
    artifact_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir / "artifacts", 0o700)
    os.chmod(artifact_dir, 0o700)
    artifact_manifest: dict[str, dict[str, Any]] = {}
    for expected_sha, (source_path, size) in sorted(source_artifacts.items()):
        payload = source_path.read_bytes()
        if len(payload) != size or sha256_bytes(payload) != expected_sha:
            raise ValueError("public-prefix source proof changed during staging")
        target = artifact_dir / expected_sha
        _write_private(target, payload)
        artifact_manifest["artifacts/sha256/" + expected_sha] = {
            "bytes": len(payload),
            "sha256": expected_sha,
        }
    for row in rows:
        validate_public_prefix_row(row, package_root=output_dir)
    _write_private(output_dir / "train.jsonl", train_payload)
    _write_private(output_dir / "development.jsonl", development_payload)
    output_manifest = {
        "schema": "one-line-public-prefix-candidate-bank-v1",
        "data_schema": PUBLIC_PREFIX_PILOT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "candidate_pool_sha256": plan["candidate_pool_sha256"],
        "files": {
            name: _file_identity(output_dir / name) for name in ("train.jsonl", "development.jsonl")
        },
        "source_artifact_count": len(artifact_manifest),
        "source_artifacts": artifact_manifest,
        "rows": len(rows),
        "split_counts": source_metadata["split_audit"]["splits"],
        "source_groups_by_split": source_metadata["source_groups_by_split"],
        "task_families_by_split": source_metadata["task_families_by_split"],
        "token_exposure": source_metadata["token_exposure"],
        "action_counts": source_metadata["action_counts"],
        "license_counts": source_metadata["license_counts"],
        "accepted_training": 0,
        "training_ready": False,
        "quality_evidence": False,
        "objective_verified": False,
        "inferability_reviewed": False,
        "human_chronology_observed": False,
        "reason_not_training_ready": (
            "R-only candidate bank; N/I/D/R functional assembly is required"
        ),
        "no_model_or_provider_calls": True,
        "physical_source_proof_bytes": total_bytes,
        "max_output_bytes": MAX_PACKAGE_BYTES,
    }
    _write_private(output_dir / "manifest.json", canonical_bytes(output_manifest) + b"\n")
    return output_manifest


def _write_private(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--expansion", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build_candidate_bank(
            materialized_dir=args.materialized,
            package_root=args.package_root,
            expansion_dir=args.expansion,
            output_dir=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("public-prefix build failed: " + str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "rows": result["rows"],
                "split_counts": result["split_counts"],
                "source_groups_by_split": result["source_groups_by_split"],
                "action_counts": result["action_counts"],
                "accepted_training": result["accepted_training"],
                "training_ready": result["training_ready"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
