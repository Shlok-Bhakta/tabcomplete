"""Allowed data contracts for the existing bounded one-line training pilot."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .context import CONTEXT_POLICY_VERSION, serialize_state_bounded
from .contract import (
    MAX_ACTION_TOKENS,
    EditAction,
    EditState,
    apply_action,
    encode_action,
    physical_lines,
)
from .data import near_duplicate_key, replay_replacement_history


@dataclass(frozen=True)
class PilotDataPolicy:
    data_schema: str
    plan_schema: str
    branch: str
    dataset_id: str
    dataset_license: str
    file_license_status: str
    source_type: str
    repository_plan_path: str


INSTINCT = PilotDataPolicy(
    "one-line-instinct-pilot-v1",
    "one-line-instinct-pilot-plan-v1",
    "research/one-line-gpu-pilot-r1",
    "continuedev/instinct-data",
    "Apache-2.0",
    "unverified",
    "continue_instinct_observed",
    "reports/research/one_line_gpu_pilot_r1/plan.json",
)
CONSTRUCTIVE = PilotDataPolicy(
    "one-line-constructive-pilot-v1",
    "one-line-constructive-pilot-plan-v1",
    "prototype/product-r2",
    "synthetic/tabcomplete-constructive-r1",
    "MIT",
    "own_synthetic_source",
    "synthetic_constructive_objective",
    "reports/prototype/product_r2/constructive_pilot_plan.json",
)
LICENSE_MIXED = PilotDataPolicy(
    "one-line-license-mixed-pilot-v1",
    "one-line-license-mixed-pilot-plan-v1",
    "prototype/product-r2",
    "public-source/license-mixed-pilot-r1",
    "LICENSE-MIXED",
    "per_file_scope_pinned",
    "public_source_supervised_candidate",
    "reports/prototype/product_r2/license_mixed_pilot_plan.json",
)
LICENSE_MIXED_HISTORY = PilotDataPolicy(
    "one-line-license-mixed-history-pilot-v2",
    "one-line-license-mixed-history-pilot-plan-v2",
    "prototype/product-r2",
    "public-source/license-mixed-history-pilot-r2",
    "LICENSE-MIXED",
    "per_file_scope_pinned",
    "reviewed_public_history_candidate",
    "reports/prototype/product_r2/license_mixed_history_pilot_plan_v2.json",
)
REVIEWED_PUBLIC_HISTORY_SOURCE = "reviewed_public_history_candidate"

PUBLIC_SOURCE_TYPES = frozenset(
    {"synthetic_public_source_task", "muse_author_public_candidate"}
)
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_REVISION = re.compile(r"[a-f0-9]{40,64}\Z")


def policy_for_schema(schema: object) -> PilotDataPolicy:
    for policy in (INSTINCT, CONSTRUCTIVE, LICENSE_MIXED, LICENSE_MIXED_HISTORY):
        if schema == policy.data_schema:
            return policy
    raise ValueError("unapproved bounded-pilot data schema")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _license_mixed_id(row: Mapping[str, Any]) -> str | None:
    candidate_id = row.get("candidate_id")
    row_id = row.get("id")
    if candidate_id is not None and row_id is not None and candidate_id != row_id:
        raise ValueError("LICENSE-MIXED candidate ID aliases disagree")
    value = candidate_id if candidate_id is not None else row_id
    return value if isinstance(value, str) and value else None


def license_mixed_row_bindings(row: Mapping[str, Any]) -> dict[str, str]:
    """Bind the accepted action to the exact model state and synthetic history."""
    state = EditState.from_mapping(row["state"])
    action = EditAction(**row["action"])
    after_source = row.get("after_source")
    if not isinstance(after_source, str):
        raise ValueError("LICENSE-MIXED row lacks its reconstructed after-state")
    return {
        "state_sha256": _sha256(_canonical_bytes(asdict(state))),
        "action_sha256": _sha256(_canonical_bytes(asdict(action))),
        "after_source_sha256": _sha256(after_source.encode("utf-8")),
        "history_sha256": _sha256(
            _canonical_bytes([asdict(edit) for edit in state.history])
        ),
        "near_duplicate_sha256": near_duplicate_key(row),
    }


def validate_aggregate_budget(
    budgets: Mapping[str, Any],
    *,
    planned_tokens: int,
    session_seconds: int,
    external_campaign_tokens: int | None = None,
) -> None:
    """Count previous failed sessions as well as completed training exposure."""
    prior_tokens = budgets.get("prior_training_input_tokens")
    prior_seconds = budgets.get("prior_session_wall_seconds")
    if (
        type(prior_tokens) is not int
        or type(prior_seconds) is not int
        or prior_tokens < 0
        or prior_seconds < 0
        or prior_tokens + planned_tokens > 100_000_000
        or prior_seconds + session_seconds > 24 * 3600
        or (external_campaign_tokens is not None and external_campaign_tokens != prior_tokens)
    ):
        raise ValueError("constructive pilot lacks valid aggregate campaign accounting")


def validate_constructive_splits(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Hold out actual authored families within one declared synthetic source.

    Public repository isolation remains in data.validate_splits. This diagnostic
    is authored in a single repository and does not claim repository diversity.
    """
    ids: set[str] = set()
    states: set[str] = set()
    grouped: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    sources: set[str] = set()
    for row in rows:
        identifier = row.get("id")
        split = row.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("duplicate or missing constructive row id")
        if split not in {"train", "development", "test_new_mechanism"}:
            raise ValueError("unknown constructive split")
        ids.add(identifier)
        counts[split] += 1
        source = row.get("source_repo")
        if not isinstance(source, str) or not source:
            raise ValueError("constructive source identity missing")
        sources.add(source)
        state_hash = hashlib.sha256(
            json.dumps(
                asdict(EditState.from_mapping(row["state"])), sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        if state_hash in states:
            raise ValueError("duplicate constructive model input state")
        states.add(state_hash)
        keys = [("near_duplicate", near_duplicate_key(row))]
        for field in ("source_group_id", "session_or_commit", "generator_family", "template_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("constructive grouping identity missing: " + field)
            keys.append((field, value))
        for key in keys:
            prior_split = grouped.setdefault(key, split)
            if prior_split != split:
                raise ValueError("constructive family or near duplicate crosses splits")
    if len(sources) != 1:
        raise ValueError("constructive pilot must declare its one authored source")
    return {
        "rows": len(rows),
        "splits": dict(sorted(counts.items())),
        "authored_source_count": len(sources),
        "repository_diversity_claimed": False,
        "grouping_policy": "constructive-family-split-v1",
    }


def validate_constructive_manifest(manifest: Mapping[str, Any]) -> None:
    """Require the pinned evidence produced by independent candidate review."""
    for key, expected in (
        ("schema", CONSTRUCTIVE.data_schema),
        ("dataset_id", CONSTRUCTIVE.dataset_id),
        ("dataset_license", CONSTRUCTIVE.dataset_license),
        ("source_file_license_status", CONSTRUCTIVE.file_license_status),
    ):
        if manifest.get(key) != expected:
            raise ValueError("constructive pilot manifest identity mismatch: " + key)
    for key in ("generator_sha256", "oracle_sha256", "independent_review_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get(key, ""))):
            raise ValueError("constructive pilot manifest lacks pinned evidence: " + key)


def constructive_row_bindings(row: Mapping[str, Any]) -> dict[str, str]:
    """Bind review to the actual canonical input and supervised action."""
    values = {
        "state_sha256": asdict(EditState.from_mapping(row["state"])),
        "action_sha256": asdict(EditAction(**row["action"])),
    }
    return {
        key: hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        for key, value in values.items()
    }


def validate_constructive_review(
    manifest: Mapping[str, Any], rows: list[Mapping[str, Any]], review_path: Path
) -> None:
    """Require the independent review artifact, not self-certified row flags."""
    payload = review_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("independent_review_sha256"):
        raise ValueError("constructive independent review artifact hash mismatch")
    review = json.loads(payload)
    if (
        not isinstance(review, dict)
        or review.get("schema") != "one-line-constructive-independent-review-v1"
        or review.get("generator_sha256") != manifest["generator_sha256"]
        or review.get("oracle_sha256") != manifest["oracle_sha256"]
        or review.get("full_split_audit_complete") is not True
        or not isinstance(review.get("independent_reviewer_id"), str)
        or not review["independent_reviewer_id"]
        or not isinstance(review.get("rows"), list)
    ):
        raise ValueError("constructive independent review identity or audit is incomplete")
    entries: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], str] = {}
    audited_counts: Counter[str] = Counter()
    group_fields = ("source_group_id", "session_or_commit", "generator_family", "template_id")
    proof_flags = (
        "accepted_training",
        "inferability_reviewed",
        "objective_verified",
        "wrong_action_controls_rejected",
        "history_leakage_check",
    )
    for entry in review["rows"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid constructive review row")
        identifier = entry.get("id")
        split = entry.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in entries:
            raise ValueError("duplicate or missing constructive review row id")
        if split not in {"train", "development", "test_new_mechanism"}:
            raise ValueError("invalid constructive review split")
        entries[identifier] = entry
        audited_counts[split] += 1
        for field in ("state_sha256", "action_sha256", "near_duplicate_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", str(entry.get(field, ""))):
                raise ValueError("constructive review is missing a bound hash: " + field)
        for field in (*group_fields, "near_duplicate_sha256"):
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("constructive review grouping identity missing: " + field)
            previous = grouped.setdefault((field, value), split)
            if previous != split:
                raise ValueError("full constructive review audit crosses splits")
    if (
        dict(audited_counts) != manifest.get("candidate_split_counts")
        or audited_counts["test_new_mechanism"] < 1
    ):
        raise ValueError("constructive review does not cover the complete reserved split audit")
    approved = {
        key
        for key, entry in entries.items()
        if entry["split"] in {"train", "development"} and entry.get("accepted_training") is True
    }
    if approved != {row["id"] for row in rows}:
        raise ValueError("constructive shard IDs differ from independent approved review")
    for row in rows:
        entry = entries[row["id"]]
        expected = {
            **constructive_row_bindings(row),
            **{field: row[field] for field in group_fields},
            "near_duplicate_sha256": near_duplicate_key(row),
            "split": row["split"],
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("constructive row differs from independently reviewed state/action")
        if row.get("independent_reviewer_id") != review["independent_reviewer_id"] or any(
            entry.get(flag) is not True for flag in proof_flags
        ):
            raise ValueError("constructive row lacks actual independent approval")


def _pinned_artifact_bytes(
    root: Path | None,
    path_value: object,
    expected_sha256: object,
    expected_bytes: object,
    *,
    label: str,
) -> bytes:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("LICENSE-MIXED " + label + " artifact is unpinned")
    path = Path(path_value)
    if root is None:
        raise ValueError("LICENSE-MIXED artifact requires an explicit staged package root")
    if path.is_absolute():
        raise ValueError("LICENSE-MIXED portable artifact path must be package-relative")
    if ".." in path.parts:
        raise ValueError("LICENSE-MIXED " + label + " relative artifact escapes its package")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
        or type(expected_bytes) is not int
        or expected_bytes < 0
    ):
        raise ValueError("LICENSE-MIXED " + label + " relative artifact is unpinned")
    return _read_relative_artifact(
        root,
        path_value,
        label=label,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def _provenance_value(
    row: Mapping[str, Any], metadata: Mapping[str, Any], key: str
) -> object:
    """Read a provenance field from metadata or its older flat row location."""
    nested = metadata.get(key)
    flat = row.get(key)
    if nested is not None and flat is not None and nested != flat:
        raise ValueError("LICENSE-MIXED provenance field disagrees: " + key)
    return nested if nested is not None else flat


def _spdx_scope_entry(scope_payload: bytes, candidate_id: str) -> Mapping[str, Any]:
    try:
        raw = _strict_json(scope_payload, label="path-scope evidence")
    except ValueError:
        raise ValueError("LICENSE-MIXED path-scope evidence is invalid") from None
    if not isinstance(raw, Mapping) or raw.get("schema") != "exact-parent-license-scope-v2":
        raise ValueError("LICENSE-MIXED path-scope evidence has an unknown schema")
    results = raw.get("results")
    if not isinstance(results, list):
        raise ValueError("LICENSE-MIXED path-scope evidence has no result list")
    matches = [
        item
        for item in results
        if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "verified_path_scope":
        raise ValueError("LICENSE-MIXED file lacks one verified path-scope record")
    return matches[0]


def validate_license_mixed_source_artifacts(
    row: Mapping[str, Any], *, package_root: Path | None = None
) -> dict[str, str]:
    """Reopen each pinned public source, scope, and transform artifact for a row."""
    if package_root is None:
        raise ValueError("LICENSE-MIXED artifact validation requires an explicit package root")
    metadata = row.get("authoring_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("LICENSE-MIXED row lacks per-file authoring metadata")
    candidate_id = _license_mixed_id(row)
    source_repo = metadata.get("source_repo")
    source_revision = metadata.get("source_revision")
    source_tree_sha = metadata.get("source_tree_sha")
    source_path = metadata.get("source_path")
    source_sha256 = metadata.get("source_sha256")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(source_repo, str)
        or not source_repo
        or not isinstance(source_revision, str)
        or _REVISION.fullmatch(source_revision) is None
        or not isinstance(source_tree_sha, str)
        or _REVISION.fullmatch(source_tree_sha) is None
        or not isinstance(source_path, str)
        or not source_path
        or Path(source_path).is_absolute()
        or ".." in Path(source_path).parts
        or not isinstance(source_sha256, str)
        or _SHA256.fullmatch(source_sha256) is None
    ):
        raise ValueError("LICENSE-MIXED source file identity is incomplete")
    source_payload = _pinned_artifact_bytes(
        package_root,
        _provenance_value(row, metadata, "source_artifact_path"),
        _provenance_value(row, metadata, "source_artifact_sha256"),
        _provenance_value(row, metadata, "source_artifact_bytes"),
        label="source",
    )
    if _sha256(source_payload) != source_sha256:
        raise ValueError("LICENSE-MIXED raw source bytes differ from pinned source hash")

    license_id = metadata.get("path_license")
    license_sha256 = metadata.get("path_license_sha256")
    license_path = metadata.get("license_path")
    license_git_blob_sha = metadata.get("path_license_git_blob_sha")
    if (
        license_id not in {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
        or row.get("source_license") != license_id
        or metadata.get("source_license") != license_id
        or metadata.get("license_scope_status") != "verified_path_scope"
        or not isinstance(license_path, str)
        or not license_path
        or Path(license_path).is_absolute()
        or ".." in Path(license_path).parts
        or not isinstance(license_sha256, str)
        or _SHA256.fullmatch(license_sha256) is None
        or not isinstance(license_git_blob_sha, str)
        or re.fullmatch(r"[a-f0-9]{40}", license_git_blob_sha) is None
    ):
        raise ValueError("LICENSE-MIXED file license identity is incomplete")
    license_payload = _pinned_artifact_bytes(
        package_root,
        _provenance_value(row, metadata, "path_license_artifact_path"),
        _provenance_value(row, metadata, "path_license_artifact_sha256"),
        _provenance_value(row, metadata, "path_license_artifact_bytes"),
        label="license",
    )
    if _sha256(license_payload) != license_sha256:
        raise ValueError("LICENSE-MIXED license bytes differ from pinned license hash")
    scope_payload = _pinned_artifact_bytes(
        package_root,
        _provenance_value(row, metadata, "license_scope_artifact_path"),
        _provenance_value(row, metadata, "license_scope_sha256"),
        _provenance_value(row, metadata, "license_scope_artifact_bytes"),
        label="path-scope",
    )
    scope = _spdx_scope_entry(scope_payload, candidate_id)
    path_scope = scope.get("path_scope")
    if not isinstance(path_scope, Mapping):
        raise ValueError("LICENSE-MIXED path-scope record is incomplete")
    expected_license = license_id.casefold()
    spdx = path_scope.get("spdx")
    source_header_spdx = scope.get("source_header_spdx")
    if (
        scope.get("repository") != source_repo
        or scope.get("parent_commit") != source_revision
        or scope.get("source_path") != source_path
        or scope.get("source_sha256") != source_sha256
        or scope.get("source_group_id") != row.get("source_group_id")
        or path_scope.get("sha256") != license_sha256
        or path_scope.get("license_path") != license_path
        or path_scope.get("git_blob_sha") != license_git_blob_sha
        or path_scope.get("scope") != "root"
        or not isinstance(spdx, list)
        or [str(item).casefold() for item in spdx] != [expected_license]
        or path_scope.get("spdx_ambiguous") is not False
        or path_scope.get("spdx_expression_count") != 0
        or scope.get("source_header_spdx_ambiguous") is not False
        or scope.get("source_header_spdx_expression_count") != 0
        or not isinstance(source_header_spdx, list)
        or any(str(item).casefold() != expected_license for item in source_header_spdx)
        or scope.get("additional_license_references") != []
        or scope.get("reuse_dep5_references") != []
    ):
        raise ValueError("LICENSE-MIXED path-scope record differs from source/license identity")
    root_license = scope.get("root_license")
    if (
        not isinstance(root_license, Mapping)
        or root_license.get("path") != license_path
        or root_license.get("sha256") != license_sha256
        or root_license.get("git_blob_sha") != license_git_blob_sha
        or not isinstance(root_license.get("root_spdx"), list)
        or [str(item).casefold() for item in root_license["root_spdx"]] != [expected_license]
    ):
        raise ValueError("LICENSE-MIXED root license blob differs from file scope")

    transform = metadata.get("transform")
    if not isinstance(transform, Mapping) or set(transform) != {
        "kind",
        "start_line_1based",
        "end_line_1based_inclusive",
        "parent_source_sha256",
        "selected_source_sha256",
    }:
        raise ValueError("LICENSE-MIXED transform descriptor is incomplete")
    transform_sha256 = metadata.get("transform_sha256")
    if (
        not isinstance(transform_sha256, str)
        or _SHA256.fullmatch(transform_sha256) is None
        or _sha256(_canonical_bytes(transform)) != transform_sha256
        or transform.get("parent_source_sha256") != source_sha256
    ):
        raise ValueError("LICENSE-MIXED transform descriptor hash mismatch")
    lines = physical_lines(source_payload)
    start, end = transform.get("start_line_1based"), transform.get("end_line_1based_inclusive")
    if (
        type(start) is not int
        or type(end) is not int
        or start < 1
        or end < start
        or end > len(lines)
    ):
        raise ValueError("LICENSE-MIXED line-slice bounds are invalid")
    selected = b"".join(line.raw for line in lines[start - 1 : end])
    selected_sha256 = _sha256(selected)
    expected_kind = (
        "exact_parent_file"
        if start == 1 and end == len(lines)
        else "contiguous_parent_file_line_slice"
    )
    if (
        transform.get("kind") != expected_kind
        or transform.get("selected_source_sha256") != selected_sha256
        or metadata.get("selected_source_sha256") != selected_sha256
    ):
        raise ValueError("LICENSE-MIXED line-slice output differs from pinned transform")
    try:
        selected_source = selected.decode("utf-8")
        state = EditState.from_mapping(row["state"])
        replayed = replay_replacement_history(
            selected_source,
            state.history,
            file_id=state.file_id,
            filetype=state.filetype,
        )
    except (KeyError, UnicodeDecodeError, TypeError, ValueError):
        raise ValueError("LICENSE-MIXED source transform/history cannot be replayed") from None
    if replayed != state.source:
        raise ValueError("LICENSE-MIXED source transform/history differs from model state")
    history_sha256 = row.get("history_sha256")
    expected_history_sha256 = _sha256(
        _canonical_bytes([asdict(edit) for edit in state.history])
    )
    if history_sha256 != expected_history_sha256:
        raise ValueError("LICENSE-MIXED synthetic history hash mismatch")
    chronology = row.get("human_chronology_observed", row.get("chronology_observed"))
    if chronology is not False:
        raise ValueError("LICENSE-MIXED row claims unverified human chronology")
    return {
        "source_tree_sha": source_tree_sha,
        "source_sha256": source_sha256,
        "path_license_sha256": license_sha256,
        "license_scope_sha256": str(metadata["license_scope_sha256"]),
        "transform_sha256": transform_sha256,
        "history_sha256": expected_history_sha256,
    }


def validate_license_mixed_splits(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject public-source groups, task families, and near duplicates crossing splits."""
    ids: set[str] = set()
    states: set[str] = set()
    normalized_inputs: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    repositories: set[str] = set()
    allowed_splits = {"train", "development", "test_new_repo", "test_new_mechanism"}
    for row in rows:
        identifier, split = _license_mixed_id(row), row.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("duplicate or missing LICENSE-MIXED row id")
        if split not in allowed_splits:
            raise ValueError("invalid LICENSE-MIXED split")
        ids.add(identifier)
        counts[split] += 1
        metadata = row.get("authoring_metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("LICENSE-MIXED row source metadata is invalid")
        source_repo = row.get("source_repo", metadata.get("source_repo"))
        if (
            not isinstance(source_repo, str)
            or not source_repo
            or (
                "source_repo" in row
                and "source_repo" in metadata
                and row["source_repo"] != metadata["source_repo"]
            )
        ):
            raise ValueError("LICENSE-MIXED repository identity is missing or inconsistent")
        repositories.add(source_repo.casefold())
        state = EditState.from_mapping(row["state"])
        state_sha256 = _sha256(_canonical_bytes(asdict(state)))
        if state_sha256 in states:
            raise ValueError("duplicate LICENSE-MIXED model input state")
        states.add(state_sha256)
        normalized_input_sha256 = _normalized_model_input_sha256(row)
        if normalized_input_sha256 in normalized_inputs:
            raise ValueError("duplicate normalized LICENSE-MIXED model input")
        normalized_inputs.add(normalized_input_sha256)
        near_sha256 = near_duplicate_key(row)
        if row.get("near_duplicate_sha256", near_sha256) != near_sha256:
            raise ValueError("LICENSE-MIXED near-duplicate hash mismatch")
        keys = (
            ("source_group_id", row.get("source_group_id")),
            ("source_repo", source_repo.casefold()),
            ("session_or_commit", row.get("session_or_commit")),
            ("task_family_id", row.get("task_family_id", metadata.get("task_family_id"))),
            ("near_duplicate_sha256", near_sha256),
        )
        for field, value in keys:
            if not isinstance(value, str) or not value:
                raise ValueError("LICENSE-MIXED grouping identity missing: " + field)
            prior_split = groups.setdefault((field, value.casefold()), str(split))
            if prior_split != split:
                raise ValueError("LICENSE-MIXED grouping identity crosses splits: " + field)
    return {
        "rows": len(rows),
        "splits": dict(sorted(counts.items())),
        "source_repository_count": len(repositories),
        "grouping_policy": "license-mixed-source-task-near-duplicate-v1",
    }


def _normalized_model_input_sha256(row: Mapping[str, Any]) -> str:
    """Hash exact model-visible state while ignoring file and repository aliases."""
    state = EditState.from_mapping(row["state"])
    normalized_state = {
        "filetype": state.filetype,
        "source": state.source,
        "target_row": state.target_row,
        "cursor_col": state.cursor_col,
        "history": [asdict(edit) for edit in state.history],
        "relevant": list(state.relevant),
    }
    return _sha256(_canonical_bytes(normalized_state))


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("LICENSE-MIXED evidence has duplicate JSON keys")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_unique_json_pairs)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("LICENSE-MIXED " + label + " is not strict JSON") from None


def validate_license_mixed_manifest(
    manifest: Mapping[str, Any], *, policy: PilotDataPolicy = LICENSE_MIXED
) -> None:
    """Require the frozen mixed-license policy and immutable shard identities."""
    if policy not in {LICENSE_MIXED, LICENSE_MIXED_HISTORY}:
        raise ValueError("unapproved LICENSE-MIXED policy")
    for key, expected in (
        ("schema", policy.data_schema),
        ("dataset_id", policy.dataset_id),
        ("dataset_license", policy.dataset_license),
        ("source_file_license_status", policy.file_license_status),
    ):
        if manifest.get(key) != expected:
            raise ValueError("LICENSE-MIXED manifest identity mismatch: " + key)
    for key in (
        "train_sha256",
        "development_sha256",
        "independent_review_sha256",
    ):
        if not isinstance(manifest.get(key), str) or _SHA256.fullmatch(manifest[key]) is None:
            raise ValueError("LICENSE-MIXED manifest lacks pinned evidence: " + key)
    review_path = manifest.get("independent_review_path")
    review_bytes = manifest.get("independent_review_bytes")
    if (
        not isinstance(review_path, str)
        or not review_path
        or Path(review_path).is_absolute()
        or ".." in Path(review_path).parts
        or type(review_bytes) is not int
        or review_bytes < 1
    ):
        raise ValueError("LICENSE-MIXED independent review artifact is unpinned")
    for key in ("train_count", "dev_count"):
        if type(manifest.get(key)) is not int or manifest[key] < 1:
            raise ValueError("LICENSE-MIXED manifest count or artifact size is invalid: " + key)
    if not 128 <= manifest["train_count"] <= 1024 or manifest["dev_count"] < 64:
        raise ValueError("LICENSE-MIXED pilot is below the frozen train/development floors")
    if (
        manifest.get("file_groups_disjoint") is not True
        or not isinstance(manifest.get("candidate_split_counts"), Mapping)
    ):
        raise ValueError("LICENSE-MIXED manifest lacks split audit identity")
    artifact_root = manifest.get("artifact_root", ".")
    if (
        not isinstance(artifact_root, str)
        or not artifact_root
        or Path(artifact_root).is_absolute()
        or ".." in Path(artifact_root).parts
    ):
        raise ValueError("LICENSE-MIXED artifact root is not package-relative")
    oracle_fields = (
        "oracle_results_path",
        "oracle_results_sha256",
        "oracle_results_bytes",
        "oracle_evaluator_sha256",
        "oracle_diagnostics_path",
        "oracle_diagnostics_sha256",
        "oracle_diagnostics_bytes",
    )
    present_oracle_fields = [key for key in oracle_fields if key in manifest]
    if present_oracle_fields and len(present_oracle_fields) != len(oracle_fields):
        raise ValueError("LICENSE-MIXED oracle proof manifest is incomplete")
    for key in (
        "oracle_results_sha256",
        "oracle_evaluator_sha256",
        "oracle_diagnostics_sha256",
    ):
        if key in manifest and (
            not isinstance(manifest[key], str) or _SHA256.fullmatch(manifest[key]) is None
        ):
            raise ValueError("LICENSE-MIXED oracle digest is invalid: " + key)
    for key in ("oracle_results_bytes", "oracle_diagnostics_bytes"):
        if key in manifest and (type(manifest[key]) is not int or manifest[key] < 1):
            raise ValueError("LICENSE-MIXED oracle artifact size is invalid: " + key)
    for key in ("oracle_results_path", "oracle_diagnostics_path"):
        if key in manifest and (
            not isinstance(manifest[key], str)
            or not manifest[key]
            or Path(manifest[key]).is_absolute()
            or ".." in Path(manifest[key]).parts
        ):
            raise ValueError("LICENSE-MIXED oracle artifact path is not package-relative")
    provenance_fields = (
        "provenance_manifest_path",
        "provenance_manifest_sha256",
        "provenance_manifest_bytes",
    )
    present_provenance_fields = [key for key in provenance_fields if key in manifest]
    if present_provenance_fields and len(present_provenance_fields) != len(provenance_fields):
        raise ValueError("LICENSE-MIXED provenance manifest reference is incomplete")
    if "provenance_manifest_sha256" in manifest and (
        not isinstance(manifest["provenance_manifest_sha256"], str)
        or _SHA256.fullmatch(manifest["provenance_manifest_sha256"]) is None
        or type(manifest.get("provenance_manifest_bytes")) is not int
        or manifest["provenance_manifest_bytes"] < 1
        or not isinstance(manifest.get("provenance_manifest_path"), str)
        or not manifest["provenance_manifest_path"]
        or Path(manifest["provenance_manifest_path"]).is_absolute()
        or ".." in Path(manifest["provenance_manifest_path"]).parts
    ):
        raise ValueError("LICENSE-MIXED provenance manifest reference is invalid")
    split_counts = manifest["candidate_split_counts"]
    if any(
        split not in {"train", "development", "test_new_repo", "test_new_mechanism"}
        or type(count) is not int
        or count < 0
        for split, count in split_counts.items()
    ):
        raise ValueError("LICENSE-MIXED candidate split counts are invalid")
    if (
        split_counts.get("train", 0) < manifest["train_count"]
        or split_counts.get("development", 0) < manifest["dev_count"]
    ):
        raise ValueError("LICENSE-MIXED candidate split audit is smaller than accepted shards")


def _read_relative_artifact(
    root: Path,
    path_value: object,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> bytes:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("LICENSE-MIXED " + label + " path is absent")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("LICENSE-MIXED " + label + " path is outside its package")
    path = root / relative
    try:
        if root.is_symlink():
            raise ValueError("LICENSE-MIXED " + label + " package root is a symlink")
        resolved_root = root.resolve(strict=True)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("LICENSE-MIXED " + label + " path contains a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("LICENSE-MIXED " + label + " path escapes its package")
        if path.is_symlink() or not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError("LICENSE-MIXED " + label + " is not a regular file")
        payload = resolved.read_bytes()
    except OSError:
        raise ValueError("LICENSE-MIXED " + label + " is unavailable") from None
    if expected_sha256 is not None and _sha256(payload) != expected_sha256:
        raise ValueError("LICENSE-MIXED " + label + " hash mismatch")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError("LICENSE-MIXED " + label + " size mismatch")
    return payload


def _oracle_results(
    manifest: Mapping[str, Any], root: Path
) -> dict[str, tuple[dict[str, Any], str]]:
    payload = _read_relative_artifact(
        root,
        manifest.get("oracle_results_path"),
        label="oracle results",
        expected_sha256=str(manifest["oracle_results_sha256"]),
        expected_bytes=int(manifest["oracle_results_bytes"]),
    )
    records: dict[str, tuple[dict[str, Any], str]] = {}
    try:
        lines = payload.splitlines()
        for line in lines:
            if not line.strip():
                raise ValueError("blank line")
            record = _strict_json(line, label="oracle result row")
            if not isinstance(record, dict):
                raise ValueError("non-object")
            identifier = record.get("candidate_id")
            if not isinstance(identifier, str) or not identifier or identifier in records:
                raise ValueError("duplicate or missing id")
            records[identifier] = (record, _sha256(_canonical_bytes(record)))
    except (UnicodeError, ValueError):
        raise ValueError("LICENSE-MIXED oracle results are invalid") from None
    if not records:
        raise ValueError("LICENSE-MIXED oracle results are empty")
    return records


def _oracle_diagnostics(
    manifest: Mapping[str, Any], root: Path
) -> dict[tuple[str, str], tuple[dict[str, Any], str]]:
    payload = _read_relative_artifact(
        root,
        manifest.get("oracle_diagnostics_path"),
        label="oracle diagnostics",
        expected_sha256=str(manifest["oracle_diagnostics_sha256"]),
        expected_bytes=int(manifest["oracle_diagnostics_bytes"]),
    )
    records: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for diagnostic in _strict_jsonl(payload, label="oracle diagnostics"):
        candidate_id, variant = diagnostic.get("candidate_id"), diagnostic.get("variant")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(variant, str)
            or not variant
        ):
            raise ValueError("LICENSE-MIXED oracle diagnostics have duplicate or missing IDs")
        key = (candidate_id, variant)
        if key in records:
            raise ValueError("LICENSE-MIXED oracle diagnostics have duplicate or missing IDs")
        records[key] = (diagnostic, _sha256(_canonical_bytes(diagnostic)))
    if not records:
        raise ValueError("LICENSE-MIXED oracle diagnostics are empty")
    return records


def _artifact_descriptor_bytes(
    root: Path, value: object, *, label: str
) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError("LICENSE-MIXED " + label + " artifact descriptor is invalid")
    payload = _pinned_artifact_bytes(
        root,
        value.get("path"),
        value.get("sha256"),
        value.get("bytes"),
        label=label,
    )
    return payload, dict(value)


def _strict_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line in payload.splitlines():
            if not line.strip():
                raise ValueError("blank row")
            value = _strict_json(line, label=label)
            if not isinstance(value, dict):
                raise ValueError("nonobject row")
            records.append(value)
    except (UnicodeError, ValueError):
        raise ValueError("LICENSE-MIXED " + label + " JSONL is invalid") from None
    return records


def _valid_elapsed_seconds(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    )


def _unique_index(
    rows: list[dict[str, Any]], *, key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in indexed:
            raise ValueError("LICENSE-MIXED " + label + " identity is duplicate or missing")
        indexed[value] = row
    return indexed


def _license_mixed_provenance(
    manifest: Mapping[str, Any], artifact_root: Path
) -> dict[str, Any] | None:
    if "provenance_manifest_path" not in manifest:
        return None
    provenance_payload = _pinned_artifact_bytes(
        artifact_root,
        manifest.get("provenance_manifest_path"),
        manifest.get("provenance_manifest_sha256"),
        manifest.get("provenance_manifest_bytes"),
        label="source provenance manifest",
    )
    provenance = _strict_json(provenance_payload, label="source provenance manifest")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != "private-public-source-provenance-manifest-v2"
        or provenance.get("training_ready") not in {False, None}
        or provenance.get("accepted_training_count") not in {0, None}
    ):
        raise ValueError(
            "LICENSE-MIXED source provenance manifest is not the pinned public-source form"
        )
    required = (
        "source_verification_manifest",
        "source_verification_candidate_rows",
        "source_task_spec",
        "private_objective_fixtures",
        "candidates",
    )
    if any(key not in provenance for key in required) or not isinstance(
        provenance.get("candidates"), list
    ):
        raise ValueError("LICENSE-MIXED source provenance bundle is incomplete")
    source_manifest_payload, source_manifest_ref = _artifact_descriptor_bytes(
        artifact_root, provenance["source_verification_manifest"], label="source audit manifest"
    )
    source_manifest = _strict_json(source_manifest_payload, label="source audit manifest")
    source_rows_payload, source_rows_ref = _artifact_descriptor_bytes(
        artifact_root,
        provenance["source_verification_candidate_rows"],
        label="source audit candidate rows",
    )
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema") != "commit-sequence-source-verification-result-v1"
        or source_manifest.get("status") != "complete"
        or source_manifest.get("accepted_training") != 0
        or source_manifest.get("chronology_observed") is not False
        or source_manifest.get("candidate_results_sha256") != source_rows_ref["sha256"]
        or _sha256(source_rows_payload) != source_manifest.get("candidate_results_sha256")
    ):
        raise ValueError("LICENSE-MIXED public-source verification audit is incomplete")
    source_rows = _unique_index(
        _strict_jsonl(source_rows_payload, label="source verification"),
        key="candidate_id",
        label="source verification row",
    )
    spec_payload, spec_ref = _artifact_descriptor_bytes(
        artifact_root, provenance["source_task_spec"], label="source task spec"
    )
    spec = _strict_json(spec_payload, label="source task spec")
    fixture_payload, fixture_ref = _artifact_descriptor_bytes(
        artifact_root, provenance["private_objective_fixtures"], label="objective fixtures"
    )
    fixture_doc = _strict_json(fixture_payload, label="objective fixtures")
    seeds = spec.get("seeds") if isinstance(spec, Mapping) else None
    fixtures = fixture_doc.get("seeds") if isinstance(fixture_doc, Mapping) else None
    if not isinstance(seeds, list) or not isinstance(fixtures, list):
        raise ValueError("LICENSE-MIXED source task/fixture identity is incomplete")
    spec_by_candidate: dict[str, dict[str, Any]] = {}
    spec_by_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        if not isinstance(seed, dict):
            raise ValueError("LICENSE-MIXED source task spec row is invalid")
        candidate_id, seed_id = seed.get("candidate_id"), seed.get("seed_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(seed_id, str)
            or not seed_id
            or candidate_id in spec_by_candidate
            or seed_id in spec_by_seed
        ):
            raise ValueError("LICENSE-MIXED source task spec IDs are duplicate or missing")
        spec_by_candidate[candidate_id] = seed
        spec_by_seed[seed_id] = seed
    fixture_by_seed: dict[str, dict[str, Any]] = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("LICENSE-MIXED objective fixture row is invalid")
        seed_id = fixture.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id or seed_id in fixture_by_seed:
            raise ValueError("LICENSE-MIXED objective fixture IDs are duplicate or missing")
        fixture_by_seed[seed_id] = fixture
    if set(spec_by_seed) != set(fixture_by_seed):
        raise ValueError("LICENSE-MIXED task spec and fixture seeds disagree")
    for seed_id, fixture in fixture_by_seed.items():
        fixture_candidate = fixture.get("candidate_id")
        if (
            fixture_candidate is not None
            and fixture_candidate != spec_by_seed[seed_id]["candidate_id"]
        ):
            raise ValueError("LICENSE-MIXED fixture candidate binding disagrees with its spec")
    if any(not isinstance(item, dict) for item in provenance["candidates"]):
        raise ValueError("LICENSE-MIXED source provenance candidate is invalid")
    provenance_candidates = _unique_index(
        provenance["candidates"],
        key="candidate_id",
        label="source provenance candidate",
    )
    return {
        "provenance": provenance,
        "source_verification_manifest": source_manifest,
        "source_manifest_ref": source_manifest_ref,
        "source_rows": source_rows,
        "source_rows_ref": source_rows_ref,
        "spec": spec,
        "spec_ref": spec_ref,
        "spec_by_candidate": spec_by_candidate,
        "spec_by_seed": spec_by_seed,
        "fixtures": fixture_doc,
        "fixture_ref": fixture_ref,
        "fixture_by_seed": fixture_by_seed,
        "provenance_candidates": provenance_candidates,
    }


def _validate_synthetic_provenance_row(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    identifier = _license_mixed_id(row)
    if identifier is None:
        raise ValueError("LICENSE-MIXED synthetic row has no candidate identity")
    seed_id = row.get("seed_id")
    mapped_seed = provenance["spec_by_candidate"].get(identifier)
    if (
        not isinstance(seed_id, str)
        or not isinstance(mapped_seed, Mapping)
        or mapped_seed.get("seed_id") != seed_id
        or result.get("seed_id") != seed_id
    ):
        raise ValueError("LICENSE-MIXED candidate/seed mapping differs from its frozen spec")
    fixture = provenance["fixture_by_seed"].get(seed_id)
    if not isinstance(fixture, Mapping) or not isinstance(fixture.get("oracle"), Mapping):
        raise ValueError("LICENSE-MIXED candidate has no frozen independent fixture")
    state = EditState.from_mapping(row["state"])
    if (
        not state.relevant
        or mapped_seed.get("student_request") != state.relevant[0]
        or fixture.get("action") != row.get("action")
    ):
        raise ValueError("LICENSE-MIXED synthetic request/gold differs from its frozen task")
    try:
        wrong_action = EditAction(**fixture["wrong_action"])
        wrong_after = apply_action(state, wrong_action)
    except (KeyError, TypeError, ValueError):
        raise ValueError("LICENSE-MIXED synthetic behavior-breaking action is invalid") from None
    if wrong_after == state.source or wrong_after == row.get("after_source"):
        raise ValueError("LICENSE-MIXED synthetic behavior-breaking action has no distinct effect")
    fixture_sha256 = _sha256(_canonical_bytes(fixture["oracle"]))
    if fixture_sha256 != result.get("fixture_sha256"):
        raise ValueError("LICENSE-MIXED oracle fixture differs from the frozen seed fixture")
    fixture_binding = row.get("objective_fixture_binding")
    if not isinstance(fixture_binding, Mapping) or fixture_binding.get("sha256") != fixture_sha256:
        raise ValueError("LICENSE-MIXED training row lacks its frozen objective fixture binding")

    metadata = row.get("authoring_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("LICENSE-MIXED synthetic row lacks source provenance")
    candidate_provenance = provenance["provenance_candidates"].get(identifier)
    if not isinstance(candidate_provenance, Mapping):
        raise ValueError("LICENSE-MIXED candidate is absent from source provenance manifest")
    source_result = provenance["source_rows"].get(identifier)
    if not isinstance(source_result, Mapping):
        raise ValueError("LICENSE-MIXED candidate is absent from source verification audit")
    source_pair = source_result.get("source_pair")
    parent_license = source_result.get("parent_license")
    root_license = (
        parent_license.get("root_license") if isinstance(parent_license, Mapping) else None
    )
    if not isinstance(source_pair, Mapping) or not isinstance(parent_license, Mapping):
        raise ValueError("LICENSE-MIXED source verification row lacks parent identity")
    source_identity = {
        "source_repo": metadata.get("source_repo"),
        "source_revision": metadata.get("source_revision"),
        "source_tree_sha": metadata.get("source_tree_sha"),
        "source_path": metadata.get("source_path"),
        "source_sha256": metadata.get("source_sha256"),
        "source_group_id": row.get("source_group_id"),
    }
    observed_source = {
        "source_repo": source_result.get("repository"),
        "source_revision": source_result.get("parent_commit"),
        "source_tree_sha": parent_license.get("tree_sha"),
        "source_path": source_result.get("file_path"),
        "source_sha256": source_pair.get("parent_sha256"),
        "source_group_id": source_result.get("source_group_id"),
    }
    if (
        source_identity != observed_source
        or source_result.get("status") != "source_and_license_verified_for_human_review"
        or source_result.get("candidate_id") != identifier
        or not isinstance(root_license, Mapping)
    ):
        raise ValueError("LICENSE-MIXED row differs from exact-parent source verification")

    file_bindings = {
        "repository": metadata.get("source_repo"),
        "source_revision": metadata.get("source_revision"),
        "source_tree_sha": metadata.get("source_tree_sha"),
        "source_path": metadata.get("source_path"),
        "parent_source_sha256": metadata.get("source_sha256"),
        "selected_source_sha256": metadata.get("selected_source_sha256"),
        "transform_sha256": metadata.get("transform_sha256"),
        "source_group_id": row.get("source_group_id"),
        "path_license": metadata.get("path_license"),
        "path_license_spdx": metadata.get("path_license_spdx"),
        "path_license_scope": metadata.get("path_license_scope"),
        "path_license_git_blob_sha": metadata.get("path_license_git_blob_sha"),
        "path_license_sha256": metadata.get("path_license_sha256"),
        "history_sha256": row.get("history_sha256"),
        "seed_id": seed_id,
        "state_sha256": license_mixed_row_bindings(row)["state_sha256"],
        "context_sha256": row.get("context_sha256"),
        "objective_fixture_sha256": fixture_sha256,
    }
    if any(candidate_provenance.get(key) != value for key, value in file_bindings.items()):
        raise ValueError("LICENSE-MIXED row differs from frozen source provenance manifest")
    for reference_key, metadata_path, metadata_sha, metadata_bytes in (
        (
            "source_artifact",
            "source_artifact_path",
            "source_artifact_sha256",
            "source_artifact_bytes",
        ),
        (
            "path_license_artifact",
            "path_license_artifact_path",
            "path_license_artifact_sha256",
            "path_license_artifact_bytes",
        ),
        (
            "license_scope_artifact",
            "license_scope_artifact_path",
            "license_scope_sha256",
            "license_scope_artifact_bytes",
        ),
    ):
        reference = candidate_provenance.get(reference_key)
        if not isinstance(reference, Mapping) or (
            metadata.get(metadata_path) != reference.get("path")
            or metadata.get(metadata_sha) != reference.get("sha256")
            or metadata.get(metadata_bytes) != reference.get("bytes")
        ):
            raise ValueError("LICENSE-MIXED per-file artifact differs from provenance manifest")
    return fixture_sha256


def _validate_public_history_provenance_row(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Bind a fixed-state history candidate to its public source and oracle fixture."""
    identifier = _license_mixed_id(row)
    seed_id = row.get("seed_id")
    mapped_seed = provenance["spec_by_candidate"].get(identifier)
    if (
        identifier is None
        or not isinstance(seed_id, str)
        or not isinstance(mapped_seed, Mapping)
        or mapped_seed.get("seed_id") != seed_id
        or result.get("seed_id") != seed_id
        or row.get("source_type") != REVIEWED_PUBLIC_HISTORY_SOURCE
        or row.get("history_order") != "synthetic_fixed_before_provider"
    ):
        raise ValueError("LICENSE-MIXED fixed-history candidate/seed binding is invalid")
    fixture = provenance["fixture_by_seed"].get(seed_id)
    if not isinstance(fixture, Mapping) or not isinstance(fixture.get("oracle"), Mapping):
        raise ValueError("LICENSE-MIXED fixed-history row lacks its objective fixture")
    state = EditState.from_mapping(row["state"])
    if (
        not state.history
        or state.relevant
        or fixture.get("action") != row.get("action")
        or mapped_seed.get("action", row.get("action")) != row.get("action")
    ):
        raise ValueError("LICENSE-MIXED fixed-history gold differs from its frozen task")
    try:
        wrong_action = EditAction(**fixture["wrong_action"])
        wrong_after = apply_action(state, wrong_action)
        gold_after = apply_action(state, EditAction(**row["action"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("LICENSE-MIXED fixed-history behavior control is invalid") from None
    if wrong_after in {state.source, gold_after}:
        raise ValueError("LICENSE-MIXED fixed-history behavior control has no distinct effect")
    fixture_sha256 = _sha256(_canonical_bytes(fixture["oracle"]))
    if fixture_sha256 != result.get("fixture_sha256"):
        raise ValueError("LICENSE-MIXED fixed-history oracle differs from its frozen fixture")
    fixture_binding = row.get("objective_fixture_binding")
    if not isinstance(fixture_binding, Mapping) or fixture_binding.get("sha256") != fixture_sha256:
        raise ValueError("LICENSE-MIXED fixed-history row lacks its bound objective fixture")

    metadata = row.get("authoring_metadata")
    candidate_provenance = provenance["provenance_candidates"].get(identifier)
    source_result = provenance["source_rows"].get(identifier)
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(candidate_provenance, Mapping)
        or not isinstance(source_result, Mapping)
    ):
        raise ValueError("LICENSE-MIXED fixed-history source provenance is incomplete")
    source_pair = source_result.get("source_pair")
    parent_license = source_result.get("parent_license")
    root_license = (
        parent_license.get("root_license") if isinstance(parent_license, Mapping) else None
    )
    if not isinstance(source_pair, Mapping) or not isinstance(parent_license, Mapping):
        raise ValueError(
            "LICENSE-MIXED fixed-history source verification row lacks parent identity"
        )
    source_identity = {
        "source_repo": metadata.get("source_repo"),
        "source_revision": metadata.get("source_revision"),
        "source_tree_sha": metadata.get("source_tree_sha"),
        "source_path": metadata.get("source_path"),
        "source_sha256": metadata.get("source_sha256"),
        "source_group_id": row.get("source_group_id"),
    }
    observed_source = {
        "source_repo": source_result.get("repository"),
        "source_revision": source_result.get("parent_commit"),
        "source_tree_sha": parent_license.get("tree_sha"),
        "source_path": source_result.get("file_path"),
        "source_sha256": source_pair.get("parent_sha256"),
        "source_group_id": source_result.get("source_group_id"),
    }
    if (
        source_identity != observed_source
        or source_result.get("status") != "source_and_license_verified_for_human_review"
        or source_result.get("candidate_id") != identifier
        or not isinstance(root_license, Mapping)
    ):
        raise ValueError("LICENSE-MIXED fixed-history row differs from exact-parent source proof")

    file_bindings = {
        "repository": metadata.get("source_repo"),
        "source_revision": metadata.get("source_revision"),
        "source_tree_sha": metadata.get("source_tree_sha"),
        "source_path": metadata.get("source_path"),
        "parent_source_sha256": metadata.get("source_sha256"),
        "selected_source_sha256": metadata.get("selected_source_sha256"),
        "transform_sha256": metadata.get("transform_sha256"),
        "source_group_id": row.get("source_group_id"),
        "path_license": metadata.get("path_license"),
        "path_license_spdx": metadata.get("path_license_spdx"),
        "path_license_scope": metadata.get("path_license_scope"),
        "path_license_git_blob_sha": metadata.get("path_license_git_blob_sha"),
        "path_license_sha256": metadata.get("path_license_sha256"),
        "history_sha256": license_mixed_row_bindings(row)["history_sha256"],
        "seed_id": seed_id,
        "state_sha256": license_mixed_row_bindings(row)["state_sha256"],
        "context_sha256": row.get("context_sha256"),
        "objective_fixture_sha256": fixture_sha256,
        "history_order": row.get("history_order"),
    }
    if any(candidate_provenance.get(key) != value for key, value in file_bindings.items()):
        raise ValueError("LICENSE-MIXED fixed-history row differs from its source manifest")
    for reference_key, metadata_path, metadata_sha, metadata_bytes in (
        (
            "source_artifact",
            "source_artifact_path",
            "source_artifact_sha256",
            "source_artifact_bytes",
        ),
        (
            "path_license_artifact",
            "path_license_artifact_path",
            "path_license_artifact_sha256",
            "path_license_artifact_bytes",
        ),
        (
            "license_scope_artifact",
            "license_scope_artifact_path",
            "license_scope_sha256",
            "license_scope_artifact_bytes",
        ),
    ):
        reference = candidate_provenance.get(reference_key)
        if not isinstance(reference, Mapping) or any(
            metadata.get(field) != reference.get(ref_field)
            for field, ref_field in (
                (metadata_path, "path"),
                (metadata_sha, "sha256"),
                (metadata_bytes, "bytes"),
            )
        ):
            raise ValueError("LICENSE-MIXED fixed-history artifact differs from source manifest")
    return fixture_sha256


def _context_action_bindings(
    row: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    try:
        state = EditState.from_mapping(row["state"])
        action = EditAction(**row["action"])
        context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
        response = encode_action(action)
        response_tokens = len(tokenizer.encode(response, add_special_tokens=False)) + 1
        eos_id = tokenizer.eos_token_id
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("LICENSE-MIXED tokenizer/state/action contract is invalid") from None
    context_sha256 = _sha256(context.text.encode("utf-8"))
    if (
        context.input_tokens is None
        or context.input_tokens > 1024
        or context.input_tokens + response_tokens > 2048
        or response_tokens > MAX_ACTION_TOKENS
        or not isinstance(eos_id, int)
        or eos_id < 0
        or ("prompt" in row and row["prompt"] != context.text)
        or row.get("context_sha256") != context_sha256
        or row.get("context_policy", CONTEXT_POLICY_VERSION) != CONTEXT_POLICY_VERSION
    ):
        raise ValueError("LICENSE-MIXED row differs from frozen q25 context/action contract")
    return {
        **license_mixed_row_bindings(row),
        "context_sha256": context_sha256,
        "input_tokens": context.input_tokens,
        "action_tokens_including_eos": response_tokens,
        "context_includes_request": bool(context.included_relevant),
    }


def _verified_oracle_record(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifact_root: Path,
    expected_fixture_sha256: str | None,
    expected_fixture_artifact: Mapping[str, Any] | None,
    expected_runtime_sha256: str | None,
    fixture_task: Mapping[str, Any] | None,
    fixture_objective: Mapping[str, Any] | None,
    diagnostics: Mapping[tuple[str, str], tuple[Mapping[str, Any], str]],
) -> dict[str, str]:
    identifier = _license_mixed_id(row)
    bindings = license_mixed_row_bindings(row)
    state = EditState.from_mapping(row["state"])
    action = EditAction(**row["action"])
    required_record_keys = {
        "candidate_id",
        "seed_id",
        "split",
        "source_group_id",
        "task_family_id",
        "template_id",
        "source_type",
        "state_sha256",
        "action_sha256",
        "context_sha256",
        "history_sha256",
        "fixture_sha256",
        "fixture_artifact_path",
        "fixture_artifact_sha256",
        "fixture_artifact_bytes",
        "evaluator_sha256",
        "runtime_sha256",
        "execution_backend",
        "network_access",
        "human_chronology_observed",
        "variants",
    }
    if set(record) != required_record_keys:
        raise ValueError("LICENSE-MIXED oracle result has an unknown or incomplete schema")
    expected_identity = {
        "candidate_id": identifier,
        "split": row.get("split"),
        "source_group_id": row.get("source_group_id"),
        "task_family_id": row.get("task_family_id"),
        "template_id": row.get("template_id"),
        "source_type": row.get("source_type"),
        "seed_id": row.get("seed_id"),
        "state_sha256": bindings["state_sha256"],
        "action_sha256": bindings["action_sha256"],
        "context_sha256": row.get("context_sha256"),
        "history_sha256": bindings["history_sha256"],
        "evaluator_sha256": manifest.get("oracle_evaluator_sha256"),
        "execution_backend": "container",
        "network_access": "none",
        "human_chronology_observed": False,
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("LICENSE-MIXED oracle result does not bind to its training row")
    if record.get("runtime_sha256") != expected_runtime_sha256:
        raise ValueError("LICENSE-MIXED oracle result runtime differs from the pinned fixture")
    fixture_sha256 = record.get("fixture_sha256")
    if (
        not isinstance(fixture_sha256, str)
        or _SHA256.fullmatch(fixture_sha256) is None
        or (expected_fixture_sha256 is not None and fixture_sha256 != expected_fixture_sha256)
    ):
        raise ValueError("LICENSE-MIXED oracle fixture binding is missing or inconsistent")
    fixture_path = record.get("fixture_artifact_path")
    fixture_sha = record.get("fixture_artifact_sha256")
    fixture_bytes = record.get("fixture_artifact_bytes")
    if (
        not isinstance(fixture_sha, str)
        or _SHA256.fullmatch(fixture_sha) is None
        or type(fixture_bytes) is not int
        or fixture_bytes < 1
        or not isinstance(expected_fixture_artifact, Mapping)
        or fixture_path != expected_fixture_artifact.get("path")
        or fixture_sha != expected_fixture_artifact.get("sha256")
        or fixture_bytes != expected_fixture_artifact.get("bytes")
        or record.get("runtime_sha256") != expected_runtime_sha256
    ):
        raise ValueError("LICENSE-MIXED oracle fixture artifact is unpinned")
    _pinned_artifact_bytes(
        artifact_root,
        fixture_path,
        fixture_sha,
        fixture_bytes,
        label="objective fixture",
    )
    variants = record.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("LICENSE-MIXED oracle variants are missing")
    if not isinstance(fixture_task, Mapping):
        raise ValueError("LICENSE-MIXED frozen synthetic task is absent")
    if not isinstance(fixture_objective, Mapping):
        raise ValueError("LICENSE-MIXED objective fixture is not available for status checks")
    compile_configured = bool(fixture_objective.get("compile"))
    test_configured = bool(fixture_objective.get("test"))
    expected_variant_names = {"gold", "behavior_breaking"}
    if action.kind != "keep":
        expected_variant_names.add("before")
    if set(variants) != expected_variant_names:
        raise ValueError("LICENSE-MIXED oracle does not test the required controls")
    normalized: dict[str, Mapping[str, Any]] = {}
    variant_keys = {
        "action_sha256",
        "after_source_sha256",
        "functional_expected",
        "functional_status",
        "parse_status",
        "compile_status",
        "test_status",
        "working_tree_sha256",
        "case_id",
        "case_attempt_id",
        "request_id",
        "diagnostic_sha256",
    }
    allowed_check_statuses = {
        "pass",
        "fail",
        "timeout",
        "error",
        "unavailable",
        "not_run",
    }
    for name, variant in variants.items():
        if not isinstance(variant, Mapping) or set(variant) != variant_keys:
            raise ValueError("LICENSE-MIXED oracle variant is incomplete")
        if (
            not isinstance(variant.get("action_sha256"), str)
            or _SHA256.fullmatch(variant["action_sha256"]) is None
            or not isinstance(variant.get("after_source_sha256"), str)
            or _SHA256.fullmatch(variant["after_source_sha256"]) is None
            or not isinstance(variant.get("working_tree_sha256"), str)
            or _SHA256.fullmatch(variant["working_tree_sha256"]) is None
            or not isinstance(variant.get("diagnostic_sha256"), str)
            or _SHA256.fullmatch(variant["diagnostic_sha256"]) is None
            or not isinstance(variant.get("case_id"), str)
            or not variant["case_id"]
            or not isinstance(variant.get("case_attempt_id"), str)
            or not variant["case_attempt_id"]
            or not isinstance(variant.get("request_id"), str)
            or not variant["request_id"]
            or not isinstance(variant.get("functional_expected"), bool)
            or variant.get("functional_status") not in {"pass", "fail", "unknown"}
            or any(variant.get(key) not in allowed_check_statuses for key in (
                "parse_status",
                "compile_status",
                "test_status",
            ))
        ):
            raise ValueError("LICENSE-MIXED oracle variant identity or status is invalid")
        normalized[str(name)] = variant
        diagnostic_record = diagnostics.get((str(identifier), str(name)))
        if diagnostic_record is None:
            raise ValueError("LICENSE-MIXED oracle result lacks its diagnostic bytes")
        diagnostic, diagnostic_sha256 = diagnostic_record
        required_diagnostic_keys = {
            "candidate_id",
            "variant",
            "case_id",
            "case_attempt_id",
            "request_id",
            "parse_status",
            "compile_status",
            "test_status",
            "compile_configured",
            "test_configured",
            "working_tree_sha256",
            "checks",
        }
        checks = diagnostic.get("checks")
        if (
            set(diagnostic) != required_diagnostic_keys
            or not isinstance(checks, Mapping)
            or set(checks) != {"parse", "compile", "test"}
            or diagnostic.get("compile_configured") is not compile_configured
            or diagnostic.get("test_configured") is not test_configured
        ):
            raise ValueError("LICENSE-MIXED oracle diagnostic schema or fixture binding is invalid")
        for check_name in ("parse", "compile", "test"):
            check = checks[check_name]
            if (
                not isinstance(check, Mapping)
                or set(check) != {"status", "returncode", "elapsed_seconds", "stdout", "stderr"}
                or check.get("status") != diagnostic.get(check_name + "_status")
                or not (
                    check.get("returncode") is None
                    or (type(check.get("returncode")) is int)
                )
                or not _valid_elapsed_seconds(check.get("elapsed_seconds"))
                or not isinstance(check.get("stdout"), str)
                or len(check["stdout"].encode("utf-8")) > 64 * 1024
                or not isinstance(check.get("stderr"), str)
                or len(check["stderr"].encode("utf-8")) > 64 * 1024
            ):
                raise ValueError("LICENSE-MIXED oracle diagnostic check details are invalid")
        diagnostic_fields = (
            "case_id",
            "case_attempt_id",
            "request_id",
            "parse_status",
            "compile_status",
            "test_status",
            "working_tree_sha256",
        )
        if (
            variant["diagnostic_sha256"] != diagnostic_sha256
            or any(diagnostic.get(key) != variant.get(key) for key in diagnostic_fields)
        ):
            raise ValueError("LICENSE-MIXED oracle summary differs from bound execution details")

    def derived_functional_status(variant: Mapping[str, Any]) -> str:
        parse_status = variant["parse_status"]
        compile_status = variant["compile_status"]
        test_status = variant["test_status"]
        if not compile_configured and compile_status != "not_run":
            raise ValueError("LICENSE-MIXED unconfigured compile check was executed")
        if not test_configured and test_status != "not_run":
            raise ValueError("LICENSE-MIXED unconfigured test check was executed")
        checked_statuses = [parse_status]
        if compile_configured:
            checked_statuses.append(compile_status)
        if test_configured:
            checked_statuses.append(test_status)
        if any(status in {"timeout", "error", "unavailable"} for status in checked_statuses):
            return "unknown"
        if parse_status == "fail" or (compile_configured and compile_status == "fail"):
            return "fail"
        if test_configured and test_status == "fail":
            return "fail"
        if parse_status == "not_run" or (
            compile_configured and compile_status == "not_run"
        ) or (test_configured and test_status == "not_run"):
            return "unknown"
        return "pass"

    if any(
        derived_functional_status(variant) != variant["functional_status"]
        for variant in normalized.values()
    ):
        raise ValueError("LICENSE-MIXED functional status does not match evaluator checks")

    gold = normalized["gold"]
    if (
        gold["action_sha256"] != bindings["action_sha256"]
        or gold["after_source_sha256"] != bindings["after_source_sha256"]
        or gold["functional_expected"] is not True
        or gold["functional_status"] != "pass"
        or gold["parse_status"] != "pass"
        or gold["compile_status"] not in {"pass", "not_run"}
        or gold["test_status"] not in {"pass", "not_run"}
    ):
        raise ValueError("LICENSE-MIXED positive action lacks a functional objective pass")
    bad = normalized["behavior_breaking"]
    try:
        expected_bad_action = EditAction(**fixture_task["wrong_action"])
        expected_bad_after = apply_action(state, expected_bad_action)
        fixture_action = EditAction(**fixture_task["action"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("LICENSE-MIXED frozen control action is invalid") from None
    if (
        fixture_action != action
        or bad["action_sha256"]
        != _sha256(_canonical_bytes(asdict(expected_bad_action)))
        or bad["after_source_sha256"] != _sha256(expected_bad_after.encode("utf-8"))
        or expected_bad_after == state.source
        or bad["action_sha256"] == gold["action_sha256"]
        or bad["after_source_sha256"] == gold["after_source_sha256"]
        or bad["working_tree_sha256"] == gold["working_tree_sha256"]
        or bad["functional_expected"] is not False
        or bad["functional_status"] != "fail"
        or bad["parse_status"] != "pass"
        or bad["compile_status"] != ("pass" if compile_configured else "not_run")
        or not test_configured
        or bad["test_status"] != "fail"
    ):
        raise ValueError("LICENSE-MIXED behavior-breaking control did not fail functionally")
    if action.kind == "keep":
        source_type = row.get("source_type")
        if source_type == REVIEWED_PUBLIC_HISTORY_SOURCE:
            if (
                row.get("history_order") != "synthetic_fixed_before_provider"
                or not state.history
                or state.relevant
            ):
                raise ValueError("LICENSE-MIXED fixed-history N row lacks a visible history")
        elif source_type != "synthetic_public_source_task":
            raise ValueError("Muse-authored no-edit rows lack independent decision evidence")
        if source_type == "synthetic_public_source_task" and (
            not state.relevant
            or not state.relevant[0].strip()
            or row.get("visible_request_location") != "state.relevant[0]"
        ):
            raise ValueError("synthetic N row lacks a visible explicit request")
    else:
        before = normalized["before"]
        expected_before = EditAction("keep")
        if (
            before["action_sha256"] != _sha256(_canonical_bytes(asdict(expected_before)))
            or before["after_source_sha256"] != _sha256(state.source.encode("utf-8"))
            or before["functional_expected"] is not False
            or before["functional_status"] != "fail"
            or before["parse_status"] != "pass"
            or not (
                (
                    before["compile_status"] == "fail"
                    and before["test_status"] == "not_run"
                )
                or (
                    before["compile_status"] in {"pass", "not_run"}
                    and before["test_status"] == "fail"
                )
            )
        ):
            raise ValueError("LICENSE-MIXED edit lacks a failing prestate no-op control")
    return {
        "fixture_sha256": fixture_sha256,
        "evaluator_sha256": str(record["evaluator_sha256"]),
        "runtime_sha256": str(record["runtime_sha256"]),
    }


def _runtime_sha256(fixture: Mapping[str, Any]) -> str:
    oracle = fixture.get("oracle")
    image = oracle.get("container_image") if isinstance(oracle, Mapping) else None
    if not isinstance(image, str) or re.fullmatch(
        r"[^\s]+@sha256:[a-f0-9]{64}", image
    ) is None:
        raise ValueError("LICENSE-MIXED objective fixture runtime is not digest pinned")
    descriptor = {
        "kind": "sandbox_container",
        "identity": image,
        "identity_sha256": _sha256(image.encode("utf-8")),
    }
    return _sha256(_canonical_bytes(descriptor))


def _license_mixed_review_split_audit(
    entries: list[dict[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    ids: set[str] = set()
    states: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    group_fields = (
        "source_group_id",
        "source_repo",
        "session_or_commit",
        "task_family_id",
        "near_duplicate_sha256",
    )
    for entry in entries:
        identifier, split = entry.get("candidate_id"), entry.get("split")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in ids
            or split not in {"train", "development", "test_new_repo", "test_new_mechanism"}
        ):
            raise ValueError("LICENSE-MIXED review candidate identity or split is invalid")
        ids.add(identifier)
        counts[str(split)] += 1
        state_sha = entry.get("state_sha256")
        near_sha = entry.get("near_duplicate_sha256")
        if (
            not isinstance(state_sha, str)
            or _SHA256.fullmatch(state_sha) is None
            or state_sha in states
            or not isinstance(near_sha, str)
            or _SHA256.fullmatch(near_sha) is None
        ):
            raise ValueError("LICENSE-MIXED review has duplicate state or invalid hash")
        states.add(state_sha)
        for field in group_fields:
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("LICENSE-MIXED review grouping identity is missing: " + field)
            normalized = value.casefold() if field == "source_repo" else value
            prior = groups.setdefault((field, normalized), str(split))
            if prior != split:
                raise ValueError("LICENSE-MIXED review group crosses splits: " + field)
    if dict(sorted(counts.items())) != manifest.get("candidate_split_counts"):
        raise ValueError("LICENSE-MIXED full split review counts disagree with the manifest")
    return {
        "rows": len(entries),
        "splits": dict(sorted(counts.items())),
        "group_fields": list(group_fields),
        "state_duplicates": 0,
    }


def _row_source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("authoring_metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "source_repo": row.get("source_repo", metadata.get("source_repo")),
        "source_revision": row.get("source_revision", metadata.get("source_revision")),
        "source_tree_sha": row.get("source_tree_sha", metadata.get("source_tree_sha")),
        "source_path": row.get("source_path", metadata.get("source_path")),
        "source_sha256": row.get("source_sha256", metadata.get("source_sha256")),
        "source_license": row.get("source_license", metadata.get("source_license")),
        "source_license_sha256": row.get(
            "source_license_sha256",
            metadata.get("path_license_sha256", metadata.get("license_sha256")),
        ),
    }


def license_mixed_artifact_root(manifest: Mapping[str, Any], data_root: Path) -> Path:
    relative = Path(str(manifest.get("artifact_root", ".")))
    candidate = data_root / relative
    try:
        if data_root.is_symlink():
            raise ValueError("LICENSE-MIXED package root is a symlink")
        root = data_root.resolve(strict=True)
        current = data_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("LICENSE-MIXED artifact root contains a symlink")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_dir() or candidate.is_symlink():
            raise ValueError("LICENSE-MIXED artifact root is outside the frozen package")
        return resolved
    except OSError:
        raise ValueError("LICENSE-MIXED artifact root is unavailable") from None


def validate_license_mixed_review(
    manifest: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    tokenizer: Any,
    package_root: Path,
    policy: PilotDataPolicy = LICENSE_MIXED,
) -> dict[str, Any]:
    """Recheck portable per-row proofs and the full review split audit.

    This path reads only frozen artifacts and status records. It performs no
    sandbox, compilation, provider, or network work.
    """
    if policy not in {LICENSE_MIXED, LICENSE_MIXED_HISTORY}:
        raise ValueError("unapproved LICENSE-MIXED review policy")
    validate_license_mixed_manifest(manifest, policy=policy)
    review_payload = _read_relative_artifact(
        package_root,
        manifest.get("independent_review_path"),
        label="independent review",
        expected_sha256=str(manifest["independent_review_sha256"]),
        expected_bytes=int(manifest["independent_review_bytes"]),
    )
    review = _strict_json(review_payload, label="independent review")
    if (
        not isinstance(review, dict)
        or review.get("schema")
        != (
            "one-line-license-mixed-history-independent-review-v2"
            if policy is LICENSE_MIXED_HISTORY
            else "one-line-license-mixed-independent-review-v1"
        )
        or review.get("full_split_audit_complete") is not True
        or not isinstance(review.get("independent_reviewer_id"), str)
        or not review["independent_reviewer_id"]
        or not isinstance(review.get("rows"), list)
    ):
        raise ValueError("LICENSE-MIXED independent review identity or audit is incomplete")
    review_entries = review["rows"]
    if any(not isinstance(entry, dict) for entry in review_entries):
        raise ValueError("LICENSE-MIXED independent review contains an invalid row")
    allowed_review_types = (
        {REVIEWED_PUBLIC_HISTORY_SOURCE}
        if policy is LICENSE_MIXED_HISTORY
        else PUBLIC_SOURCE_TYPES
    )
    if any(entry.get("source_type") not in allowed_review_types for entry in review_entries):
        raise ValueError("LICENSE-MIXED review contains a source type outside its schema")
    audit = _license_mixed_review_split_audit(review_entries, manifest)
    for entry in review_entries:
        if entry.get("split") in {"train", "development"}:
            if type(entry.get("accepted_training")) is not bool:
                raise ValueError("LICENSE-MIXED candidate lacks an explicit acceptance disposition")
        elif entry.get("accepted_training") is not False:
            raise ValueError("LICENSE-MIXED reserved candidate was marked for training")
    approved_ids = {
        entry["candidate_id"]
        for entry in review_entries
        if entry.get("split") in {"train", "development"}
        and entry.get("accepted_training") is True
    }
    row_ids = {_license_mixed_id(row) for row in rows}
    if None in row_ids or approved_ids != row_ids:
        raise ValueError("LICENSE-MIXED train/development shards differ from reviewed IDs")
    entries = {str(entry["candidate_id"]): entry for entry in review_entries}
    counts = Counter(str(row.get("split")) for row in rows)
    if counts["train"] != manifest.get("train_count") or counts["development"] != manifest.get(
        "dev_count"
    ):
        raise ValueError("LICENSE-MIXED data shard counts differ from its manifest")
    split_audit = validate_license_mixed_splits(list(rows))
    artifact_root = license_mixed_artifact_root(manifest, package_root)
    provenance = _license_mixed_provenance(manifest, artifact_root)
    has_synthetic = any(row.get("source_type") == "synthetic_public_source_task" for row in rows)
    has_public_history = any(
        row.get("source_type") == REVIEWED_PUBLIC_HISTORY_SOURCE for row in rows
    )
    oracle_keys = (
        "oracle_results_path",
        "oracle_results_sha256",
        "oracle_results_bytes",
        "oracle_evaluator_sha256",
        "oracle_diagnostics_path",
        "oracle_diagnostics_sha256",
        "oracle_diagnostics_bytes",
    )
    has_oracle = all(key in manifest for key in oracle_keys)
    if has_synthetic and (not has_oracle or provenance is None):
        raise ValueError("LICENSE-MIXED synthetic rows lack bound objective/source artifacts")
    if has_public_history and (
        policy is not LICENSE_MIXED_HISTORY or not has_oracle or provenance is None
    ):
        raise ValueError("LICENSE-MIXED fixed-history rows lack their v2 objective/source package")
    oracle_records = _oracle_results(manifest, artifact_root) if has_oracle else {}
    diagnostics = _oracle_diagnostics(manifest, artifact_root) if has_oracle else {}
    expected_audit_ids = set(oracle_records)
    if (has_synthetic or has_public_history) and not {
        str(_license_mixed_id(row))
        for row in rows
        if row.get("source_type")
        in {"synthetic_public_source_task", REVIEWED_PUBLIC_HISTORY_SOURCE}
    }.issubset(expected_audit_ids):
        raise ValueError("LICENSE-MIXED synthetic training rows lack oracle records")
    used_diagnostics: set[tuple[str, str]] = set()
    if has_oracle:
        from tinycomplete.eval.code_benchmark import evaluate_prediction

        evaluator_path = Path(evaluate_prediction.__code__.co_filename).resolve()
        if _sha256(evaluator_path.read_bytes()) != manifest.get("oracle_evaluator_sha256"):
            raise ValueError("LICENSE-MIXED evaluator source differs from its pinned identity")
        for identifier, (record, record_sha256) in oracle_records.items():
            entry = entries.get(identifier)
            if entry is None or entry.get("oracle_result_sha256") != record_sha256:
                raise ValueError("LICENSE-MIXED full review omits an oracle result binding")
            if any(
                entry.get(key) != record.get(key)
                for key in (
                    "candidate_id",
                    "split",
                    "source_group_id",
                    "task_family_id",
                    "template_id",
                    "source_type",
                    "state_sha256",
                    "action_sha256",
                    "context_sha256",
                    "history_sha256",
                    "fixture_sha256",
                    "evaluator_sha256",
                    "runtime_sha256",
                )
            ):
                raise ValueError("LICENSE-MIXED full review disagrees with oracle row identity")
            variants = record.get("variants")
            if not isinstance(variants, Mapping):
                raise ValueError("LICENSE-MIXED oracle variants are missing from full review")
            for name, variant in variants.items():
                if not isinstance(variant, Mapping):
                    raise ValueError("LICENSE-MIXED oracle variant is invalid")
                diagnostic_entry = diagnostics.get((identifier, str(name)))
                if diagnostic_entry is None:
                    raise ValueError("LICENSE-MIXED full review lacks an oracle diagnostic")
                diagnostic, diagnostic_sha = diagnostic_entry
                if (
                    diagnostic_sha != variant.get("diagnostic_sha256")
                    or any(
                        diagnostic.get(key) != variant.get(key)
                        for key in (
                            "case_id",
                            "case_attempt_id",
                            "request_id",
                            "parse_status",
                            "compile_status",
                            "test_status",
                            "working_tree_sha256",
                        )
                    )
                ):
                    raise ValueError("LICENSE-MIXED full review diagnostic binding is invalid")
                used_diagnostics.add((identifier, str(name)))
    accepted_muse_rows: list[Mapping[str, Any]] = []
    reserved_repositories = frozenset(
        str(entry["source_repo"])
        for entry in review_entries
        if entry.get("split") in {"test_new_repo", "test_new_mechanism"}
    )
    verified_rows = 0
    for row in rows:
        row_id = _license_mixed_id(row)
        if row_id is None:
            raise ValueError("LICENSE-MIXED row has no candidate identity")
        entry = entries.get(row_id)
        if entry is None:
            raise ValueError("LICENSE-MIXED row is absent from independent review")
        bindings = _context_action_bindings(row, tokenizer)
        source_identity = _row_source_identity(row)
        source_type = row.get("source_type")
        if source_type not in allowed_review_types:
            raise ValueError("LICENSE-MIXED row source type is outside its policy schema")
        expected = {
            "candidate_id": row_id,
            "split": row.get("split"),
            "source_type": source_type,
            "source_group_id": row.get("source_group_id"),
            "source_repo": source_identity["source_repo"],
            "source_revision": source_identity["source_revision"],
            "source_path": source_identity["source_path"],
            "source_sha256": source_identity["source_sha256"],
            "source_tree_sha": source_identity["source_tree_sha"],
            "source_license": source_identity["source_license"],
            "source_license_sha256": source_identity["source_license_sha256"],
            "session_or_commit": row.get("session_or_commit"),
            "task_family_id": row.get("task_family_id"),
            "template_id": row.get("template_id"),
            "state_sha256": bindings["state_sha256"],
            "action_sha256": bindings["action_sha256"],
            "after_source_sha256": bindings["after_source_sha256"],
            "history_sha256": bindings["history_sha256"],
            "near_duplicate_sha256": bindings["near_duplicate_sha256"],
            "context_sha256": bindings["context_sha256"],
            "human_chronology_observed": False,
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("LICENSE-MIXED row differs from independently reviewed bindings")
        for field in (
            "accepted_training",
            "inferability_reviewed",
            "objective_verified",
            "wrong_action_controls_rejected",
            "history_leakage_check",
        ):
            if entry.get(field) is not True:
                raise ValueError("LICENSE-MIXED independent review lacks approval: " + field)
        if entry.get("independent_reviewer_id") != review["independent_reviewer_id"]:
            raise ValueError("LICENSE-MIXED row reviewer identity differs from review manifest")
        if source_type == "synthetic_public_source_task":
            if not bindings["context_includes_request"]:
                raise ValueError(
                    "LICENSE-MIXED synthetic request is omitted by the bounded context"
                )
            if provenance is None:
                raise ValueError("LICENSE-MIXED synthetic source provenance is absent")
            artifact_bindings = validate_license_mixed_source_artifacts(
                row, package_root=artifact_root
            )
            if any(entry.get(key) != value for key, value in artifact_bindings.items()):
                raise ValueError("LICENSE-MIXED review differs from per-file source artifacts")
            record_tuple = oracle_records.get(row_id)
            if record_tuple is None:
                raise ValueError("LICENSE-MIXED synthetic row has no objective result")
            record, result_sha256 = record_tuple
            fixture_sha256 = _validate_synthetic_provenance_row(row, record, provenance)
            fixture = provenance["fixture_by_seed"][str(row["seed_id"])]
            fixture_artifact = provenance["fixture_ref"]
            fixture_oracle = fixture["oracle"]
            proof = _verified_oracle_record(
                row,
                record,
                manifest=manifest,
                artifact_root=artifact_root,
                expected_fixture_sha256=fixture_sha256,
                expected_fixture_artifact=fixture_artifact,
                expected_runtime_sha256=_runtime_sha256(fixture),
                fixture_task=fixture,
                fixture_objective=fixture_oracle,
                diagnostics=diagnostics,
            )
            if (
                entry.get("oracle_result_sha256") != result_sha256
                or entry.get("fixture_sha256") != proof["fixture_sha256"]
                or entry.get("evaluator_sha256") != proof["evaluator_sha256"]
                or entry.get("runtime_sha256") != proof["runtime_sha256"]
            ):
                raise ValueError("LICENSE-MIXED review does not bind the actual oracle result")
            used_diagnostics.update((row_id, name) for name in record["variants"])
            role_path = entry.get("role_evidence_artifact_path")
            role_sha = entry.get("role_evidence_sha256")
            role_bytes = entry.get("role_evidence_bytes")
            if (
                not isinstance(role_path, str)
                or not isinstance(role_sha, str)
                or _SHA256.fullmatch(role_sha) is None
                or type(role_bytes) is not int
                or role_bytes < 1
            ):
                raise ValueError("LICENSE-MIXED synthetic role evidence is unpinned")
            role_payload = _read_relative_artifact(
                artifact_root,
                role_path,
                label="synthetic role evidence",
                expected_sha256=role_sha,
                expected_bytes=role_bytes,
            )
            candidate = dict(row)
            candidate.setdefault("id", row_id)
            from .muse_acceptance_package import verify_role_evidence

            role_decision = verify_role_evidence(candidate, role_payload, tokenizer=tokenizer)
            if (
                not role_decision.accepted
                or role_decision.evidence.get("role_evidence_sha256") != role_sha
            ):
                raise ValueError("LICENSE-MIXED synthetic row lacks verified blind role evidence")
        elif source_type == REVIEWED_PUBLIC_HISTORY_SOURCE:
            if policy is not LICENSE_MIXED_HISTORY:
                raise ValueError("LICENSE-MIXED v1 cannot accept fixed-history rows")
            state = EditState.from_mapping(row["state"])
            if (
                row.get("history_order") != "synthetic_fixed_before_provider"
                or state.relevant
                or not state.history
                or provenance is None
            ):
                raise ValueError("LICENSE-MIXED fixed-history row declaration is invalid")
            artifact_bindings = validate_license_mixed_source_artifacts(
                row, package_root=artifact_root
            )
            if any(entry.get(key) != value for key, value in artifact_bindings.items()):
                raise ValueError("LICENSE-MIXED review differs from fixed-history source artifacts")
            record_tuple = oracle_records.get(row_id)
            if record_tuple is None:
                raise ValueError("LICENSE-MIXED fixed-history row has no objective result")
            record, result_sha256 = record_tuple
            fixture_sha256 = _validate_public_history_provenance_row(
                row, record, provenance
            )
            fixture = provenance["fixture_by_seed"][str(row["seed_id"])]
            proof = _verified_oracle_record(
                row,
                record,
                manifest=manifest,
                artifact_root=artifact_root,
                expected_fixture_sha256=fixture_sha256,
                expected_fixture_artifact=provenance["fixture_ref"],
                expected_runtime_sha256=_runtime_sha256(fixture),
                fixture_task=fixture,
                fixture_objective=fixture["oracle"],
                diagnostics=diagnostics,
            )
            if (
                entry.get("oracle_result_sha256") != result_sha256
                or entry.get("fixture_sha256") != proof["fixture_sha256"]
                or entry.get("evaluator_sha256") != proof["evaluator_sha256"]
                or entry.get("runtime_sha256") != proof["runtime_sha256"]
                or entry.get("history_order") != row.get("history_order")
            ):
                raise ValueError("LICENSE-MIXED review does not bind fixed-history oracle evidence")
            role_path = entry.get("role_evidence_artifact_path")
            role_sha = entry.get("role_evidence_sha256")
            role_bytes = entry.get("role_evidence_bytes")
            if (
                not isinstance(role_path, str)
                or not isinstance(role_sha, str)
                or _SHA256.fullmatch(role_sha) is None
                or type(role_bytes) is not int
                or role_bytes < 1
            ):
                raise ValueError("LICENSE-MIXED fixed-history role evidence is unpinned")
            role_proof_ref = row.get("role_execution_proof_ref")
            if not isinstance(role_proof_ref, Mapping):
                raise ValueError("LICENSE-MIXED fixed-history runner proof is unpinned")
            if entry.get("role_execution_proof_ref") != role_proof_ref:
                raise ValueError("LICENSE-MIXED review differs from fixed-history runner proof")
            proof_role_path = (
                str(role_proof_ref.get("root", ""))
                + "/"
                + str(role_proof_ref.get("role_evidence_path", ""))
            )
            if (
                role_path != proof_role_path
                or role_sha != role_proof_ref.get("role_evidence_sha256")
                or role_bytes != role_proof_ref.get("role_evidence_bytes")
            ):
                raise ValueError(
                    "LICENSE-MIXED fixed-history role evidence differs from runner proof"
                )
            from .fixed_state_role_receipts import verify_fixed_state_role_execution_proof

            role_decision = verify_fixed_state_role_execution_proof(
                row, role_proof_ref, package_root=package_root, tokenizer=tokenizer
            )
            if (
                not role_decision.accepted
                or role_decision.evidence.get("role_evidence_sha256") != role_sha
                or role_decision.evidence.get("functional_status")
                != "not_evaluated_by_role_runner"
            ):
                raise ValueError(
                    "LICENSE-MIXED fixed-history runner proof failed portable verification"
                )
            used_diagnostics.update((row_id, name) for name in record["variants"])
        elif source_type == "muse_author_public_candidate":
            if row.get("id") != row_id:
                raise ValueError("LICENSE-MIXED Muse row must preserve its package candidate ID")
            if row.get("source_license_sha256") != source_identity["source_license_sha256"]:
                raise ValueError("LICENSE-MIXED Muse source license hash is inconsistent")
            from .muse_acceptance_package import verify_muse_candidate_row

            package_decision = verify_muse_candidate_row(
                row,
                package_root=package_root,
                tokenizer=tokenizer,
                existing_rows=accepted_muse_rows,
                reserved_repositories=reserved_repositories,
            )
            if not package_decision.accepted:
                raise ValueError("LICENSE-MIXED Muse acceptance package is not valid")
            if entry.get("acceptance_package_ref") != row.get("acceptance_package_ref"):
                raise ValueError("LICENSE-MIXED review differs from Muse acceptance package")
            accepted_muse_rows.append(row)
        else:
            raise ValueError("LICENSE-MIXED row source type is not allowed")
        verified_rows += 1
    if has_oracle and used_diagnostics != set(diagnostics):
        raise ValueError("LICENSE-MIXED oracle diagnostics contain unbound or unused rows")
    return {
        "verified_rows": verified_rows,
        "split_audit": split_audit,
        "full_review_audit": audit,
        "oracle_result_count": len(oracle_records),
        "diagnostic_count": len(diagnostics),
        "quality_evidence": False,
    }


def validate_pilot_row(
    row: Mapping[str, Any], policy: PilotDataPolicy, *, package_root: Path | None = None
) -> None:
    if policy in {LICENSE_MIXED, LICENSE_MIXED_HISTORY}:
        validate_license_mixed_row(row, package_root=package_root, policy=policy)
        return
    if row.get("source_type") != policy.source_type:
        raise ValueError("pilot shard contains an unapproved source type")
    validation = row.get("validation", {})
    if not isinstance(validation, Mapping) or validation.get("replay_verified") is not True:
        raise ValueError("pilot row is not replay-verified")
    if policy == INSTINCT:
        return
    if row.get("source_license") != policy.dataset_license:
        raise ValueError("constructive pilot source license mismatch")
    for key in (
        "history_replays_to_state",
        "apply_reconstructs_after",
        "objective_verified",
        "inferability_reviewed",
        "accepted_training",
        "wrong_action_controls_rejected",
        "history_leakage_check",
        "after_objective_passes",
    ):
        if validation.get(key) is not True:
            raise ValueError("constructive pilot row lacks independent validation: " + key)
    for key in ("source_group_id", "template_id", "independent_reviewer_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError("constructive pilot identity is missing: " + key)
    if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("oracle_sha256", ""))):
        raise ValueError("constructive pilot oracle is not hash-pinned")
    action = row.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("constructive pilot action is missing")
    before_passes = validation.get("before_objective_passes")
    if not isinstance(before_passes, bool) or before_passes != (action.get("kind") == "keep"):
        raise ValueError("constructive edit/keep objective control disagrees with the action")


def validate_license_mixed_row(
    row: Mapping[str, Any],
    *,
    package_root: Path | None = None,
    policy: PilotDataPolicy = LICENSE_MIXED,
) -> None:
    """Check the row shape; the pinned review and outcome artifacts gate use."""
    if package_root is None:
        raise ValueError("LICENSE-MIXED row validation requires an explicit package root")
    if policy not in {LICENSE_MIXED, LICENSE_MIXED_HISTORY}:
        raise ValueError("unapproved LICENSE-MIXED row policy")
    source_type = row.get("source_type")
    allowed_types = (
        {REVIEWED_PUBLIC_HISTORY_SOURCE}
        if policy is LICENSE_MIXED_HISTORY
        else PUBLIC_SOURCE_TYPES
    )
    if source_type not in allowed_types:
        raise ValueError("LICENSE-MIXED pilot contains an unapproved source type")
    if row.get("human_chronology_observed", row.get("chronology_observed")) is not False:
        raise ValueError("LICENSE-MIXED row claims unverified human chronology")
    try:
        state = EditState.from_mapping(row["state"])
        action = EditAction(**row["action"])
        after_source = row["after_source"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("LICENSE-MIXED state/action fields are invalid") from None
    if not isinstance(after_source, str):
        raise ValueError("LICENSE-MIXED row lacks its after-state")
    if apply_action(state, action) != after_source:
        raise ValueError("LICENSE-MIXED action does not reconstruct its after-state")
    source_license = row.get("source_license")
    metadata = row.get("authoring_metadata")
    if not isinstance(source_license, str):
        raise ValueError("LICENSE-MIXED row lacks its file-scoped license")
    if source_type == "synthetic_public_source_task":
        if (
            row.get("visible_request_location") != "state.relevant[0]"
            or not state.relevant
            or not state.relevant[0].strip()
        ):
            raise ValueError("LICENSE-MIXED synthetic row lacks its visible explicit request")
        if action.kind == "keep" and not state.relevant[0].strip():
            raise ValueError("synthetic keep requires a visible explicit request")
        if not isinstance(metadata, Mapping):
            raise ValueError("LICENSE-MIXED synthetic row lacks per-file metadata")
        validate_license_mixed_source_artifacts(row, package_root=package_root)
    elif source_type == REVIEWED_PUBLIC_HISTORY_SOURCE:
        if (
            policy is not LICENSE_MIXED_HISTORY
            or row.get("history_order") != "synthetic_fixed_before_provider"
            or not state.history
            or state.relevant
            or not isinstance(row.get("seed_id"), str)
            or not isinstance(metadata, Mapping)
        ):
            raise ValueError("LICENSE-MIXED fixed-history row declaration is invalid")
        if row.get("context_policy") != CONTEXT_POLICY_VERSION:
            raise ValueError("LICENSE-MIXED fixed-history row context policy is invalid")
        validate_license_mixed_source_artifacts(row, package_root=package_root)
    else:
        reference = row.get("acceptance_package_ref")
        from .muse_acceptance_package import PACKAGE_SCHEMA, PACKAGE_SCHEMA_V2

        if (
            not isinstance(reference, Mapping)
            or reference.get("schema") not in {PACKAGE_SCHEMA, PACKAGE_SCHEMA_V2}
            or not isinstance(reference.get("root"), str)
            or not reference["root"]
            or Path(reference["root"]).is_absolute()
            or ".." in Path(reference["root"]).parts
            or reference.get("candidate_id") != row.get("candidate_id", row.get("id"))
            or any(
                not isinstance(reference.get(key), str)
                or _SHA256.fullmatch(str(reference.get(key))) is None
                for key in ("manifest_sha256", "candidate_sha256", "source_row_sha256")
            )
        ):
            raise ValueError("LICENSE-MIXED Muse row lacks its pinned acceptance package")
        if action.kind == "keep":
            raise ValueError("Muse-authored no-edit rows have no observed decision evidence")
