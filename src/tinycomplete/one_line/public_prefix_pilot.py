"""Narrow policy helpers for licensed public source-prefix pilot rows.

This policy treats a reviewed parent file's exact current line as a
next-token/line-completion target after a synthetic typed prefix. It makes no
human-chronology or behavioral-quality claim. It is separate from the
license-mixed fixed-history policy, whose stronger review and oracle receipts
remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from tinycomplete.one_line.context import CONTEXT_POLICY_VERSION, serialize_state
from tinycomplete.one_line.contract import (
    EditAction,
    EditState,
    apply_action,
    physical_lines,
)

PUBLIC_PREFIX_PILOT_SCHEMA = "one-line-public-prefix-typing-pilot-v1"
PUBLIC_PREFIX_SOURCE_TYPE = "licensed_public_prefix_completion"
PUBLIC_PREFIX_HISTORY_ORDER = "synthetic_typed_return_prefix_v1"
PUBLIC_SYNTHETIC_PILOT_SCHEMA = "one-line-public-synthetic-functional-mix-v1"
PUBLIC_SYNTHETIC_PLAN_SCHEMA = "one-line-public-synthetic-functional-mix-plan-v1"
SYNTHETIC_FUNCTIONAL_SOURCE_TYPE = "author_owned_synthetic_functional"
SYNTHETIC_FUNCTIONAL_LICENSE = "author_owned_synthetic_not_published"
PUBLIC_PREFIX_SUPPORTED_LICENSES = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
)
PILOT_MINIMUM_TRAIN_ROWS = 128
PILOT_MINIMUM_DEVELOPMENT_ROWS = 64
MINIMUM_TRAIN_FAMILIES_PER_ACTION = 12
MINIMUM_DEVELOPMENT_FAMILIES_PER_ACTION = 4

_ACTION_FAMILY_BY_KIND = {
    "keep": "N",
    "insert_before": "I",
    "delete_line": "D",
    "replace_line": "R",
}
_PREFIX_TOKEN_NORMALIZER = re.compile(r"(?:\b\d+(?:\.\d+)?\b)|(?:\b[A-Za-z_$][\w$]*\b)")
_SHA256_PATTERN = re.compile(r"[a-f0-9]{64}\Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def action_family(action: Mapping[str, Any] | EditAction) -> str:
    kind = action.kind if isinstance(action, EditAction) else action.get("kind")
    try:
        return _ACTION_FAMILY_BY_KIND[str(kind)]
    except KeyError:
        raise ValueError("public-prefix pilot action kind is invalid") from None


def public_prefix_template_key(state: EditState) -> str:
    """Recompute the frozen prefix materializer's split-group near key."""
    lines = physical_lines(state.source.encode("utf-8"))
    excerpt = b"\n".join(
        line.content for line in lines[max(0, state.target_row - 5) : state.target_row + 1]
    )
    normalized = _PREFIX_TOKEN_NORMALIZER.sub("_", excerpt.decode("utf-8"))
    normalized = " ".join(normalized.split())
    return sha256_bytes((state.filetype + "\0" + normalized).encode("utf-8"))


def validate_public_prefix_transition(
    row: Mapping[str, Any],
    *,
    parent_source: bytes,
) -> dict[str, Any]:
    """Bind a typed return-prefix state and R label to the exact parent line.

    The source suffix is used only as the private target oracle. The serialized
    state must be the byte-exact parent prefix through the cursor, including
    the original line terminator and no bytes after the cursor on that line.
    """
    try:
        state = EditState.from_mapping(row["state"])
        raw_action = row["action"]
        if not isinstance(raw_action, Mapping):
            raise TypeError
        action = EditAction(**raw_action)
        prompt = row["prompt"]
        after_source = row["after_source"]
        history_before_sha = row["history_before_sha256"]
        target_line_sha = row["target_source_line_sha256"]
        context_sha = row["context_sha256"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("public-prefix state/action binding is invalid") from None

    from tinycomplete.one_line.contract import physical_lines

    source_lines = physical_lines(parent_source)
    state_lines = physical_lines(state.source.encode("utf-8"))
    target_row = state.target_row
    if target_row >= len(source_lines) or target_row >= len(state_lines):
        raise ValueError("public-prefix target row is absent from the parent source")
    parent_line = source_lines[target_row]
    state_line = state_lines[target_row]
    if parent_line.terminator != state_line.terminator:
        raise ValueError("public-prefix target line terminator differs from parent source")

    history = state.history
    if len(history) != 1:
        raise ValueError("public-prefix state must contain one typed-prefix edit")
    edit = history[0]
    prefix = edit.new_text.encode("utf-8")
    previous_prefix = edit.old_text.encode("utf-8")
    target_prefix = state_line.content
    if (
        edit.row != target_row
        or not prefix.startswith(previous_prefix)
        or target_prefix != prefix
        or state.cursor_col != len(prefix)
        or prefix.strip() != b"return"
        or not prefix.endswith(b" ")
        or state.filetype not in {"python", "go", "rust", "typescript"}
    ):
        raise ValueError("public-prefix history does not end at a typed return prefix")

    expected_state = (
        b"".join(line.raw for line in source_lines[:target_row])
        + parent_line.content[: state.cursor_col]
        + parent_line.terminator
    )
    if state.source.encode("utf-8") != expected_state:
        raise ValueError("public-prefix model input is not the exact parent prefix")
    if not parent_line.content.startswith(prefix):
        raise ValueError("public-prefix answer does not begin with the typed bytes")

    before_source = (
        b"".join(line.raw for line in state_lines[:target_row])
        + previous_prefix
        + state_line.terminator
    )
    if sha256_bytes(before_source) != history_before_sha:
        raise ValueError("public-prefix synthetic history does not replay from its before state")

    answer = parent_line.content.decode("utf-8")
    if (
        action.kind != "replace_line"
        or action.text != answer
        or sha256_bytes(answer.encode("utf-8")) != target_line_sha
    ):
        raise ValueError("public-prefix R label differs from the exact parent source line")
    if apply_action(state, action) != after_source:
        raise ValueError("public-prefix action does not reconstruct its recorded after state")
    if row.get("after_source_sha256") != sha256_bytes(after_source.encode("utf-8")):
        raise ValueError("public-prefix after-state hash mismatch")

    serialized = serialize_state(state)
    if (
        prompt != serialized
        or context_sha != sha256_bytes(serialized.encode("utf-8"))
        or row.get("context_policy") != CONTEXT_POLICY_VERSION
        or row.get("history_order") != PUBLIC_PREFIX_HISTORY_ORDER
        or row.get("human_chronology_observed") is not False
    ):
        raise ValueError("public-prefix prompt or chronology declaration is invalid")
    if row.get("source_type") != PUBLIC_PREFIX_SOURCE_TYPE:
        raise ValueError("public-prefix source type is invalid")
    return {
        "state_sha256": canonical_sha256(
            {
                "filetype": state.filetype,
                "source": state.source,
                "target_row": state.target_row,
                "cursor_col": state.cursor_col,
                "history": [
                    {"row": edit.row, "old_text": edit.old_text, "new_text": edit.new_text}
                    for edit in state.history
                ],
            }
        ),
        "action_sha256": canonical_sha256({"kind": action.kind, "text": action.text}),
        "history_sha256": canonical_sha256(
            [{"row": edit.row, "old_text": edit.old_text, "new_text": edit.new_text}]
        ),
        "target_line_sha256": target_line_sha,
        "action_family": "R",
    }


def validate_public_prefix_row(row: Mapping[str, Any], *, package_root: Any) -> dict[str, Any]:
    """Reopen one row's source and path-license proofs, then replay its label."""
    from pathlib import Path

    root = Path(package_root)
    metadata = row.get("authoring_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("public-prefix row lacks source provenance")
    required_string_fields = (
        "source_repo",
        "source_revision",
        "origin_transition_revision",
        "source_path",
        "source_sha256",
        "source_group_id",
        "task_family_id",
        "template_id",
    )
    for field in required_string_fields:
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError("public-prefix row identity is missing: " + field)
    for field in ("source_revision", "origin_transition_revision"):
        if len(str(row[field])) != 40 or any(ch not in "0123456789abcdef" for ch in row[field]):
            raise ValueError("public-prefix revision identity is invalid")

    license_name = row.get("source_license")
    if license_name not in PUBLIC_PREFIX_SUPPORTED_LICENSES:
        raise ValueError("public-prefix path license is not in the approved set")
    if metadata.get("source_revision") != row.get("source_revision"):
        raise ValueError("public-prefix source revision differs from row identity")
    if (
        metadata.get("source_license") != license_name
        or metadata.get("path_license") != license_name
    ):
        raise ValueError("public-prefix source license differs from its row metadata")
    if metadata.get("license_scope_status") != "verified_path_scope":
        raise ValueError("public-prefix file lacks verified parent-path scope")

    source_payload = _read_artifact(root, metadata, "source_artifact")
    if sha256_bytes(source_payload) != row.get("source_sha256"):
        raise ValueError("public-prefix parent source hash mismatch")
    scope_payload = _read_artifact(root, metadata, "license_scope_artifact")
    license_payload = _read_artifact(root, metadata, "path_license_artifact")
    root_license_payload = _read_artifact(root, metadata, "root_license_artifact")
    if license_payload != scope_payload:
        raise ValueError("public-prefix path-scope and path-license evidence differ")
    try:
        scope = json.loads(scope_payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("public-prefix scope proof is invalid JSON") from None
    if not isinstance(scope, Mapping) or scope.get("schema") != "exact-parent-license-scope-v2":
        raise ValueError("public-prefix source scope has an unknown schema")
    path_scope = scope.get("path_scope")
    root_license = scope.get("root_license")
    if not isinstance(path_scope, Mapping) or not isinstance(root_license, Mapping):
        raise ValueError("public-prefix source scope lacks license identities")
    if (
        scope.get("status") != "verified_path_scope"
        or scope.get("repository") != row.get("source_repo")
        or scope.get("parent_commit") != row.get("source_revision")
        or scope.get("source_path") != row.get("source_path")
        or scope.get("source_sha256") != row.get("source_sha256")
        or path_scope.get("scope") != "root"
        or path_scope.get("spdx") != [str(license_name).casefold()]
        or path_scope.get("spdx_ambiguous") is not False
        or path_scope.get("spdx_expression_count") != 0
        or scope.get("source_header_spdx_ambiguous") is not False
        or scope.get("source_header_spdx_expression_count") != 0
        or scope.get("additional_license_references") != []
        or scope.get("reuse_dep5_references") != []
        or path_scope.get("sha256") != metadata.get("path_license_sha256")
        or root_license.get("sha256") != metadata.get("root_license_sha256")
        or sha256_bytes(root_license_payload) != metadata.get("root_license_sha256")
    ):
        raise ValueError("public-prefix path-license proof does not bind to the source file")
    if scope.get("source_group_id") not in {None, row.get("source_group_id")} or scope.get(
        "parent_commit"
    ) != metadata.get("source_revision"):
        raise ValueError("public-prefix source group or parent commit differs from scope proof")

    state_bindings = validate_public_prefix_transition(row, parent_source=source_payload)
    template_id = row.get("template_id")
    state = EditState.from_mapping(row["state"])
    template_key = public_prefix_template_key(state)
    if (
        not isinstance(template_id, str)
        or not template_id.startswith("return-prefix-" + state.filetype + "/")
        or template_id.rsplit("/", 1)[-1] != template_key
        or row.get("near_duplicate_sha256") != template_key
    ):
        raise ValueError("public-prefix near-duplicate identity mismatch")
    return state_bindings


def validate_public_prefix_splits(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit public rows with the frozen materializer's near-key definition."""
    ids: set[str] = set()
    states: set[str] = set()
    normalized_inputs: set[str] = set()
    groups: dict[tuple[str, str], str] = {}
    counts: Counter[str] = Counter()
    fields = (
        "source_group_id",
        "source_repo",
        "session_or_commit",
        "task_family_id",
        "near_duplicate_sha256",
    )
    for row in rows:
        identifier, split = row.get("candidate_id"), row.get("split")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in ids
            or split not in {"train", "development"}
        ):
            raise ValueError("public-prefix split row identity is invalid")
        ids.add(identifier)
        counts[str(split)] += 1
        state = EditState.from_mapping(row["state"])
        state_key = canonical_sha256(asdict(state))
        normalized_state = {
            "filetype": state.filetype,
            "source": state.source,
            "target_row": state.target_row,
            "cursor_col": state.cursor_col,
            "history": [asdict(edit) for edit in state.history],
            "relevant": list(state.relevant),
        }
        normalized_key = canonical_sha256(normalized_state)
        if state_key in states or normalized_key in normalized_inputs:
            raise ValueError("public-prefix split rows contain a duplicate model input")
        states.add(state_key)
        normalized_inputs.add(normalized_key)
        if row.get("near_duplicate_sha256") != public_prefix_template_key(state):
            raise ValueError("public-prefix split near-duplicate key differs from source state")
        for field in fields:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("public-prefix split grouping identity is missing: " + field)
            normalized = value.casefold() if field == "source_repo" else value
            prior = groups.setdefault((field, normalized), str(split))
            if prior != split:
                raise ValueError("public-prefix source group crosses splits: " + field)
    return {
        "rows": len(rows),
        "splits": dict(sorted(counts.items())),
        "group_fields": list(fields),
        "state_duplicates": 0,
        "normalized_input_duplicates": 0,
        "near_key_policy": "verified-parent-prefix-materializer-v3",
    }


def _read_artifact(root: Any, metadata: Mapping[str, Any], prefix: str) -> bytes:
    from pathlib import Path

    relative = metadata.get(prefix + "_path")
    expected_sha = metadata.get(prefix + "_sha256")
    expected_size = metadata.get(prefix + "_bytes")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or type(expected_size) is not int
        or expected_size < 0
    ):
        raise ValueError("public-prefix artifact identity is incomplete: " + prefix)
    root_path = Path(root).resolve()
    artifact_path = (root_path / relative).resolve()
    if root_path not in artifact_path.parents or not artifact_path.is_file():
        raise ValueError("public-prefix artifact path escapes or is absent: " + prefix)
    payload = artifact_path.read_bytes()
    if len(payload) != expected_size or sha256_bytes(payload) != expected_sha:
        raise ValueError("public-prefix artifact hash/size mismatch: " + prefix)
    return payload
