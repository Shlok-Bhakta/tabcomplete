#!/usr/bin/env python3
"""Read-only audit for one scripted Rust-editor collector session.

The optional duplicate-delivery check POSTs the exact saved event envelopes to
the existing tailnet collector API. It is opt-in because it performs a network
request, although the collector should treat these event IDs idempotently.
No prompt, source, raw response, action text, or absolute file path is emitted.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import ipaddress
import json
import math
import os
import pathlib
import re
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

DEFAULT_DB = pathlib.Path("/mnt/ssd/collector-data/collector.sqlite")
DEFAULT_COLLECTOR = "http://100.100.163.102:8787"
DEFAULT_REPORT_DIR = pathlib.Path(__file__).resolve().parent
MAX_BLOB_BYTES = 8 * 1024 * 1024
MODEL_REGISTRY = {
    "q25": {
        "sha256": "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb",
        "model_protocol": "single-line-edit-v1",
        "wire_version": "single-line-edit-v1",
        "context_policy_versions": frozenset(
            {
                "single-line-context-v2",
                "single-line-cursor-last-context-v1",
            }
        ),
        "output_tokens": 64,
        "precision": "Q4_K_M",
    },
    "sweep": {
        "sha256": "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4",
        "model_protocol": "sweep-full-file-v1",
        "wire_version": "single-line-edit-v1",
        "context_policy_versions": frozenset({"sweep-window-context-v1"}),
        "output_tokens": 192,
        "precision": "Q4_K_M",
    },
}
BLOB_FIELDS = {"content_hash", "context_blob_hash", "raw_response_hash", "action_blob_hash"}
TERMINAL_LIFECYCLES = {
    "cancelled_unseen",
    "invalidated_unseen",
    "shadow_only",
    "model_no_edit",
    "request_failed",
    "invalid_output",
    "automatic_multiline_suppressed",
    "automatic_policy_suppressed",
    "expired",
}


class AuditError(Exception):
    """A stable, content-free verification failure."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def nonnegative_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def require_context_policy(model_spec: dict[str, Any], policy: Any) -> str:
    """Validate a declared policy while returning its exact recorded value."""
    if not isinstance(policy, str) or policy not in model_spec["context_policy_versions"]:
        raise AuditError("request_context_policy_mismatch")
    return policy


def compare_backend_timings(stored: Any, terminal: Any) -> dict[str, Any]:
    """Compare exact count fields and tolerate only JSON float round-trip noise."""
    count_fields = ("cache_n", "prompt_n", "predicted_n")
    duration_fields = ("prompt_ms", "predicted_ms", "total_ms")
    expected_fields = set(count_fields + duration_fields)
    valid_objects = isinstance(stored, dict) and isinstance(terminal, dict)
    matching_fields = (
        valid_objects and set(stored) == expected_fields and set(terminal) == expected_fields
    )
    exact_match = matching_fields and stored == terminal
    close_fields = []
    matches = bool(matching_fields)
    if matching_fields:
        for key in count_fields:
            if type(stored[key]) is not int or type(terminal[key]) is not int:
                matches = False
            elif stored[key] != terminal[key]:
                matches = False
        for key in duration_fields:
            if not nonnegative_number(stored[key]) or not nonnegative_number(terminal[key]):
                matches = False
                continue
            if stored[key] != terminal[key]:
                close = math.isclose(stored[key], terminal[key], rel_tol=1e-12, abs_tol=1e-9)
                if close:
                    close_fields.append(
                        {
                            "field": key,
                            "stored_event_value": stored[key],
                            "raw_terminal_value": terminal[key],
                        }
                    )
                else:
                    matches = False
    return {
        "matches": matches,
        "exact_match": exact_match,
        "float_round_trip_tolerance": {"relative": 1e-12, "absolute_ms": 1e-9},
        "float_fields_with_round_trip_difference": close_fields,
    }


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_json_object(raw: str | bytes | None) -> dict[str, Any]:
    if raw is None:
        return {}
    value = json.loads(raw, object_pairs_hook=strict_object)
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


def parse_event_payload(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    value = json.loads(raw, object_pairs_hook=strict_object)
    if isinstance(value, dict):
        return value
    # Older collector builds serialized empty Lua tables as [] for navigation
    # events. Treat only that exact empty shape as an empty payload.
    if value == []:
        return {}
    raise ValueError("event payload is not an object")


def connect_readonly(db_path: pathlib.Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    return connection


def limited_blob_bytes(row: sqlite3.Row) -> bytes:
    stored = bytes(row["content"])
    if row["original_bytes"] < 0 or row["original_bytes"] > MAX_BLOB_BYTES:
        raise AuditError("blob_size_outside_audit_limit")
    if len(stored) != row["stored_bytes"]:
        raise AuditError("blob_stored_length_mismatch")
    if row["compression"] == "raw":
        data = stored
    elif row["compression"] == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(stored), mode="rb") as stream:
                data = stream.read(row["original_bytes"] + 1)
        except (OSError, EOFError, ValueError):
            raise AuditError("blob_decompression_failed") from None
    else:
        raise AuditError("blob_compression_unknown")
    if len(data) != row["original_bytes"]:
        raise AuditError("blob_original_length_mismatch")
    if sha256(data) != row["sha256"]:
        raise AuditError("blob_digest_mismatch")
    return data


def load_database_snapshot(
    db_path: pathlib.Path, session_id: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        connection = connect_readonly(db_path)
    except sqlite3.Error:
        raise AuditError("sqlite_open_failed") from None
    try:
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise AuditError("sqlite_integrity_check_failed")
        foreign_key_issues = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
        if foreign_key_issues:
            raise AuditError("sqlite_foreign_key_check_failed")

        session = connection.execute(
            "SELECT session_id,repo_id,started_at,ended_at FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise AuditError("session_not_found")
        event_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM events WHERE session_id=? ORDER BY sequence_number,event_id",
                (session_id,),
            )
        ]
        if not event_rows:
            raise AuditError("session_has_no_events")
        try:
            for row in event_rows:
                row["payload"] = parse_event_payload(row["payload_json"])
        except (json.JSONDecodeError, UnicodeError, ValueError):
            raise AuditError("event_payload_malformed") from None

        sequence = [row["sequence_number"] for row in event_rows]
        if sequence != list(range(1, len(sequence) + 1)):
            raise AuditError("session_sequence_not_contiguous_from_one")
        if len({row["event_id"] for row in event_rows}) != len(event_rows):
            raise AuditError("duplicate_event_id_in_session")

        blob_rows = connection.execute(
            "SELECT sha256,original_bytes,stored_bytes,compression,content "
            "FROM blobs ORDER BY sha256"
        ).fetchall()
        blobs: dict[str, bytes] = {}
        blob_records = []
        for row in blob_rows:
            data = limited_blob_bytes(row)
            blobs[row["sha256"]] = data
            blob_records.append(
                {
                    "sha256": row["sha256"],
                    "original_bytes": row["original_bytes"],
                    "stored_bytes": row["stored_bytes"],
                    "compression": row["compression"],
                    "hash_matches": True,
                }
            )

        referenced: set[str] = set()
        for row in event_rows:
            for key in BLOB_FIELDS:
                value = row["payload"].get(key)
                if value is None:
                    continue
                if not isinstance(value, str) or len(value) != 64:
                    raise AuditError("blob_reference_malformed")
                if value not in blobs:
                    raise AuditError("session_blob_reference_missing")
                referenced.add(value)

        projection_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_projection'"
        ).fetchone()
        if projection_exists is None:
            raise AuditError("prediction_projection_missing")
        projections = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM prediction_projection WHERE session_id=? ORDER BY prediction_id",
                (session_id,),
            )
        ]
        repo = connection.execute(
            "SELECT root_name FROM repositories WHERE repo_id=?", (session["repo_id"],)
        ).fetchone()
        if repo is None:
            raise AuditError("session_repository_missing")
        file_ids_by_repo_path = {
            (row["repo_id"], row["relative_path"]): row["file_id"]
            for row in connection.execute(
                "SELECT repo_id,relative_path,file_id FROM files WHERE repo_id=?",
                (session["repo_id"],),
            )
        }

        session_event_count = len(event_rows)
        session_projection_count = len(projections)
        global_counts = {
            "events": connection.execute("SELECT count(*) FROM events").fetchone()[0],
            "projections": connection.execute(
                "SELECT count(*) FROM prediction_projection"
            ).fetchone()[0],
            "blobs": connection.execute("SELECT count(*) FROM blobs").fetchone()[0],
        }
        projection_snapshot = projection_digest(projections)
        event_snapshot = event_digest(event_rows)
        blob_snapshot = sha256(
            json.dumps(blob_records, sort_keys=True, separators=(",", ":")).encode()
        )
        session_info = {
            "repo_id": session["repo_id"],
            "root_name": repo["root_name"],
            "file_ids_by_repo_path": file_ids_by_repo_path,
            "event_rows": event_rows,
            "projections": projections,
            "event_count": session_event_count,
            "projection_count": session_projection_count,
            "global_counts": global_counts,
            "event_digest": event_snapshot,
            "projection_digest": projection_snapshot,
            "blob_records": blob_records,
            "blob_digest": blob_snapshot,
            "session_blob_reference_count": len(referenced),
        }
        return session_info, blobs
    except sqlite3.Error:
        raise AuditError("sqlite_read_failed") from None
    finally:
        connection.rollback()
        connection.close()


def event_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        values = [
            row.get(key)
            for key in (
                "event_id",
                "session_id",
                "sequence_number",
                "event_type",
                "timestamp_ms",
                "file_id",
                "changedtick",
                "cursor_row",
                "cursor_col",
                "mode",
                "payload_json",
            )
        ]
        digest.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def projection_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        values = [row[key] for key in sorted(row)]
        digest.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def split_joined(text: str, count: int) -> list[bytes]:
    if count == 0:
        if text != "":
            raise AuditError("delta_empty_range_has_text")
        return []
    parts = text.split("\n")
    if len(parts) != count:
        raise AuditError("delta_line_count_mismatch")
    try:
        return [part.encode("utf-8") for part in parts]
    except UnicodeError:
        raise AuditError("delta_text_invalid_utf8") from None


def decode_anchor(data: bytes, payload: dict[str, Any]) -> tuple[list[bytes], str, bool]:
    fileformat = payload.get("fileformat", "unix")
    eol = payload.get("eol", True)
    if fileformat not in {"unix", "dos"} or type(eol) is not bool:
        raise AuditError("anchor_file_format_invalid")
    normalized = data.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise AuditError("anchor_contains_lone_cr")
    if data == b"":
        lines = [b""]
    else:
        lines = normalized.split(b"\n")
        if eol and lines[-1] == b"":
            lines.pop()
    if payload.get("line_count") is not None and payload["line_count"] != len(lines):
        raise AuditError("anchor_line_count_mismatch")
    if not lines:
        lines = [b""]
    return lines, fileformat, eol


def encode_lines(lines: list[bytes], fileformat: str, eol: bool) -> bytes:
    if not lines or (len(lines) == 1 and lines[0] == b""):
        return b""
    separator = b"\r\n" if fileformat == "dos" else b"\n"
    return separator.join(lines) + (separator if eol else b"")


def apply_delta(lines: list[bytes], payload: dict[str, Any]) -> list[bytes]:
    start, old_end, new_end = (
        payload.get(key)
        for key in (
            "start_row",
            "old_end_row",
            "new_end_row",
        )
    )
    deleted, inserted = payload.get("deleted_text"), payload.get("inserted_text")
    if any(type(value) is not int for value in (start, old_end, new_end)):
        raise AuditError("delta_range_malformed")
    if start < 0 or old_end < start or new_end < start or old_end > len(lines):
        raise AuditError("delta_range_outside_replay")
    if not isinstance(deleted, str) or not isinstance(inserted, str):
        raise AuditError("delta_text_malformed")
    removed = split_joined(deleted, old_end - start)
    added = split_joined(inserted, new_end - start)
    if lines[start:old_end] != removed:
        raise AuditError("delta_deleted_lines_do_not_match_replay")
    return lines[:start] + added + lines[old_end:]


def replay_session(rows: list[dict[str, Any]], blobs: dict[str, bytes]) -> dict[str, Any]:
    event_by_id = {row["event_id"]: row for row in rows}
    prediction_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in rows:
        prediction_id = event["payload"].get("prediction_id")
        if isinstance(prediction_id, str):
            prediction_groups[prediction_id].append(event)
    traces = []
    for prediction_id, group in prediction_groups.items():
        request = next((row for row in group if row["event_type"] == "prediction_requested"), None)
        generated = next(
            (row for row in group if row["event_type"] == "prediction_generated"), None
        )
        dismissals = [row for row in group if row["event_type"] == "prediction_dismissed"]
        if not request or not generated or len(dismissals) != 1:
            continue
        req = request["payload"]
        dismissal = dismissals[0]
        edit = event_by_id.get(dismissal["payload"].get("ended_by_event_id"))
        if not edit or edit["event_type"] != "edit_delta":
            continue
        if edit["payload"].get("change_origin") == "buffer_reload":
            continue
        hash_value = generated["payload"].get("action_blob_hash")
        if not isinstance(hash_value, str) or hash_value not in blobs:
            continue
        action = decode_action_blob(blobs[hash_value])
        editable = req.get("editable_range")
        target_row = editable.get("start_row") if isinstance(editable, dict) else None
        path = req.get("file")
        if (
            action["kind"] != "replace_line"
            or type(target_row) is not int
            or not isinstance(path, str)
        ):
            continue
        next_request = next(
            (
                row
                for row in rows
                if row["sequence_number"] > dismissal["sequence_number"]
                and row["event_type"] == "prediction_requested"
                and row["payload"].get("file") == path
            ),
            None,
        )
        traces.append(
            {
                "prediction_id": prediction_id,
                "path": path,
                "target_row": target_row,
                "proposal": action["text"].encode("utf-8"),
                "dismissal_sequence": dismissal["sequence_number"],
                "causal_edit_event_id": edit["event_id"],
                "stop_sequence": next_request["sequence_number"] if next_request else None,
                "observations": [],
                "awaiting_key_delta": False,
                "pending_key_event_id": None,
                "continuity_break": None,
                "continuity_key_edit_pairs": [],
                "eventual_exact_match": None,
            }
        )

    delta_ids_needed = {
        row["payload"].get("ended_by_event_id")
        for row in rows
        if row["event_type"] == "prediction_dismissed"
        and isinstance(row["payload"].get("ended_by_event_id"), str)
    }
    seq_to_id = {row["sequence_number"]: row["event_id"] for row in rows}
    for row in rows:
        if row["event_type"] == "prediction_accepted":
            through = row["payload"].get("applied_through_sequence")
            if type(through) is int and through in seq_to_id:
                applied_id = seq_to_id[through]
                delta_ids_needed.add(applied_id)
                applied_row = next(
                    (event for event in rows if event["event_id"] == applied_id), None
                )
                if applied_row and applied_row["event_type"] == "edit_delta":
                    applied = applied_row["payload"]
                    for candidate in rows:
                        if (
                            candidate["sequence_number"] <= row["sequence_number"]
                            or candidate["event_type"] != "edit_delta"
                        ):
                            continue
                        delta = candidate["payload"]
                        if (
                            delta.get("path") == applied.get("path")
                            and delta.get("start_row") == applied.get("start_row")
                            and delta.get("deleted_text") == applied.get("inserted_text")
                            and delta.get("inserted_text") == applied.get("deleted_text")
                        ):
                            delta_ids_needed.add(candidate["event_id"])

    states: dict[str, dict[str, Any]] = {}
    event_states: dict[str, dict[str, Any]] = {}
    request_sources: dict[str, dict[str, Any]] = {}
    anchor_count = 0
    delta_count = 0
    unanchored_delta_count = 0
    replay_mismatch_count = 0
    replay_segments = []
    path_delta_counts: collections.Counter[str] = collections.Counter()

    for row in rows:
        kind = row["event_type"]
        payload = row["payload"]
        for trace in traces:
            if row["sequence_number"] <= trace["dismissal_sequence"]:
                continue
            if (
                trace["stop_sequence"] is not None
                and row["sequence_number"] >= trace["stop_sequence"]
            ):
                continue
            if trace["continuity_break"] or trace["eventual_exact_match"]:
                continue
            if kind == "key":
                if trace["awaiting_key_delta"]:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "key_without_intervening_edit",
                    }
                else:
                    trace["awaiting_key_delta"] = True
                    trace["pending_key_event_id"] = row["event_id"]
            elif kind == "edit_delta":
                if payload.get("change_origin") == "buffer_reload":
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "editor_buffer_reload",
                    }
                elif payload.get("change_origin") is not None:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "unrecognized_edit_origin",
                    }
                elif payload.get("path") != trace["path"]:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "edit_on_unrelated_path",
                    }
                elif not trace["awaiting_key_delta"]:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "edit_without_preceding_key",
                    }
                elif trace["path"] not in states:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "edit_without_replay_anchor",
                    }
            elif kind != "edit_delta":
                trace["continuity_break"] = {
                    "event_id": row["event_id"],
                    "sequence": row["sequence_number"],
                    "reason": "intervening_non_key_event",
                    "event_type": kind,
                }
        if kind in {"buffer_open", "buffer_write"}:
            content_hash = payload.get("content_hash")
            path = payload.get("path")
            if (
                not isinstance(content_hash, str)
                or content_hash not in blobs
                or not isinstance(path, str)
            ):
                continue
            if payload.get("blob_uploaded") is not True:
                raise AuditError("replay_anchor_not_uploaded")
            data = blobs[content_hash]
            if sha256(data) != content_hash:
                raise AuditError("replay_anchor_digest_mismatch")
            if path in states:
                previous = states[path]
                replayed = encode_lines(previous["lines"], previous["fileformat"], previous["eol"])
                match = replayed == data
                if not match:
                    replay_mismatch_count += 1
                replay_segments.append(
                    {
                        "path_sha256": sha256(path.encode()),
                        "from_sequence": previous["anchor_sequence"],
                        "to_sequence": row["sequence_number"],
                        "delta_count": previous["delta_count"],
                        "replayed_bytes": len(replayed),
                        "anchor_bytes": len(data),
                        "byte_equal": match,
                        "anchor_sha256": content_hash,
                    }
                )
            lines, fileformat, eol = decode_anchor(data, payload)
            states[path] = {
                "lines": lines,
                "fileformat": fileformat,
                "eol": eol,
                "anchor_sequence": row["sequence_number"],
                "anchor_event_id": row["event_id"],
                "delta_count": 0,
                "anchor_sha256": content_hash,
            }
            anchor_count += 1
        elif kind == "edit_delta":
            path = payload.get("path")
            if not isinstance(path, str) or path not in states:
                unanchored_delta_count += 1
                continue
            state = states[path]
            before = encode_lines(state["lines"], state["fileformat"], state["eol"])
            next_lines = apply_delta(state["lines"], payload)
            change_origin = payload.get("change_origin")
            if change_origin not in {None, "buffer_reload"}:
                raise AuditError("delta_change_origin_invalid")
            has_reload_metadata = "eol_after" in payload or "fileformat_after" in payload
            if has_reload_metadata and change_origin != "buffer_reload":
                raise AuditError("delta_reload_metadata_without_origin")
            next_fileformat = state["fileformat"]
            next_eol = state["eol"]
            if change_origin == "buffer_reload":
                if (
                    payload.get("start_row") != 0
                    or payload.get("old_end_row") != len(state["lines"])
                    or payload.get("new_end_row") != len(next_lines)
                ):
                    raise AuditError("reload_delta_not_full_buffer")
                next_fileformat = payload.get("fileformat_after", next_fileformat)
                next_eol = payload.get("eol_after", next_eol)
                if next_fileformat not in {"unix", "dos"} or type(next_eol) is not bool:
                    raise AuditError("reload_buffer_metadata_invalid")
            after = encode_lines(next_lines, next_fileformat, next_eol)
            if row["event_id"] in delta_ids_needed:
                event_states[row["event_id"]] = {
                    "path": path,
                    "before": before,
                    "after": after,
                    "sequence": row["sequence_number"],
                    "payload": payload,
                }
            state["lines"] = next_lines
            state["fileformat"] = next_fileformat
            state["eol"] = next_eol
            state["delta_count"] += 1
            delta_count += 1
            path_delta_counts[sha256(path.encode())] += 1
            for trace in traces:
                if trace["path"] != path or row["sequence_number"] <= trace["dismissal_sequence"]:
                    continue
                if (
                    trace["stop_sequence"] is not None
                    and row["sequence_number"] >= trace["stop_sequence"]
                ):
                    continue
                target = trace["target_row"]
                start, old_end, new_end = (
                    payload[key] for key in ("start_row", "old_end_row", "new_end_row")
                )
                changed_target = False
                if start <= target < old_end:
                    target = start
                    changed_target = True
                elif start == target and old_end == start and new_end > start:
                    changed_target = True
                elif target >= old_end:
                    target += new_end - old_end
                trace["target_row"] = target
                if not changed_target or target >= len(next_lines):
                    if not trace["continuity_break"] and not trace["eventual_exact_match"]:
                        trace["continuity_break"] = {
                            "event_id": row["event_id"],
                            "sequence": row["sequence_number"],
                            "reason": "edit_did_not_change_target_line",
                        }
                    continue
                current_line = next_lines[target]
                proposal = trace["proposal"]
                common = common_bytes(proposal, current_line)
                if current_line == proposal:
                    relation = "typed_match"
                elif current_line == proposal[: len(current_line)]:
                    relation = "matching_prefix_only"
                elif common > 0:
                    relation = "diverged_after_shared_prefix"
                else:
                    relation = "divergent"
                trace["observations"].append(
                    {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "derived_input_relation": relation,
                        "derived_common_prefix_bytes": common,
                    }
                )
                if trace["continuity_break"] or trace["eventual_exact_match"]:
                    continue
                if not trace["awaiting_key_delta"]:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "edit_without_preceding_key",
                    }
                    continue
                trace["awaiting_key_delta"] = False
                trace["continuity_key_edit_pairs"].append(
                    {
                        "key_event_id": trace["pending_key_event_id"],
                        "edit_event_id": row["event_id"],
                        "edit_sequence": row["sequence_number"],
                        "derived_input_relation": relation,
                    }
                )
                trace["pending_key_event_id"] = None
                if relation not in {"matching_prefix_only", "typed_match"}:
                    trace["continuity_break"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "reason": "input_diverged_from_proposal_prefix",
                    }
                elif relation == "typed_match":
                    trace["eventual_exact_match"] = {
                        "event_id": row["event_id"],
                        "sequence": row["sequence_number"],
                        "derived_input_relation": relation,
                        "derived_common_prefix_bytes": common,
                    }
        elif kind == "prediction_requested":
            prediction_id = payload.get("prediction_id")
            path = payload.get("file")
            if isinstance(prediction_id, str) and isinstance(path, str) and path in states:
                state = states[path]
                request_sources[prediction_id] = {
                    "source": encode_lines(state["lines"], state["fileformat"], state["eol"]),
                    "path": path,
                    "fileformat": state["fileformat"],
                    "eol": state["eol"],
                    "sequence": row["sequence_number"],
                    "anchor_sequence": state["anchor_sequence"],
                    "anchor_event_id": state["anchor_event_id"],
                    "anchor_sha256": state["anchor_sha256"],
                }

    return {
        "anchor_count": anchor_count,
        "delta_count": delta_count,
        "unanchored_delta_count": unanchored_delta_count,
        "replay_mismatch_count": replay_mismatch_count,
        "replay_segments": replay_segments,
        "path_delta_counts": dict(path_delta_counts),
        "event_states": event_states,
        "request_sources": request_sources,
        "seq_to_id": seq_to_id,
        "post_dismissal_trajectories": [
            {
                "prediction_id": trace["prediction_id"],
                "causal_edit_event_id": trace["causal_edit_event_id"],
                "dismissal_sequence": trace["dismissal_sequence"],
                "observation_count": len(trace["observations"]),
                "observations": trace["observations"],
                "continuity_key_edit_pairs": trace["continuity_key_edit_pairs"],
                "continuity_break": trace["continuity_break"],
                "eventual_exact_match": trace["eventual_exact_match"],
                "eventual_exact_match_continuity_verified": trace["eventual_exact_match"]
                is not None
                and trace["continuity_break"] is None,
                "later_divergence_observed": any(
                    observation["derived_input_relation"] == "diverged_after_shared_prefix"
                    for observation in trace["observations"]
                ),
                "interpretation": "derived from later scripted buffer deltas; not human feedback",
            }
            for trace in traces
        ],
    }


def decode_action_blob(data: bytes) -> dict[str, Any]:
    try:
        action = json.loads(data.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise AuditError("canonical_action_blob_invalid") from None
    if not isinstance(action, dict) or set(action) != {"kind", "text"}:
        raise AuditError("canonical_action_blob_schema_invalid")
    if action["kind"] in {"keep", "delete_line"}:
        if action["text"] is not None:
            raise AuditError("canonical_action_text_invalid")
    elif action["kind"] in {"replace_line", "insert_before"}:
        if not isinstance(action["text"], str) or any(ch in action["text"] for ch in "\r\n\0"):
            raise AuditError("canonical_action_text_invalid")
    else:
        raise AuditError("canonical_action_kind_invalid")
    return action


def decode_single_line(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise AuditError("raw_q25_output_invalid_utf8") from None
    if text == "N":
        return {"kind": "keep", "text": None}
    if text == "D":
        return {"kind": "delete_line", "text": None}
    if text.startswith("R\t") and not any(ch in text[2:] for ch in "\r\n\0"):
        return {"kind": "replace_line", "text": text[2:]}
    if text.startswith("I\t") and not any(ch in text[2:] for ch in "\r\n\0"):
        return {"kind": "insert_before", "text": text[2:]}
    raise AuditError("raw_q25_wire_invalid")


def parse_sse(
    raw: bytes, model_spec: dict[str, Any], model_sha: str
) -> tuple[bytes, dict[str, Any]]:
    """Recover wire content and validate the persisted real-model terminal frame."""
    frames = re.split(rb"\r?\n\r?\n", raw)
    content: list[bytes] = []
    terminal = None
    token_id_count = 0
    saw_token_list = False
    saw_nonempty_text = False
    frame_count = 0
    for frame in frames:
        if not frame:
            continue
        data_lines = []
        for line in frame.splitlines():
            if line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
        if not data_lines:
            continue
        try:
            event = parse_json_object(b"\n".join(data_lines))
        except (json.JSONDecodeError, UnicodeError, ValueError):
            raise AuditError("raw_sse_frame_invalid") from None
        frame_count += 1
        if type(event.get("stop")) is not bool:
            raise AuditError("raw_sse_stop_flag_invalid")
        if terminal is not None:
            raise AuditError("raw_sse_data_after_terminal")
        if event["stop"]:
            terminal = event
            continue
        text = event.get("content")
        if not isinstance(text, str):
            raise AuditError("raw_sse_content_invalid")
        try:
            content.append(text.encode("utf-8"))
        except UnicodeError:
            raise AuditError("raw_sse_content_invalid_utf8") from None
        if text:
            saw_nonempty_text = True
        tokens = event.get("tokens")
        if tokens is not None:
            saw_token_list = True
            if not isinstance(tokens, list) or any(
                type(token) is not int or token < 0 for token in tokens
            ):
                raise AuditError("raw_sse_token_ids_invalid")
            token_id_count += len(tokens)
    if terminal is None or frame_count == 0:
        raise AuditError("raw_sse_terminal_missing")
    if terminal.get("stop_type") != "eos":
        raise AuditError("generation_not_ended_by_eos")
    if terminal.get("model_protocol") != model_spec["model_protocol"]:
        raise AuditError("raw_sse_model_protocol_mismatch")
    if terminal.get("model_sha256") != model_sha:
        raise AuditError("raw_sse_model_digest_mismatch")
    predicted = terminal.get("tokens_predicted")
    if type(predicted) is not int or predicted < 1 or predicted > model_spec["output_tokens"]:
        raise AuditError("raw_sse_predicted_token_count_invalid")
    if saw_token_list and token_id_count != predicted:
        raise AuditError("raw_sse_token_ids_do_not_match_terminal_count")
    if not saw_token_list:
        raise AuditError("raw_sse_sampled_token_ids_missing")
    canonical = terminal.get("canonical_action")
    if not isinstance(canonical, dict) or set(canonical) != {"kind", "text"}:
        raise AuditError("raw_sse_canonical_action_invalid")
    try:
        decoded = decode_action_blob(
            json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError):
        raise AuditError("raw_sse_canonical_action_invalid") from None
    timings = terminal.get("timings")
    if not isinstance(timings, dict) or not all(
        nonnegative_number(timings.get(key))
        for key in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "total_ms")
    ):
        raise AuditError("raw_sse_timings_invalid")
    if timings["predicted_n"] != predicted:
        raise AuditError("raw_sse_timing_token_count_mismatch")
    return b"".join(content), {
        "terminal": terminal,
        "canonical_action": decoded,
        "token_id_count": token_id_count,
        "sampled_token_ids_present": saw_token_list,
        "nonempty_text_observed": saw_nonempty_text,
    }


def source_lines(data: bytes) -> list[tuple[bytes, bytes]]:
    """Rust context.rs physical lines as (content, terminator), no synthetic EOF row."""
    result = []
    start = 0
    for index, byte in enumerate(data):
        if byte == 10:
            if index > start and data[index - 1] == 13:
                content_end, terminator = index - 1, b"\r\n"
            else:
                content_end, terminator = index, b"\n"
            if b"\r" in data[start:content_end]:
                raise AuditError("source_contains_lone_cr")
            result.append((data[start:content_end], terminator))
            start = index + 1
    if start < len(data):
        if b"\r" in data[start:]:
            raise AuditError("source_contains_lone_cr")
        result.append((data[start:], b""))
    return result


def apply_action_to_source(source: bytes, row: int, action: dict[str, Any]) -> bytes:
    lines = source_lines(source)
    if row < 0 or row > len(lines):
        raise AuditError("action_target_outside_source")
    if action["kind"] == "keep":
        return source
    if action["kind"] == "replace_line":
        if row >= len(lines):
            raise AuditError("replace_target_outside_source")
        lines[row] = (action["text"].encode("utf-8"), lines[row][1])
    elif action["kind"] == "delete_line":
        if row >= len(lines):
            raise AuditError("delete_target_outside_source")
        del lines[row]
    elif action["kind"] == "insert_before":
        if row < len(lines):
            terminator = lines[row][1] or (lines[row - 1][1] if row > 0 else b"\n")
            lines.insert(row, (action["text"].encode("utf-8"), terminator))
        else:
            if lines and not lines[-1][1]:
                lines[-1] = (lines[-1][0], b"\n")
            lines.append((action["text"].encode("utf-8"), b""))
    return b"".join(content + terminator for content, terminator in lines)


def action_range(source: bytes, row: int, action: dict[str, Any]) -> dict[str, Any] | None:
    lines = source_lines(source)
    if action["kind"] == "keep":
        return None
    if row < 0 or row > len(lines):
        raise AuditError("action_target_outside_source")
    start_byte = sum(len(content) + len(term) for content, term in lines[:row])
    start_row, start_col = row, 0
    end_row, end_col = row, 0
    end_byte = start_byte
    includes_terminator = False
    if action["kind"] == "replace_line":
        if row >= len(lines):
            raise AuditError("replace_target_outside_source")
        end_col = len(lines[row][0])
        end_byte += len(lines[row][0])
    elif action["kind"] == "delete_line":
        if row >= len(lines):
            raise AuditError("delete_target_outside_source")
        content, terminator = lines[row]
        end_byte += len(content) + len(terminator)
        includes_terminator = bool(terminator)
        if terminator and row + 1 < len(lines):
            end_row, end_col = row + 1, 0
        else:
            end_col = len(content)
    elif action["kind"] != "insert_before":
        raise AuditError("canonical_action_kind_invalid")
    return {
        "start_row": start_row,
        "start_col": start_col,
        "end_row": end_row,
        "end_col": end_col,
        "start_byte": start_byte,
        "end_byte": end_byte,
        "end_exclusive": True,
        "includes_terminator": includes_terminator,
    }


def map_sweep(raw: bytes, source: bytes, target_row: int) -> dict[str, Any] | None:
    lines = source_lines(source)
    if target_row < 0 or target_row >= len(lines):
        return None
    for radius in range(12, -1, -1):
        start = max(0, target_row - radius)
        end = min(len(lines), target_row + radius + 1)
        window = b"".join(content + term for content, term in lines[start:end])
        if len(window) > 65536:
            continue
        if raw == window:
            return {"kind": "keep", "text": None}
        prefix = b"".join(content + term for content, term in lines[start:target_row])
        target_content = lines[target_row][0]
        suffix = window[len(prefix) + len(target_content) :]
        if not raw.startswith(prefix) or not raw.endswith(suffix):
            continue
        if len(raw) < len(prefix) + len(suffix):
            continue
        replacement = raw[len(prefix) : len(raw) - len(suffix) if suffix else len(raw)]
        if any(byte in replacement for byte in (b"\r", b"\n", b"\0")):
            continue
        try:
            return {"kind": "replace_line", "text": replacement.decode("utf-8")}
        except UnicodeError:
            continue
    return None


def action_for_event(
    group: list[dict[str, Any]], blobs: dict[str, bytes]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in group:
        by_type[row["event_type"]].append(row)
    generated = by_type.get("prediction_generated", [])
    shown = by_type.get("prediction_shown", [])
    if len(generated) > 1 or len(shown) > 1:
        raise AuditError("prediction_generation_or_display_duplicated")
    generated_row = generated[0] if generated else None
    shown_row = shown[0] if shown else None
    action = None
    if generated_row:
        hash_value = generated_row["payload"].get("action_blob_hash")
        if not isinstance(hash_value, str) or hash_value not in blobs:
            raise AuditError("generated_action_blob_missing")
        action = decode_action_blob(blobs[hash_value])
        if generated_row["payload"].get("canonical_action") != action["kind"]:
            raise AuditError("generated_action_kind_mismatch")
        if shown_row:
            shown_payload = shown_row["payload"]
            if shown_payload.get("action_blob_hash") != hash_value:
                raise AuditError("shown_action_blob_link_mismatch")
            if shown_payload.get("action") != action["kind"]:
                raise AuditError("shown_action_kind_mismatch")
            shown_text = shown_payload.get("proposed_text")
            expected_text = "" if action["kind"] == "delete_line" else action["text"]
            if shown_text != expected_text and not (
                action["kind"] == "delete_line" and shown_text is None
            ):
                raise AuditError("shown_text_does_not_match_action_blob")
    return generated_row, shown_row, action


def classify_group(
    prediction_id: str,
    group: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    blobs: dict[str, bytes],
    replay: dict[str, Any],
    root_name: str | None,
    repo_id: str | None,
) -> dict[str, Any]:
    group.sort(key=lambda row: row["sequence_number"])
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in group:
        by_type[row["event_type"]].append(row)
    requests = by_type.get("prediction_requested", [])
    if len(requests) != 1:
        raise AuditError("prediction_request_missing_or_duplicated")
    request = requests[0]
    req = request["payload"]
    request_id = req.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise AuditError("request_identity_missing")
    for row in group:
        if row["payload"].get("request_id") != request_id:
            raise AuditError("request_identity_link_mismatch")
        if row["payload"].get("prediction_id") != prediction_id:
            raise AuditError("prediction_identity_link_mismatch")
        if row["payload"].get("synthetic") is not True:
            raise AuditError("prediction_event_not_marked_synthetic")
        if row["payload"].get("human_verified") is True:
            raise AuditError("synthetic_event_claims_human_verification")
    if req.get("synthetic") is not True:
        raise AuditError("prediction_not_marked_synthetic")
    if req.get("human_verified") is True:
        raise AuditError("synthetic_request_claims_human_verification")
    generated, shown, action = action_for_event(group, blobs)

    if shown and (generated is None or generated["sequence_number"] >= shown["sequence_number"]):
        raise AuditError("display_precedes_generation")
    if shown and shown["sequence_number"] <= request["sequence_number"]:
        raise AuditError("display_precedes_request")
    if generated and generated["sequence_number"] <= request["sequence_number"]:
        raise AuditError("generation_precedes_request")

    terminal_rows = [
        row
        for row in group
        if row["event_type"]
        in {
            "prediction_accepted",
            "prediction_partially_accepted",
            "prediction_dismissed",
        }
    ]
    terminal_lifecycles = [
        row
        for row in by_type.get("heartbeat", [])
        if row["payload"].get("prediction_lifecycle") in TERMINAL_LIFECYCLES
    ]
    if (
        len(terminal_rows) > 1
        or len(terminal_lifecycles) > 1
        or (terminal_rows and terminal_lifecycles)
    ):
        raise AuditError("multiple_terminal_decisions")
    primary = (
        terminal_rows[0]
        if terminal_rows
        else (terminal_lifecycles[0] if terminal_lifecycles else None)
    )
    if primary is None:
        raise AuditError("prediction_missing_terminal_decision")
    if primary and primary["sequence_number"] <= request["sequence_number"]:
        raise AuditError("terminal_decision_precedes_request")
    if primary and shown and primary["sequence_number"] <= shown["sequence_number"]:
        raise AuditError("terminal_decision_precedes_display")

    accepted_rows = by_type.get("prediction_accepted", [])
    rejected_rows = by_type.get("prediction_rejected", [])
    dismissed_rows = by_type.get("prediction_dismissed", [])
    if accepted_rows:
        if (
            len(accepted_rows) != 1
            or dismissed_rows
            or rejected_rows
            or primary is not accepted_rows[0]
        ):
            raise AuditError("acceptance_terminal_cardinality_invalid")
        if shown is None or accepted_rows[0]["payload"].get("shown_event_id") != shown["event_id"]:
            raise AuditError("acceptance_show_link_invalid")
    elif rejected_rows:
        # The collector's explicit reject emits a dismissal outcome and a
        # companion prediction_rejected event for the same single decision.
        if len(rejected_rows) != 1 or len(dismissed_rows) != 1:
            raise AuditError("explicit_rejection_pair_invalid")
        if dismissed_rows[0]["payload"].get("outcome") != "rejected_explicit":
            raise AuditError("explicit_rejection_outcome_invalid")
        if rejected_rows[0]["sequence_number"] <= dismissed_rows[0]["sequence_number"]:
            raise AuditError("explicit_rejection_event_order_invalid")
    elif primary and primary["event_type"] == "prediction_dismissed":
        if len(dismissed_rows) != 1:
            raise AuditError("dismissal_terminal_cardinality_invalid")
    elif primary and primary["event_type"] == "prediction_partially_accepted":
        if (
            len(by_type.get("prediction_partially_accepted", [])) != 1
            or dismissed_rows
            or rejected_rows
        ):
            raise AuditError("partial_acceptance_terminal_cardinality_invalid")

    outcome = None
    if accepted_rows:
        outcome = "accepted"
    elif by_type.get("prediction_partially_accepted"):
        outcome = "partially_accepted"
    elif rejected_rows:
        outcome = "rejected_explicit"
    elif dismissed_rows:
        outcome = dismissed_rows[0]["payload"].get("outcome", "dismissed_unknown")
    elif terminal_lifecycles:
        outcome = terminal_lifecycles[0]["payload"].get("prediction_lifecycle")

    model_alias = req.get("model_alias") or req.get("model")
    model_protocol = req.get("model_protocol")
    model_sha = req.get("model_gguf_sha256") or req.get("model_revision")
    runtime_hash = req.get("runtime_config_hash")
    context_policy = req.get("context_policy_version")
    context_hash = req.get("context_hash")
    context_blob_hash = req.get("context_blob_hash")
    wire_version = req.get("wire_version")
    precision = req.get("precision")
    if not isinstance(context_hash, str) or not isinstance(context_blob_hash, str):
        raise AuditError("request_context_hash_missing")
    if context_hash != context_blob_hash or context_blob_hash not in blobs:
        raise AuditError("request_context_blob_link_invalid")
    if sha256(blobs[context_blob_hash]) != context_hash:
        raise AuditError("request_context_prompt_hash_invalid")
    if wire_version != "single-line-edit-v1":
        raise AuditError("canonical_wire_version_invalid")

    model_spec = MODEL_REGISTRY.get(model_alias) if isinstance(model_alias, str) else None
    if model_spec is None:
        raise AuditError("model_alias_not_allowlisted")
    if model_protocol != model_spec["model_protocol"] or model_sha != model_spec["sha256"]:
        raise AuditError("request_model_identity_mismatch")
    if (
        precision != model_spec["precision"]
        or req.get("max_output_tokens") != model_spec["output_tokens"]
    ):
        raise AuditError("request_model_precision_or_cap_mismatch")
    context_policy = require_context_policy(model_spec, context_policy)
    if not isinstance(runtime_hash, str) or len(runtime_hash) != 64:
        raise AuditError("request_runtime_identity_invalid")
    if any(character not in "0123456789abcdef" for character in runtime_hash.lower()):
        raise AuditError("request_runtime_identity_invalid")
    source_info = replay["request_sources"].get(prediction_id)
    if source_info is None:
        raise AuditError("request_pre_state_unavailable")
    if sha256(source_info["source"]) != req.get("pre_state_hash"):
        raise AuditError("request_pre_state_hash_mismatch")

    backend_timing_comparison = None
    if generated:
        gp = generated["payload"]
        expected_generated = {
            "model_alias": model_alias,
            "model_protocol": model_protocol,
            "model_sha256": model_sha,
            "runtime_config_hash": runtime_hash,
            "context_hash": context_hash,
            "context_policy_version": context_policy,
            "wire_version": wire_version,
            "max_output_tokens": model_spec["output_tokens"],
            "canonical_action": action["kind"] if action else None,
        }
        if any(gp.get(key) != value for key, value in expected_generated.items()):
            raise AuditError("generated_model_or_context_identity_mismatch")
        if gp.get("stop_type") != "eos":
            raise AuditError("generation_not_ended_by_eos")
        raw_hash = gp.get("raw_response_hash")
        if not isinstance(raw_hash, str) or raw_hash not in blobs:
            raise AuditError("raw_model_response_blob_missing")
        raw, raw_details = parse_sse(blobs[raw_hash], model_spec, model_sha)
        if raw_details["canonical_action"] != action:
            raise AuditError("terminal_action_blob_mismatch")
        terminal = raw_details["terminal"]
        if model_protocol == "single-line-edit-v1":
            if decode_single_line(raw) != action:
                raise AuditError("q25_raw_wire_action_mismatch")
        else:
            source_info = replay["request_sources"].get(prediction_id)
            editable = req.get("editable_range")
            target_row = editable.get("start_row") if isinstance(editable, dict) else None
            if source_info is None or type(target_row) is not int:
                raise AuditError("sweep_pre_state_unavailable")
            mapped = map_sweep(raw, source_info["source"], target_row)
            if mapped != action:
                raise AuditError("sweep_raw_output_action_mismatch")
        if gp.get("first_token_observation") != "sampled_token_ids":
            raise AuditError("first_token_provenance_invalid")
        if gp.get("first_token_at_ms") is None:
            raise AuditError("first_token_timing_missing")
        if raw_details["nonempty_text_observed"] and gp.get("first_text_at_ms") is None:
            raise AuditError("first_text_timing_missing")
        prompt_tokens = gp.get("prompt_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise AuditError("prompt_token_count_missing")
        backend_timing_comparison = compare_backend_timings(
            gp.get("backend_timings"), terminal.get("timings")
        )
        if not backend_timing_comparison["matches"]:
            raise AuditError("stored_terminal_timings_mismatch")
        if gp.get("model_sha256") != terminal.get("model_sha256"):
            raise AuditError("stored_terminal_model_digest_mismatch")
        if gp.get("model_protocol") != terminal.get("model_protocol"):
            raise AuditError("stored_terminal_model_protocol_mismatch")
        if not action or gp.get("action_blob_hash") != (
            shown["payload"].get("action_blob_hash") if shown else gp.get("action_blob_hash")
        ):
            raise AuditError("generated_action_blob_link_invalid")
        timings = gp.get("backend_timings")
        if not isinstance(timings, dict) or not all(
            nonnegative_number(timings.get(key))
            for key in (
                "cache_n",
                "prompt_n",
                "prompt_ms",
                "predicted_n",
                "predicted_ms",
                "total_ms",
            )
        ):
            raise AuditError("generation_timings_missing_or_invalid")
        if timings["predicted_n"] > model_spec["output_tokens"]:
            raise AuditError("generation_exceeded_output_cap")

    if shown:
        sp = shown["payload"]
        if sp.get("context_hash") != context_hash or sp.get("wire_version") != wire_version:
            raise AuditError("display_context_or_wire_link_invalid")
        if sp.get("active_buffer") is not True or sp.get("focused") is not True:
            raise AuditError("display_not_active_or_focused")

    if action and generated and shown:
        editable = req.get("editable_range")
        target_row = editable.get("start_row") if isinstance(editable, dict) else None
        if source_info is None or type(target_row) is not int:
            raise AuditError("proposal_pre_state_unavailable")
        expected_bytes = apply_action_to_source(source_info["source"], target_row, action)
        range_value = action_range(source_info["source"], target_row, action)
        if range_value is None:
            raise AuditError("displayed_keep_action_invalid")
        shown_payload = shown["payload"]
        show_range = {
            "start_row": shown_payload.get("proposed_start", {}).get("row")
            if isinstance(shown_payload.get("proposed_start"), dict)
            else None,
            "start_col": shown_payload.get("proposed_start", {}).get("col")
            if isinstance(shown_payload.get("proposed_start"), dict)
            else None,
            "end_row": shown_payload.get("proposed_end", {}).get("row")
            if isinstance(shown_payload.get("proposed_end"), dict)
            else None,
            "end_col": shown_payload.get("proposed_end", {}).get("col")
            if isinstance(shown_payload.get("proposed_end"), dict)
            else None,
            "start_byte": shown_payload.get("proposed_start_byte"),
            "end_byte": shown_payload.get("proposed_end_byte"),
            "end_exclusive": shown_payload.get("proposed_range_end_exclusive"),
            "includes_terminator": shown_payload.get("proposed_range_includes_terminator"),
        }
        if show_range != range_value:
            raise AuditError("shown_edit_range_does_not_match_action")
        if accepted_rows:
            accepted_range = accepted_rows[0]["payload"].get("editable_range")
            if accepted_range != range_value:
                raise AuditError("accepted_range_does_not_match_shown_action")
            if (
                accepted_rows[0]["payload"].get("proposed_start_byte") != range_value["start_byte"]
                or accepted_rows[0]["payload"].get("proposed_end_byte") != range_value["end_byte"]
            ):
                raise AuditError("accepted_byte_range_link_invalid")
    else:
        expected_bytes = None

    accepted_delta_linked = False
    undo_verified = False
    undo_event_id = None
    accepted_delta = None
    accepted_delta_event_id = None
    accepted_delta_sequence = None
    if accepted_rows:
        accepted = accepted_rows[0]
        through = accepted["payload"].get("applied_through_sequence")
        linked_id = replay["seq_to_id"].get(through) if type(through) is int else None
        accepted_delta = replay["event_states"].get(linked_id) if linked_id else None
        if (
            accepted_delta is None
            or accepted_delta["sequence"] != through
            or accepted_delta["path"]
            != replay["request_sources"].get(prediction_id, {}).get("path")
        ):
            raise AuditError("accepted_edit_delta_link_invalid")
        if accepted_delta["after"] != expected_bytes:
            raise AuditError("accepted_buffer_bytes_do_not_match_action")
        accepted_delta_linked = True
        accepted_delta_event_id = linked_id
        accepted_delta_sequence = through
        applied_payload = accepted_delta["payload"]
        for candidate in event_rows[accepted["sequence_number"] :]:
            if candidate["event_type"] != "edit_delta":
                continue
            delta = candidate["payload"]
            if (
                delta.get("path") == accepted_delta["path"]
                and delta.get("start_row") == applied_payload.get("start_row")
                and delta.get("deleted_text") == applied_payload.get("inserted_text")
                and delta.get("inserted_text") == applied_payload.get("deleted_text")
            ):
                candidate_state = replay["event_states"].get(candidate["event_id"])
                if candidate_state is None:
                    continue
                source = replay["request_sources"].get(prediction_id, {}).get("source")
                if candidate_state["after"] == source:
                    undo_event_id = candidate["event_id"]
                    undo_verified = True
                    break

    dismissal_comparison = None
    if dismissed_rows:
        dismissed = dismissed_rows[0]
        dp = dismissed["payload"]
        edit_id = dp.get("ended_by_event_id")
        event_by_id = {row["event_id"]: row for row in event_rows}
        edit_row = event_by_id.get(edit_id) if isinstance(edit_id, str) else None
        edit_state = replay["event_states"].get(edit_id) if isinstance(edit_id, str) else None
        if edit_row is not None or edit_id is not None:
            if (
                edit_row is None
                or edit_row["event_type"] != "edit_delta"
                or edit_state is None
                or edit_row["sequence_number"] != dp.get("ended_by_sequence")
                or edit_row["sequence_number"] >= dismissed["sequence_number"]
                or edit_state["path"]
                != replay["request_sources"].get(prediction_id, {}).get("path")
            ):
                raise AuditError("dismissal_edit_delta_link_invalid")
            derived = "unrelated"
            common_prefix = 0
            proposal_preexisting_prefix = 0
            newly_typed_prefix = 0
            collector_classification = None
            collector_matched_bytes = None
            request_source = replay["request_sources"][prediction_id]["source"]
            target_row = req.get("editable_range", {}).get("start_row")
            causal_edit_change_origin = edit_state["payload"].get("change_origin")
            if causal_edit_change_origin == "buffer_reload":
                derived = "buffer_reload"
            elif action and type(target_row) is int:
                collector_classification, collector_matched_bytes = classify_causal_delta(
                    action, request_source, target_row, edit_state["payload"]
                )
            if (
                causal_edit_change_origin != "buffer_reload"
                and action
                and action["kind"] == "replace_line"
            ):
                src_lines = source_lines(request_source)
                if type(target_row) is int and target_row < len(src_lines):
                    before_line = src_lines[target_row][0]
                    after_lines = source_lines(edit_state["after"])
                    proposal = action["text"].encode("utf-8")
                    if target_row < len(after_lines):
                        input_line = after_lines[target_row][0]
                        proposal_preexisting_prefix = common_bytes(before_line, proposal)
                        common_prefix = common_bytes(proposal, input_line)
                        if input_line.startswith(proposal[:proposal_preexisting_prefix]):
                            newly_typed_prefix = common_bytes(
                                proposal[proposal_preexisting_prefix:],
                                input_line[proposal_preexisting_prefix:],
                            )
                        else:
                            newly_typed_prefix = 0
                        if input_line == proposal:
                            derived = "typed_match"
                        elif input_line == proposal[: len(input_line)]:
                            derived = "matching_prefix_only"
                        elif common_prefix > 0:
                            derived = "diverged_after_shared_prefix"
                        else:
                            derived = "divergent"
            if dp.get("outcome") in {
                "typed_match",
                "typed_partial_match",
                "rejected_implicit_typing",
            }:
                expected_classification = collector_classification or "unrelated"
                if dp.get("outcome") != expected_classification:
                    raise AuditError("dismissal_outcome_disagrees_with_causal_delta")
                if dp.get("matching_prefix_bytes", 0) != collector_matched_bytes:
                    raise AuditError("dismissal_matching_byte_count_invalid")
            recorded_prefix = dp.get("matching_prefix_bytes")
            later_trace = next(
                (
                    trace
                    for trace in replay["post_dismissal_trajectories"]
                    if trace["prediction_id"] == prediction_id
                ),
                None,
            )
            dismissal_comparison = {
                "event_id": dismissed["event_id"],
                "causal_edit_event_id": edit_id,
                "causal_edit_sequence": edit_row["sequence_number"],
                "causal_edit_change_origin": causal_edit_change_origin,
                "outcome": dp.get("outcome"),
                "collector_outcome_derived_from_causal_delta": collector_classification,
                "collector_matching_bytes_derived": collector_matched_bytes,
                "derived_input_relation": derived,
                "derived_common_prefix_bytes": common_prefix,
                "proposal_bytes_already_in_pre_state": proposal_preexisting_prefix,
                "newly_typed_bytes_matching_proposal": newly_typed_prefix,
                "recorded_matching_prefix_bytes": recorded_prefix,
                "recorded_diverged_after_prefix": dp.get("diverged_after_prefix"),
                "post_dismissal_trajectory": later_trace,
                "divergence_flag_scope": "the first dismissal row cannot summarize later deltas",
                "interpretation": "scripted editor observation; not human feedback",
            }
    if outcome == "typed_match" and (
        not dismissal_comparison or dismissal_comparison["derived_input_relation"] != "typed_match"
    ):
        raise AuditError("typed_match_not_supported_by_causal_edit")
    if outcome == "rejected_implicit_typing" and dismissal_comparison is None:
        raise AuditError("implicit_rejection_lacks_causal_edit")

    projection = None
    projection_ref = None
    for row in group:
        projection_ref = row["payload"].get("prediction_id")
        break
    projection = next(
        (p for p in _CURRENT_PROJECTIONS if p["prediction_id"] == projection_ref), None
    )
    if projection is None:
        raise AuditError("prediction_projection_row_missing")
    expected_last = primary["event_id"] if primary else None
    if projection["request_event_id"] != request["event_id"]:
        raise AuditError("projection_request_link_invalid")
    if projection["shown_event_id"] != (shown["event_id"] if shown else None):
        raise AuditError("projection_show_link_invalid")
    if projection["last_outcome_event_id"] != expected_last:
        raise AuditError("projection_terminal_link_invalid")
    if projection["outcome"] != outcome:
        raise AuditError("projection_outcome_invalid")
    if projection["model_gguf_sha256"] != model_sha or projection["model_revision"] != model_sha:
        raise AuditError("projection_model_identity_invalid")
    if projection["runtime_config_hash"] != runtime_hash:
        raise AuditError("projection_runtime_identity_invalid")
    if projection["context_policy_version"] != context_policy:
        raise AuditError("projection_context_policy_invalid")
    if projection["context_blob_hash"] != context_blob_hash:
        raise AuditError("projection_context_blob_invalid")
    action_blob_hash = generated["payload"].get("action_blob_hash") if generated else None
    if projection["action_blob_hash"] != action_blob_hash:
        raise AuditError("projection_action_blob_invalid")
    if projection["review_status"] != "unreviewed" or projection["review_event_id"] is not None:
        raise AuditError("synthetic_projection_contains_review_claim")

    file_identity = req.get("file_identity") or req.get("file")
    file_identity_sha = sha256(file_identity.encode()) if isinstance(file_identity, str) else None
    if projection["file_identity"] != file_identity:
        raise AuditError("projection_file_identity_invalid")
    resolved_file_id = projection["resolved_file_id"]
    expected_file_id = None
    prefix = f"repo:{root_name}:" if root_name else None
    if isinstance(file_identity, str) and prefix and file_identity.startswith(prefix):
        relative_path = file_identity[len(prefix) :]
        expected_file_id = _file_ids_by_repo_path.get((repo_id, relative_path))
    if expected_file_id is not None and resolved_file_id != expected_file_id:
        raise AuditError("projection_resolved_file_identity_invalid")

    if projection["updated_through_sequence"] != max(row["sequence_number"] for row in group):
        raise AuditError("projection_sequence_watermark_invalid")
    return {
        "prediction_id": prediction_id,
        "request_id": request_id,
        "request_event_id": request["event_id"],
        "generated_event_id": generated["event_id"] if generated else None,
        "shown_event_id": shown["event_id"] if shown else None,
        "terminal_event_id": primary["event_id"] if primary else None,
        "terminal_event_type": primary["event_type"] if primary else None,
        "outcome": outcome,
        "request_sequence": request["sequence_number"],
        "terminal_sequence": primary["sequence_number"] if primary else None,
        "pre_state_hash": req.get("pre_state_hash"),
        "pre_state_hash_verified": True,
        "pre_state_anchor_sequence": source_info.get("anchor_sequence"),
        "pre_state_anchor_event_id": source_info.get("anchor_event_id"),
        "pre_state_anchor_sha256": source_info.get("anchor_sha256"),
        "accepted_edit_delta_event_id": accepted_delta_event_id,
        "accepted_edit_delta_sequence": accepted_delta_sequence,
        "synthetic": True,
        "human_verified": False,
        "review_status": projection["review_status"],
        "file_identity_sha256": file_identity_sha,
        "resolved_file_id": resolved_file_id,
        "model_alias": model_alias,
        "model_protocol": model_protocol,
        "model_sha256": model_sha,
        "precision": precision,
        "canonical_wire_version": wire_version,
        "context_policy_version": context_policy,
        "context_hash": context_hash,
        "context_blob_hash": context_blob_hash,
        "raw_response_hash": generated["payload"].get("raw_response_hash") if generated else None,
        "action_blob_hash": action_blob_hash,
        "backend_timing_comparison": backend_timing_comparison,
        "canonical_action_kind": action["kind"] if action else None,
        "action_blob_verified": action is not None,
        "raw_wire_verified": generated is not None,
        "accepted_edit_delta_linked": accepted_delta_linked,
        "undo_event_id": undo_event_id,
        "undo_restores_pre_state": undo_verified,
        "dismissal_comparison": dismissal_comparison,
        "projection_matches_events": True,
    }


def common_bytes(left: bytes, right: bytes) -> int:
    length = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        length += 1
    while length > 0 and (left[length : length + 1] and left[length] & 0xC0 == 0x80):
        length -= 1
    return length


def classify_causal_delta(
    action: dict[str, Any], source: bytes, target_row: int, delta: dict[str, Any]
) -> tuple[str, int]:
    """Mirror the plugin's synthetic single-line edit classifier, without human attribution."""
    lines = source_lines(source)
    kind = action["kind"]
    offered = (action.get("text") or "").encode("utf-8")
    matched = False
    if target_row < len(lines):
        original = lines[target_row][0].decode("utf-8")
        if kind == "replace_line":
            matched = (
                delta.get("old_end_row") == target_row + 1
                and delta.get("new_end_row") == target_row + 1
                and delta.get("deleted_text") == original
                and delta.get("inserted_text") == action["text"]
                and delta.get("start_row") == target_row
            )
        elif kind == "insert_before":
            matched = (
                delta.get("old_end_row") == target_row
                and delta.get("new_end_row") == target_row + 1
                and delta.get("deleted_text") == ""
                and delta.get("inserted_text") == action["text"]
                and delta.get("start_row") == target_row
            )
        elif kind == "delete_line":
            matched = (
                delta.get("old_end_row") == target_row + 1
                and delta.get("new_end_row") == target_row
                and delta.get("deleted_text") == original
                and delta.get("inserted_text") == ""
                and delta.get("start_row") == target_row
            )
    if matched:
        return "typed_match", len(offered)
    inserted = delta.get("inserted_text")
    if (
        kind != "delete_line"
        and offered
        and isinstance(inserted, str)
        and inserted
        and delta.get("start_row") == target_row
        and offered.startswith(inserted.encode("utf-8"))
    ):
        return "typed_partial_match", len(inserted.encode("utf-8"))
    return "rejected_implicit_typing", 0


_CURRENT_PROJECTIONS: list[dict[str, Any]] = []
_file_ids_by_repo_path: dict[tuple[str | None, str], int] = {}


def validate_models(
    rows: list[dict[str, Any]],
    projections: list[dict[str, Any]],
    replay: dict[str, Any],
    root_name: str | None,
    repo_id: str | None,
    blobs: dict[str, bytes],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        payload = row["payload"]
        if row["event_type"].startswith("prediction_") or row["event_type"] == "heartbeat":
            prediction_id = payload.get("prediction_id")
            if prediction_id is not None:
                if not isinstance(prediction_id, str) or not prediction_id:
                    raise AuditError("prediction_id_malformed")
                groups[prediction_id].append(row)
    if not groups:
        raise AuditError("session_has_no_prediction_events")
    if len(projections) != len(groups):
        raise AuditError("prediction_projection_cardinality_mismatch")
    _CURRENT_PROJECTIONS.clear()
    _CURRENT_PROJECTIONS.extend(projections)
    details = [
        classify_group(prediction_id, group, rows, blobs, replay, root_name, repo_id)
        for prediction_id, group in sorted(
            groups.items(), key=lambda item: min(x["sequence_number"] for x in item[1])
        )
    ]
    return details


def verify_navigation_identity(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    jumps = [row for row in rows if row["event_type"] == "file_jump"]
    identities = set()
    valid_jumps = 0
    issues = []
    for row in jumps:
        payload = row["payload"]
        source, target = payload.get("from_file"), payload.get("to_file")
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target
            or source == target
        ):
            issues.append("file_jump_identity_invalid")
            continue
        identities.update((source, target))
        later = [event for event in rows if event["sequence_number"] > row["sequence_number"]]
        following_enter = next(
            (event for event in later if event["event_type"] in {"buffer_enter", "file_jump"}), None
        )
        if following_enter and following_enter["event_type"] == "buffer_enter":
            valid_jumps += 1
        else:
            issues.append("file_jump_missing_following_buffer_enter")
    for prediction in predictions:
        digest = prediction["file_identity_sha256"]
        # The raw identity is intentionally absent from the report. Projection
        # checks already bind it to the request and resolved file row.
        if not digest:
            issues.append("prediction_file_identity_missing")
    return {
        "file_jump_count": len(jumps),
        "verified_file_jump_count": valid_jumps,
        "distinct_navigation_identity_count": len(identities),
        "identity_hashes": sorted(sha256(value.encode()) for value in identities),
        "issues": issues,
    }


def coverage_summary(
    predictions: list[dict[str, Any]], navigation: dict[str, Any], replay: dict[str, Any]
) -> dict[str, Any]:
    outcomes = collections.Counter(item["outcome"] for item in predictions)
    acceptance = [item for item in predictions if item["outcome"] == "accepted"]
    direct_typed = [
        item
        for item in predictions
        if item["outcome"] == "typed_match"
        and item["dismissal_comparison"]
        and item["dismissal_comparison"]["derived_input_relation"] == "typed_match"
    ]
    eventual_typed = []
    typed_evidence = []
    for item in direct_typed:
        comparison = item["dismissal_comparison"]
        typed_evidence.append(
            {
                "prediction_id": item["prediction_id"],
                "raw_dismissal_outcome": comparison["outcome"],
                "evidence_kind": "causal_delta_exact_match",
                "dismissal_event_id": comparison["event_id"],
                "causal_edit_event_id": comparison["causal_edit_event_id"],
                "causal_edit_sequence": comparison["causal_edit_sequence"],
                "exact_match_event_id": comparison["causal_edit_event_id"],
                "exact_match_sequence": comparison["causal_edit_sequence"],
                "interpretation": "derived scripted editor observation; not human feedback",
            }
        )
    for item in predictions:
        comparison = item["dismissal_comparison"]
        if item["outcome"] != "typed_partial_match" or comparison is None:
            continue
        trace = comparison.get("post_dismissal_trajectory")
        exact_match = trace.get("eventual_exact_match") if trace else None
        if (
            comparison.get("derived_input_relation") != "matching_prefix_only"
            or not trace
            or not trace.get("eventual_exact_match_continuity_verified")
            or not exact_match
        ):
            continue
        eventual_typed.append(item)
        typed_evidence.append(
            {
                "prediction_id": item["prediction_id"],
                "raw_dismissal_outcome": comparison["outcome"],
                "evidence_kind": "contiguous_typed_match_after_partial",
                "dismissal_event_id": comparison["event_id"],
                "causal_edit_event_id": comparison["causal_edit_event_id"],
                "causal_edit_sequence": comparison["causal_edit_sequence"],
                "exact_match_event_id": exact_match["event_id"],
                "exact_match_sequence": exact_match["sequence"],
                "post_dismissal_key_edit_pair_count": len(trace["continuity_key_edit_pairs"]),
                "causal_plus_followup_edit_delta_count": 1
                + len(trace["continuity_key_edit_pairs"]),
                "continuity_break": trace.get("continuity_break"),
                "interpretation": "derived scripted editor observation; not human feedback",
            }
        )
    divergent = [
        item
        for item in predictions
        if item["outcome"] in {"rejected_implicit_typing", "typed_partial_match"}
        and item["dismissal_comparison"] is not None
        and (
            item["dismissal_comparison"]["derived_input_relation"]
            in {"diverged_after_shared_prefix", "divergent"}
            or (
                item["dismissal_comparison"].get("post_dismissal_trajectory")
                and item["dismissal_comparison"]["post_dismissal_trajectory"][
                    "later_divergence_observed"
                ]
            )
        )
    ]
    navigation_cancel = [
        item
        for item in predictions
        if item["outcome"] in {"dismissed_navigation", "cancelled_unseen", "invalidated_unseen"}
    ]
    return {
        "outcome_counts": dict(outcomes),
        "acceptance_recorded_exactly_once": len(acceptance) == 1
        and all(item["accepted_edit_delta_linked"] for item in acceptance),
        "typed_match_causally_verified": bool(direct_typed or eventual_typed),
        "eventual_typed_match_after_partial": bool(eventual_typed),
        "typed_match_evidence": typed_evidence,
        "divergent_typing_causally_compared": len(divergent) >= 1,
        "navigation_cancellation_observed": len(navigation_cancel) >= 1,
        "undo_restored_pre_state": len(acceptance) == 1
        and acceptance[0]["undo_restores_pre_state"],
        "buffer_switch_identities_verified": navigation["verified_file_jump_count"] >= 1
        and navigation["distinct_navigation_identity_count"] >= 2,
        "sequence_contiguous": replay["sequence_contiguous"],
        "replay_anchor_deltas_match": replay["replay_mismatch_count"] == 0
        and replay["unanchored_delta_count"] == 0,
    }


def event_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        payload = row["payload"]
        item = {
            "sequence": row["sequence_number"],
            "event_id": row["event_id"],
            "event_type": row["event_type"],
        }
        for key in (
            "prediction_id",
            "request_id",
            "outcome",
            "outcome_source",
            "prediction_lifecycle",
        ):
            if key in payload:
                item[key] = payload[key]
        for key in ("context_blob_hash", "raw_response_hash", "action_blob_hash"):
            if key in payload:
                item[key] = payload[key]
        if row["event_type"] == "prediction_requested":
            for key in (
                "model_alias",
                "model_protocol",
                "model_gguf_sha256",
                "model_revision",
                "precision",
                "wire_version",
                "context_policy_version",
                "pre_state_hash",
                "editable_range",
            ):
                if key in payload:
                    item[key] = payload[key]
            identity = payload.get("file_identity") or payload.get("file")
            if isinstance(identity, str):
                item["file_identity_sha256"] = sha256(identity.encode())
        if row["event_type"] == "prediction_generated":
            for key in (
                "model_alias",
                "model_protocol",
                "model_sha256",
                "canonical_action",
                "stop_type",
                "max_output_tokens",
            ):
                if key in payload:
                    item[key] = payload[key]
        if row["event_type"] == "prediction_shown":
            for key in (
                "active_buffer",
                "focused",
                "proposed_start",
                "proposed_end",
                "proposed_start_byte",
                "proposed_end_byte",
                "proposed_range_end_exclusive",
                "proposed_range_includes_terminator",
            ):
                if key in payload:
                    item[key] = payload[key]
        if row["event_type"] == "prediction_accepted":
            for key in (
                "shown_event_id",
                "applied_through_sequence",
                "proposed_start_byte",
                "proposed_end_byte",
            ):
                if key in payload:
                    item[key] = payload[key]
        if row["event_type"] == "prediction_dismissed":
            for key in (
                "ended_by_event_id",
                "ended_by_sequence",
                "matching_prefix_bytes",
                "diverged_after_prefix",
            ):
                if key in payload:
                    item[key] = payload[key]
        if row["event_type"] == "edit_delta":
            for key in (
                "start_row",
                "old_end_row",
                "new_end_row",
                "change_origin",
                "fileformat_after",
                "eol_after",
            ):
                if key in payload:
                    item[key] = payload[key]
            for field in ("deleted_text", "inserted_text"):
                value = payload.get(field)
                if isinstance(value, str):
                    encoded = value.encode("utf-8")
                    item[f"{field}_bytes"] = len(encoded)
                    item[f"{field}_sha256"] = sha256(encoded)
        if "synthetic" in payload:
            item["synthetic"] = payload["synthetic"] is True
        if payload.get("human_verified") is True:
            item["human_verified"] = False  # scripted artifacts never make a human claim
        result.append(item)
    return result


def snapshot_result(db_path: pathlib.Path, session_id: str) -> dict[str, Any]:
    snapshot, blobs = load_database_snapshot(db_path, session_id)
    global _file_ids_by_repo_path
    _file_ids_by_repo_path = snapshot["file_ids_by_repo_path"]
    replay = replay_session(snapshot["event_rows"], blobs)
    replay["sequence_contiguous"] = True  # load_database_snapshot enforces this before return.
    predictions = validate_models(
        snapshot["event_rows"],
        snapshot["projections"],
        replay,
        snapshot["root_name"],
        snapshot["repo_id"],
        blobs,
    )
    navigation = verify_navigation_identity(snapshot["event_rows"], predictions)
    coverage = coverage_summary(predictions, navigation, replay)
    human_verified = sum(
        1 for row in snapshot["event_rows"] if row["payload"].get("human_verified") is True
    )
    if human_verified:
        raise AuditError("synthetic_session_contains_human_verified_flag")
    if any(not item["synthetic"] or item["review_status"] != "unreviewed" for item in predictions):
        raise AuditError("synthetic_feedback_provenance_invalid")
    if navigation["issues"]:
        raise AuditError(navigation["issues"][0])
    report = {
        "schema": "rust-editor-synthetic-feedback-verification-v1",
        "session_id": session_id,
        "repo_id": snapshot["repo_id"],
        "synthetic": True,
        "human_feedback": False,
        "human_verified_proposal_count": 0,
        "quality_validated": False,
        "training_enabled": False,
        "event_count": snapshot["event_count"],
        "projection_count": snapshot["projection_count"],
        "event_types": dict(
            collections.Counter(row["event_type"] for row in snapshot["event_rows"])
        ),
        "sequence_contiguous": True,
        "sqlite_integrity_check": "ok",
        "sqlite_foreign_key_issues": 0,
        "all_database_blobs": snapshot["blob_records"],
        "all_database_blob_count": len(snapshot["blob_records"]),
        "session_blob_reference_count": snapshot["session_blob_reference_count"],
        "replay": {
            key: replay[key]
            for key in (
                "anchor_count",
                "delta_count",
                "unanchored_delta_count",
                "replay_mismatch_count",
                "replay_segments",
                "path_delta_counts",
            )
        },
        "navigation_identity": navigation,
        "predictions": predictions,
        "prediction_event_summaries": event_summaries(snapshot["event_rows"]),
        "coverage": coverage,
        "database_snapshot": {
            "session_event_digest": snapshot["event_digest"],
            "session_projection_digest": snapshot["projection_digest"],
            "all_database_blob_digest": snapshot["blob_digest"],
            "session_event_count": snapshot["event_count"],
            "session_projection_count": snapshot["projection_count"],
            "global_counts": snapshot["global_counts"],
        },
        "_retry_snapshot": snapshot,
    }
    return report


def collector_url(raw: str) -> str:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise AuditError("collector_url_not_allowed")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise AuditError("collector_url_not_allowed")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        raise AuditError("collector_url_not_allowed") from None
    try:
        port = parsed.port
    except ValueError:
        raise AuditError("collector_url_not_allowed") from None
    if address not in ipaddress.ip_network("100.64.0.0/10") or port != 8787:
        raise AuditError("collector_url_not_allowed")
    return f"http://{address}:8787"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def retry_duplicate_batch(
    raw_url: str, session_id: str, rows: list[dict[str, Any]], before: dict[str, Any]
) -> dict[str, Any]:
    url = collector_url(raw_url) + "/v1/events/batch"
    events = []
    for row in rows:
        cursor = None
        if row["cursor_row"] is not None or row["cursor_col"] is not None:
            cursor = {"row": row["cursor_row"] or 0, "col": row["cursor_col"] or 0}
        event = {
            "protocol_version": 1,
            "event_id": row["event_id"],
            "session_id": row["session_id"],
            "sequence_number": row["sequence_number"],
            "timestamp_ms": row["timestamp_ms"],
            "event_type": row["event_type"],
            "file_id": row["file_id"],
            "changedtick": row["changedtick"],
            "cursor": cursor,
            "mode": row["mode"],
            "payload": row["payload"],
        }
        events.append(event)
    body = json.dumps({"protocol_version": 1, "events": events}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=20) as response:
            response_body = response.read(1024 * 1024 + 1)
            if len(response_body) > 1024 * 1024:
                raise AuditError("duplicate_retry_response_too_large")
            ack = parse_json_object(response_body)
    except urllib.error.HTTPError:
        raise AuditError("duplicate_retry_http_error") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise AuditError("duplicate_retry_transport_error") from None
    except (json.JSONDecodeError, UnicodeError, ValueError):
        raise AuditError("duplicate_retry_ack_malformed") from None

    if (
        ack.get("total") != len(events)
        or ack.get("ingested") != 0
        or ack.get("skipped_duplicate") != len(events)
    ):
        raise AuditError("duplicate_retry_ack_counts_invalid")
    return {
        "authorized_collector_api": True,
        "endpoint_host": urllib.parse.urlsplit(url).hostname,
        "batch_events": len(events),
        "ack": {key: ack[key] for key in ("ingested", "skipped_duplicate", "total")},
        "before_snapshot": {
            "session_event_digest": before["event_digest"],
            "session_projection_digest": before["projection_digest"],
            "all_database_blob_digest": before["blob_digest"],
            "event_count": before["event_count"],
            "projection_count": before["projection_count"],
            "global_counts": before["global_counts"],
        },
    }


def write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    # Drop in-memory-only details (event payloads and blob contents) before serialization.
    clean = {key: value for key, value in report.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--retry-duplicates", action="store_true")
    parser.add_argument(
        "--collector-url", default=os.environ.get("TABCOMPLETE_COLLECTOR_URL", DEFAULT_COLLECTOR)
    )
    args = parser.parse_args()
    try:
        normalized_session_id = str(uuid.UUID(args.session_id))
    except (ValueError, AttributeError):
        parser.error("--session-id must be a UUID")
    if normalized_session_id != args.session_id.lower():
        parser.error("--session-id must use canonical UUID formatting")
    output = args.output or (DEFAULT_REPORT_DIR / f"database-evidence-{args.session_id}.json")
    report: dict[str, Any] = {}
    try:
        report = snapshot_result(args.db, args.session_id)
        retry = None
        if args.retry_duplicates:
            retry = retry_duplicate_batch(
                args.collector_url,
                args.session_id,
                report["_retry_snapshot"]["event_rows"],
                report["_retry_snapshot"],
            )
            after, _after_blobs = load_database_snapshot(args.db, args.session_id)
            before = report["_retry_snapshot"]
            unchanged = (
                after["event_digest"] == before["event_digest"]
                and after["projection_digest"] == before["projection_digest"]
                and after["event_count"] == before["event_count"]
                and after["projection_count"] == before["projection_count"]
                and after["blob_digest"] == before["blob_digest"]
                and after["global_counts"] == before["global_counts"]
            )
            if not unchanged:
                raise AuditError("duplicate_retry_changed_database_snapshot")
            retry["session_rows_unchanged"] = True
            retry["projection_rows_unchanged"] = True
            retry["global_counts_unchanged"] = True
            report["retry_duplicate_delivery"] = retry
        else:
            report["retry_duplicate_delivery"] = {
                "performed": False,
                "reason": "explicit --retry-duplicates option was not supplied",
            }
        report["coverage_failures"] = sorted(
            key for key, passed in report["coverage"].items() if type(passed) is bool and not passed
        )
        if report["replay"]["replay_mismatch_count"] or report["replay"]["unanchored_delta_count"]:
            raise AuditError("buffer_history_replay_mismatch")
        if report["coverage_failures"]:
            raise AuditError("required_synthetic_session_coverage_missing")
        report["ok"] = True
        write_report(output, report)
        print(
            json.dumps(
                {
                    "ok": True,
                    "session_id": args.session_id,
                    "event_count": report["event_count"],
                    "prediction_count": len(report["predictions"]),
                    "blob_count": report["all_database_blob_count"],
                    "outcomes": report["coverage"]["outcome_counts"],
                    "coverage": report["coverage"],
                    "retry_performed": args.retry_duplicates,
                    "report": str(output.resolve()),
                },
                indent=2,
            )
        )
        return 0
    except AuditError as exc:
        # All error values above are fixed codes, never database content.
        failure = {"ok": False, "session_id": args.session_id, "error": str(exc)}
        if "coverage" in report:
            for key in (
                "coverage",
                "coverage_failures",
                "predictions",
                "navigation_identity",
                "event_types",
                "replay",
                "prediction_event_summaries",
                "database_snapshot",
                "sqlite_integrity_check",
                "sqlite_foreign_key_issues",
                "synthetic",
                "human_feedback",
                "human_verified_proposal_count",
                "quality_validated",
                "training_enabled",
                "all_database_blobs",
                "all_database_blob_count",
                "session_blob_reference_count",
                "event_count",
                "projection_count",
                "retry_duplicate_delivery",
            ):
                if key in report:
                    failure[key] = report[key]
            failure["prediction_count"] = len(report.get("predictions", []))
        try:
            write_report(output, failure)
        except OSError:
            pass
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error):
        failure = {
            "ok": False,
            "session_id": args.session_id,
            "error": "audit_io_or_sqlite_failure",
        }
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
