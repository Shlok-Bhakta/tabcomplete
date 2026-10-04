"""Read-only, conservative export of collector proposal evidence.

This never turns missing, stale, shadow, or unverified outcomes into dislikes.
It emits no code or proposed text; a later reviewed data builder may retrieve
content-addressed payloads after verifying replay and authorization.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "personalization-feedback-evidence-v6"
LEGACY_PROTOCOL_VERSION = "compact-next-edit-v1"
SINGLE_LINE_PROTOCOL_VERSION = "single-line-edit-v1"
FIM_LINE_PROTOCOL_VERSION = "q25-fim-line-completion-v1"
ACTION_BLOB_MAX_BYTES = 8192
PREDICTION_TYPES = {
    "prediction_requested",
    "prediction_shown",
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
    "prediction_generated",
    "prediction_dismissed",
    "prediction_reviewed",
}
LIFECYCLE_EVENT_TYPE = "heartbeat"
RESOLVED = {
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
    "prediction_dismissed",
}


def same_file(row: sqlite3.Row, file_path: str | None) -> bool:
    if not file_path:
        return False
    try:
        return json.loads(row["payload_json"] or "{}").get("path") == file_path
    except json.JSONDecodeError:
        return False


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate action blob key")
        value[key] = item
    return value


def _read_action_blob(
    row: sqlite3.Row | None, expected_sha256: str
) -> dict[str, object] | None:
    """Decode only a small, content-addressed action blob; never return code in export."""
    if row is None or row["original_bytes"] > ACTION_BLOB_MAX_BYTES:
        return None
    stored = bytes(row["content"])
    try:
        compression = row["compression"]
        if compression == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(stored), mode="rb") as stream:
                raw = stream.read(ACTION_BLOB_MAX_BYTES + 1)
        elif compression == "raw":
            raw = stored
        else:
            return None
        if len(raw) > ACTION_BLOB_MAX_BYTES or len(raw) != row["original_bytes"]:
            return None
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            return None
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"kind", "text"}:
        return None
    kind = value.get("kind")
    text = value.get("text")
    if isinstance(kind, str) and kind in {"keep", "delete_line"}:
        return value if text is None else None
    if isinstance(kind, str) and kind in {"replace_line", "insert_before"}:
        return value if isinstance(text, str) and "\n" not in text and "\r" not in text else None
    return None


def _load_action_blob(
    connection: sqlite3.Connection, expected_sha256: str
) -> dict[str, object] | None:
    row = connection.execute(
        "SELECT original_bytes, compression, content FROM blobs WHERE sha256 = ?",
        (expected_sha256,),
    ).fetchone()
    return _read_action_blob(row, expected_sha256)


def _proposal_identity_sha256(protocol: str, action: Mapping[str, object]) -> str:
    identity = {"wire_version": protocol, "action": action}
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _proposal_evidence(
    generated: dict | None,
    shown: dict | None,
    action_blob_loader: Callable[[str], dict[str, object] | None] | None,
    flags: list[str],
) -> dict[str, object | None]:
    generated_payload = generated["payload"] if generated else {}
    shown_payload = shown["payload"] if shown else {}
    generated_protocol = generated_payload.get("wire_version")
    shown_protocol = shown_payload.get("wire_version")
    protocol = generated_protocol or shown_protocol
    if not isinstance(protocol, str) or not protocol:
        protocol = LEGACY_PROTOCOL_VERSION
    if (
        isinstance(generated_protocol, str)
        and isinstance(shown_protocol, str)
        and generated_protocol != shown_protocol
    ):
        flags.append("action_protocol_mismatch")
        return {
            "proposal_sha256": None,
            "proposal_identity_sha256": None,
            "proposal_protocol_version": protocol,
            "canonical_action_kind": generated_payload.get("canonical_action"),
            "proposal_action_source": None,
            "proposal_sha256_semantics": "protocol_mismatch_unusable",
        }

    if protocol in {SINGLE_LINE_PROTOCOL_VERSION, FIM_LINE_PROTOCOL_VERSION}:
        generated_action_hash = generated_payload.get("action_blob_hash")
        shown_action_hash = shown_payload.get("action_blob_hash")
        if (
            protocol == FIM_LINE_PROTOCOL_VERSION
            and isinstance(generated_action_hash, str)
            and isinstance(shown_action_hash, str)
            and generated_action_hash != shown_action_hash
        ):
            flags.append("action_blob_event_mismatch")
            return {
                "proposal_sha256": None,
                "proposal_identity_sha256": None,
                "proposal_protocol_version": protocol,
                "canonical_action_kind": generated_payload.get("canonical_action"),
                "proposal_action_source": None,
                "proposal_sha256_semantics": "fim_canonical_line_action_unverified",
            }
        action_hash = generated_action_hash or shown_action_hash
        action = (
            action_blob_loader(action_hash)
            if action_blob_loader is not None and isinstance(action_hash, str)
            else None
        )
        if action is None:
            flags.append("action_blob_unverified")
            return {
                "proposal_sha256": None,
                "proposal_identity_sha256": None,
                "proposal_protocol_version": protocol,
                "canonical_action_kind": generated_payload.get("canonical_action"),
                "proposal_action_source": None,
                "proposal_sha256_semantics": (
                    "legacy_replacement_equivalence_comparison_only"
                    if protocol == SINGLE_LINE_PROTOCOL_VERSION
                    else "fim_canonical_line_action_unverified"
                ),
            }
        kind = action["kind"]
        if protocol == FIM_LINE_PROTOCOL_VERSION and kind not in {
            "replace_line",
            "insert_before",
        }:
            flags.append("fim_action_kind_unverified")
            return {
                "proposal_sha256": None,
                "proposal_identity_sha256": None,
                "proposal_protocol_version": protocol,
                "canonical_action_kind": kind,
                "proposal_action_source": "verified_action_blob",
                "proposal_sha256_semantics": "fim_line_action_unverified",
            }
        if generated_payload.get("canonical_action") not in (None, kind):
            flags.append("action_blob_event_mismatch")
            return {
                "proposal_sha256": None,
                "proposal_identity_sha256": None,
                "proposal_protocol_version": protocol,
                "canonical_action_kind": kind,
                "proposal_action_source": "verified_action_blob",
                "proposal_sha256_semantics": (
                    "legacy_replacement_equivalence_comparison_only"
                    if protocol == SINGLE_LINE_PROTOCOL_VERSION
                    else "fim_canonical_line_action_unverified"
                ),
            }
        if shown:
            shown_kind = shown_payload.get("action")
            shown_text = shown_payload.get("proposed_text")
            byte_range = _proposed_byte_range(shown)
            expected_shown_text: object = "" if kind == "delete_line" else action["text"]
            invalid_byte_range = (
                byte_range is None or not _v1_range_matches_action(kind, shown_payload)
            )
            if (
                shown_kind != kind
                or (
                    protocol == FIM_LINE_PROTOCOL_VERSION
                    and shown_payload.get("wire_version") != protocol
                )
                or shown_text != expected_shown_text
                or kind == "keep"
                or invalid_byte_range
            ):
                flags.append("action_blob_display_mismatch")
                return {
                    "proposal_sha256": None,
                    "proposal_identity_sha256": None,
                    "proposal_protocol_version": protocol,
                    "canonical_action_kind": kind,
                    "proposal_action_source": "verified_action_blob",
                    "proposal_sha256_semantics": (
                        "legacy_replacement_equivalence_comparison_only"
                        if protocol == SINGLE_LINE_PROTOCOL_VERSION
                        else "fim_canonical_line_action_unverified"
                    ),
                }
            if protocol == SINGLE_LINE_PROTOCOL_VERSION:
                # Keep the legacy proposal hash's semantic shape stable. A v1
                # action is represented as its replacement equivalent; the
                # separate identity hash retains the wire protocol and action.
                legacy_text = "" if kind == "delete_line" else action["text"]
                legacy_hash = hashlib.sha256(
                    json.dumps(
                        {"action": "replace", "text": legacy_text}, sort_keys=True
                    ).encode("utf-8")
                ).hexdigest()
                hash_semantics = "legacy_replacement_equivalence_comparison_only"
            else:
                # FIM line completion has its own canonical action hash, even
                # where its replacement text matches a single-line action.
                legacy_hash = hashlib.sha256(
                    json.dumps(
                        action,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                hash_semantics = "fim_canonical_line_action_comparison_only"
            return {
                "proposal_sha256": legacy_hash,
                "proposal_identity_sha256": _proposal_identity_sha256(protocol, action),
                "proposal_protocol_version": protocol,
                "canonical_action_kind": kind,
                "proposal_action_source": "verified_action_blob",
                "proposal_sha256_semantics": hash_semantics,
            }
        return {
            "proposal_sha256": None,
            "proposal_identity_sha256": None,
            "proposal_protocol_version": protocol,
            "canonical_action_kind": kind,
            "proposal_action_source": "verified_action_blob",
            "proposal_sha256_semantics": (
                "legacy_replacement_equivalence_comparison_only"
                if protocol == SINGLE_LINE_PROTOCOL_VERSION
                else "fim_canonical_line_action_comparison_only"
            ),
        }

    action = shown_payload.get("action")
    text = shown_payload.get("proposed_text")
    if not shown or not isinstance(action, str) or not isinstance(text, str):
        return {
            "proposal_sha256": None,
            "proposal_identity_sha256": None,
            "proposal_protocol_version": protocol,
            "canonical_action_kind": action if isinstance(action, str) else None,
            "proposal_action_source": "shown_event" if shown else None,
            "proposal_sha256_semantics": "legacy_action_text_hash",
        }
    legacy_hash = hashlib.sha256(
        json.dumps({"action": action, "text": text}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    normalized = {"kind": action, "text": text}
    return {
        "proposal_sha256": legacy_hash,
        "proposal_identity_sha256": _proposal_identity_sha256(protocol, normalized),
        "proposal_protocol_version": protocol,
        "canonical_action_kind": action,
        "proposal_action_source": "shown_event",
        "proposal_sha256_semantics": "legacy_action_text_hash",
    }


def _proposed_byte_range(shown: dict | None) -> dict[str, object] | None:
    if not shown:
        return None
    payload = shown["payload"]

    def point(value: object) -> dict[str, int] | None:
        if not isinstance(value, Mapping):
            return None
        row, col = value.get("row"), value.get("col")
        if type(row) is not int or type(col) is not int or row < 0 or col < 0:
            return None
        return {"row": row, "byte_col": col}

    start = point(payload.get("proposed_start"))
    end = point(payload.get("proposed_end"))
    if start is None or end is None:
        return None
    includes_terminator = payload.get("proposed_range_includes_terminator")
    if includes_terminator is not None and type(includes_terminator) is not bool:
        return None
    start_byte, end_byte = payload.get("proposed_start_byte"), payload.get("proposed_end_byte")
    if (start_byte is None) != (end_byte is None):
        return None
    if start_byte is not None and (
        type(start_byte) is not int
        or type(end_byte) is not int
        or start_byte < 0
        or end_byte < start_byte
    ):
        return None
    return {
        "start": start,
        "end": end,
        "start_byte": start_byte,
        "end_byte": end_byte,
        "byte_interval_end_exclusive": True,
        "includes_terminator": includes_terminator,
        "row_base": 0,
        "column_unit": "utf8_byte",
    }


def _v1_range_matches_action(kind: object, payload: Mapping[str, object]) -> bool:
    if payload.get("proposed_range_end_exclusive") is not True:
        return False
    value = _proposed_byte_range({"payload": payload})
    if value is None:
        return False
    start, end = value["start"], value["end"]
    start_byte, end_byte = value["start_byte"], value["end_byte"]
    if (
        not isinstance(start, Mapping)
        or not isinstance(end, Mapping)
        or type(start_byte) is not int
        or type(end_byte) is not int
    ):
        return False
    start_row, start_col = start.get("row"), start.get("byte_col")
    end_row, end_col = end.get("row"), end.get("byte_col")
    if not all(type(number) is int for number in (start_row, start_col, end_row, end_col)):
        return False
    start_row, start_col = cast(int, start_row), cast(int, start_col)
    end_row, end_col = cast(int, end_row), cast(int, end_col)
    start_byte, end_byte = cast(int, start_byte), cast(int, end_byte)
    if start_byte < 0 or end_byte < start_byte:
        return False
    delta = end_byte - start_byte
    coordinate_delta = end_col - start_col if start_row == end_row else None
    includes_terminator = payload.get("proposed_range_includes_terminator")
    if kind == "replace_line":
        return (
            includes_terminator is False
            and start_row == end_row
            and coordinate_delta is not None
            and coordinate_delta >= 0
            and delta == coordinate_delta
        )
    if kind == "insert_before":
        return (
            includes_terminator is False
            and start_row == end_row
            and start_col == end_col
            and delta == 0
        )
    if kind != "delete_line" or type(includes_terminator) is not bool or delta == 0:
        return False
    if end_row == start_row + 1 and end_col == 0:
        return includes_terminator is True
    if end_row != start_row or coordinate_delta is None or coordinate_delta < 0:
        return False
    if includes_terminator:
        return delta - coordinate_delta in {1, 2}
    return delta == coordinate_delta


def _accepted_action_range(resolution: dict | None) -> dict[str, object] | None:
    if not resolution or resolution["type"] != "prediction_accepted":
        return None
    payload = resolution["payload"]
    value = payload.get("editable_range")
    if not isinstance(value, Mapping):
        return None
    keys = ("start_row", "start_col", "end_row", "end_col", "start_byte", "end_byte")
    if any(type(value.get(key)) is not int or value[key] < 0 for key in keys):
        return None
    if value["end_byte"] < value["start_byte"]:
        return None
    if value.get("end_exclusive") is not True or type(value.get("includes_terminator")) is not bool:
        return None
    return {
        "start": {"row": value["start_row"], "byte_col": value["start_col"]},
        "end": {"row": value["end_row"], "byte_col": value["end_col"]},
        "start_byte": value["start_byte"],
        "end_byte": value["end_byte"],
        "includes_terminator": value["includes_terminator"],
        "byte_interval_end_exclusive": True,
        "row_base": 0,
        "column_unit": "utf8_byte",
    }


def observation_window(
    anchor: dict | None, later: list[sqlite3.Row], file_path: str | None, milliseconds: int
) -> dict:
    if not anchor:
        return {"observed": False, "censored": True, "edit_event_ids": []}
    limit = anchor["timestamp_ms"] + milliseconds
    boundary = next(
        (row for row in later if row["event_type"] in
         {"buffer_leave", "buffer_close", "session_end"}), None
    )
    usable = [
        row for row in later
        if not boundary or row["sequence_number"] < boundary["sequence_number"]
    ]
    observed = any(row["timestamp_ms"] >= limit for row in usable)
    edits = [
        row["event_id"]
        for row in usable
        if row["event_type"] == "edit_delta"
        and row["timestamp_ms"] <= limit
        and same_file(row, file_path)
    ]
    result = {"observed": observed, "censored": not observed, "edit_event_ids": edits}
    if boundary:
        result["boundary_event_id"] = boundary["event_id"]
    return result


def gap_affects_proposal(
    rows: list[sqlite3.Row], gaps: list[dict], requested: dict | None,
    resolution: dict | None, file_path: str | None, blob_hashes: set[str]
) -> bool:
    if not requested:
        return bool(gaps)
    end = resolution["sequence"] if resolution else requested["sequence"]
    for gap in gaps:
        if gap["before"] > end:
            continue
        if gap["before"] > requested["sequence"]:
            return True
        recovered = False
        for row in rows:
            if not gap["before"] <= row["sequence_number"] < requested["sequence"]:
                continue
            if row["event_type"] not in {"buffer_open", "buffer_write"}:
                continue
            if not same_file(row, file_path):
                continue
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                continue
            if payload.get("blob_uploaded") and payload.get("content_hash") in blob_hashes:
                recovered = True
        if not recovered:
            return True
    return False


def extract(db_path: Path) -> dict:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    sessions: dict[str, list[sqlite3.Row]] = defaultdict(list)
    snapshot_hash = hashlib.sha256()
    for row in connection.execute(
        "SELECT event_id, session_id, sequence_number, event_type, timestamp_ms, "
        "payload_json FROM events ORDER BY session_id, sequence_number"
    ):
        snapshot_hash.update(json.dumps(tuple(row), ensure_ascii=False).encode())
        sessions[row["session_id"]].append(row)
    projections = {}
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prediction_projection'"
    ).fetchone():
        projections = {
            (row["session_id"], row["prediction_id"]): dict(row)
            for row in connection.execute(
                "SELECT session_id,prediction_id,outcome,projection_version,"
                "updated_through_sequence FROM prediction_projection"
            )
        }
    has_blobs = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blobs'"
    ).fetchone()
    blob_hashes = (
        {row[0] for row in connection.execute("SELECT sha256 FROM blobs")}
        if has_blobs
        else set()
    )
    def action_blob_loader(sha256: str) -> dict[str, object] | None:
        if not has_blobs:
            return None
        return _load_action_blob(connection, sha256)

    sequence_gaps = []
    counts: Counter[str] = Counter()
    proposals: dict[tuple[str, str], dict] = {}
    for session_id, rows in sessions.items():
        previous = 0
        for row in rows:
            sequence = row["sequence_number"]
            if sequence != previous + 1:
                sequence_gaps.append(
                    {"session_id": session_id, "after": previous, "before": sequence}
                )
            previous = sequence
            kind = row["event_type"]
            if kind not in PREDICTION_TYPES and kind != LIFECYCLE_EVENT_TYPE:
                continue
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                counts["malformed_payload"] += 1
                continue
            if kind == LIFECYCLE_EVENT_TYPE:
                lifecycle = payload.get("prediction_lifecycle")
                if not isinstance(lifecycle, str) or not lifecycle:
                    continue
                counts["lifecycle_" + lifecycle] += 1
            else:
                counts[kind] += 1
            prediction_id = payload.get("prediction_id")
            if not isinstance(prediction_id, str) or not prediction_id:
                counts["missing_prediction_id"] += 1
                continue
            key = (session_id, prediction_id)
            proposal = proposals.setdefault(
                key, {"session_id": session_id, "prediction_id": prediction_id, "events": []}
            )
            proposal["events"].append(
                {
                    "type": kind,
                    "sequence": sequence,
                    "timestamp_ms": row["timestamp_ms"],
                    "event_id": row["event_id"],
                    "payload": payload,
                }
            )

    evidence = []
    defensible_pairs: list[dict] = []
    for proposal in proposals.values():
        events = proposal["events"]
        by_type = {event["type"]: event for event in events}
        requested = by_type.get("prediction_requested")
        shown = by_type.get("prediction_shown")
        resolutions = [event for event in events if event["type"] in RESOLVED]
        primary = [
            event
            for event in resolutions
            if event["type"]
            in {"prediction_accepted", "prediction_partially_accepted", "prediction_dismissed"}
        ]
        if not primary:
            primary = [event for event in resolutions if event["type"] == "prediction_rejected"]
        resolution = primary[0] if len(primary) == 1 else None
        reviews = [event for event in events if event["type"] == "prediction_reviewed"]
        lifecycle_events = [event for event in events if event["type"] == LIFECYCLE_EVENT_TYPE]
        flags = []
        if not requested:
            flags.append("missing_request")
        if not shown:
            flags.append("never_shown")
        if len(primary) != 1:
            flags.append("missing_or_multiple_explicit_resolutions")
        if requested and shown and requested["sequence"] >= shown["sequence"]:
            flags.append("bad_request_display_order")
        if shown and resolution and shown["sequence"] >= resolution["sequence"]:
            flags.append("bad_display_resolution_order")
        payload = requested["payload"] if requested else {}
        file_path = payload.get("file")
        if requested:
            prior_anchors = [
                row for row in sessions[proposal["session_id"]]
                if row["sequence_number"] < requested["sequence"]
                and row["event_type"] in {"buffer_open", "buffer_write"}
                and same_file(row, file_path)
            ]
            valid_anchor = False
            for row in prior_anchors:
                try:
                    anchor_payload = json.loads(row["payload_json"] or "{}")
                except json.JSONDecodeError:
                    continue
                if (anchor_payload.get("blob_uploaded") is True
                    and anchor_payload.get("content_hash") in blob_hashes):
                    valid_anchor = True
            if not valid_anchor:
                flags.append("missing_pre_state_anchor")
        review_gap = bool(
            len(reviews) == 1 and resolution
            and any(
                gap["session_id"] == proposal["session_id"]
                and gap["after"] >= resolution["sequence"]
                and gap["before"] <= reviews[0]["sequence"]
                for gap in sequence_gaps
            )
        )
        verified_review = (
            len(reviews) == 1
            and resolution is not None
            and requested is not None
            and shown is not None
            and requested["sequence"] < shown["sequence"] < resolution["sequence"]
            and shown["payload"].get("active_buffer") is True
            and shown["payload"].get("focused") is True
            and not review_gap
            and payload.get("synthetic") is False
            and reviews[0]["sequence"] > resolution["sequence"]
            and reviews[0]["payload"].get("synthetic") is False
            and reviews[0]["payload"].get("human_verified") is True
            and reviews[0]["payload"].get("review_source")
            == "explicit_editor_confirmation"
            and reviews[0]["payload"].get("resolution_event_id")
            == resolution["event_id"]
            and reviews[0]["payload"].get("outcome")
            == (
                "accepted" if resolution["type"] == "prediction_accepted"
                else "rejected_explicit" if resolution["type"] == "prediction_dismissed"
                and resolution["payload"].get("outcome") == "rejected_explicit"
                else None
            )
        )
        if not verified_review:
            flags.append("human_provenance_unverified")
        if len(reviews) > 1:
            flags.append("multiple_review_events")
        if review_gap:
            flags.append("review_interval_sequence_gap")
        if not payload.get("context_hash") or not payload.get("pre_state_hash"):
            flags.append("pre_state_unverified")
        if payload.get("context_blob_hash") and payload["context_blob_hash"] not in blob_hashes:
            flags.append("context_blob_missing")
        explicit_outcome = resolution["type"] if resolution else None
        outcome = explicit_outcome
        if outcome == "prediction_accepted":
            outcome = "accepted"
        elif outcome == "prediction_partially_accepted":
            outcome = "partially_accepted"
        elif outcome == "prediction_rejected":
            outcome = "rejected_explicit"
        elif resolution and outcome == "prediction_dismissed":
            outcome = resolution["payload"].get("outcome", "dismissed_unknown")
        if outcome is None and not shown and any(
            event["payload"].get("prediction_lifecycle") == "automatic_policy_suppressed"
            for event in lifecycle_events
        ):
            # A provider action withheld by display policy is never a human dislike.
            outcome = "automatic_policy_suppressed"
        anchor = resolution or shown
        if gap_affects_proposal(
            sessions[proposal["session_id"]],
            [gap for gap in sequence_gaps if gap["session_id"] == proposal["session_id"]],
            requested, resolution, file_path, blob_hashes,
        ):
            flags.append("unrecovered_sequence_gap")
        later = (
            [
                row
                for row in sessions[proposal["session_id"]]
                if row["sequence_number"] > anchor["sequence"]
            ]
            if anchor
            else []
        )

        next_save = next(
            (
                row
                for row in later
                if row["event_type"] == "buffer_write" and same_file(row, file_path)
            ),
            None,
        )
        linked_deltas = [
            row["event_id"]
            for row in later
            if row["event_type"] == "edit_delta" and same_file(row, file_path)
        ]
        applied_rows = (
            [
                row for row in sessions[proposal["session_id"]]
                if shown and resolution
                and shown["sequence"] < row["sequence_number"] <= resolution["sequence"]
                and row["event_type"] == "edit_delta" and same_file(row, file_path)
            ]
            if outcome == "accepted" else []
        )
        undo_event_id = None
        if applied_rows:
            try:
                applied = json.loads(applied_rows[-1]["payload_json"] or "{}")
                for row in later:
                    if row["event_type"] != "edit_delta" or not same_file(row, file_path):
                        continue
                    candidate = json.loads(row["payload_json"] or "{}")
                    if (candidate.get("start_row") == applied.get("start_row")
                        and candidate.get("deleted_text") == applied.get("inserted_text")
                        and candidate.get("inserted_text") == applied.get("deleted_text")):
                        undo_event_id = row["event_id"]
                        break
            except json.JSONDecodeError:
                flags.append("malformed_delta_payload")
        dismissed = by_type.get("prediction_dismissed")
        generated = by_type.get("prediction_generated")
        if generated and generated["payload"].get("raw_response_hash") not in blob_hashes:
            flags.append("raw_response_blob_missing")
        if generated and generated["payload"].get("action_blob_hash") not in blob_hashes:
            flags.append("action_blob_missing")
        proposal_action = _proposal_evidence(generated, shown, action_blob_loader, flags)
        action_protocol = proposal_action["proposal_protocol_version"]
        generated_payload = generated["payload"] if generated else {}
        shown_payload = shown["payload"] if shown else {}
        resolution_payload = resolution["payload"] if resolution else {}
        protocol_observations = [
            value
            for value in (
                payload.get("wire_version"),
                generated_payload.get("wire_version"),
                shown_payload.get("wire_version"),
                resolution_payload.get("wire_version"),
            )
            if isinstance(value, str) and value
        ]
        if (
            action_protocol in {SINGLE_LINE_PROTOCOL_VERSION, FIM_LINE_PROTOCOL_VERSION}
            and any(value != action_protocol for value in protocol_observations)
            and "action_protocol_mismatch" not in flags
        ):
            flags.append("action_protocol_mismatch")
        model_protocol_observations = [
            value
            for value in (
                payload.get("model_protocol"),
                generated_payload.get("model_protocol"),
            )
            if isinstance(value, str) and value
        ]
        if (
            action_protocol in {SINGLE_LINE_PROTOCOL_VERSION, FIM_LINE_PROTOCOL_VERSION}
            and any(value != action_protocol for value in model_protocol_observations)
            and "action_protocol_mismatch" not in flags
        ):
            flags.append("action_protocol_mismatch")

        model_hashes = [
            value
            for value in (
                payload.get("model_gguf_sha256"),
                generated_payload.get("model_gguf_sha256"),
                generated_payload.get("model_sha256"),
            )
            if isinstance(value, str) and value
        ]
        if len(set(model_hashes)) > 1:
            flags.append("model_identity_mismatch")
        artifact_hashes = [
            value
            for value in (
                payload.get("artifact_manifest_sha256"),
                generated_payload.get("artifact_manifest_sha256"),
            )
            if isinstance(value, str) and value
        ]
        if len(set(artifact_hashes)) > 1:
            flags.append("model_artifact_identity_mismatch")

        context_hashes = [
            value
            for value in (
                payload.get("context_hash"),
                generated_payload.get("context_hash"),
                shown_payload.get("context_hash"),
            )
            if isinstance(value, str) and value
        ]
        if len(set(context_hashes)) > 1:
            flags.append("context_identity_mismatch")

        tokenizer_hashes = [
            value
            for value in (
                payload.get("tokenizer_sha256"),
                generated_payload.get("tokenizer_sha256"),
            )
            if isinstance(value, str) and value
        ]
        if len(set(tokenizer_hashes)) > 1:
            flags.append("tokenizer_identity_mismatch")
        for tokenizer_hash_field in (
            "tokenizer_contract_sha256",
            "tokenizer_vocab_ids_sha256",
        ):
            observations = [
                value
                for value in (
                    payload.get(tokenizer_hash_field),
                    generated_payload.get(tokenizer_hash_field),
                )
                if isinstance(value, str) and value
            ]
            if len(set(observations)) > 1:
                flags.append("tokenizer_identity_mismatch")
        if action_protocol == FIM_LINE_PROTOCOL_VERSION and not tokenizer_hashes:
            # Historical or partial FIM records may omit the tokenizer
            # digest. Preserve that gap instead of inferring it from a model
            # alias or a context hash.
            flags.append("tokenizer_identity_unverified")

        accepted_action_range = _accepted_action_range(resolution)
        if (
            action_protocol in {SINGLE_LINE_PROTOCOL_VERSION, FIM_LINE_PROTOCOL_VERSION}
            and resolution
            and resolution["type"] == "prediction_accepted"
        ):
            if (
                resolution["payload"].get("wire_version") != action_protocol
                or resolution["payload"].get("action")
                != proposal_action["canonical_action_kind"]
            ):
                flags.append("accepted_action_identity_mismatch")
            shown_action_range = _proposed_byte_range(shown)
            if (
                shown_action_range is None
                or accepted_action_range is None
                or any(
                    shown_action_range.get(key) != accepted_action_range.get(key)
                    for key in (
                        "start",
                        "end",
                        "start_byte",
                        "end_byte",
                        "includes_terminator",
                    )
                )
            ):
                flags.append("accepted_action_range_mismatch")
        projection = projections.get((proposal["session_id"], proposal["prediction_id"]))
        if projection and projection["outcome"] != outcome and outcome is not None:
            flags.append("projection_outcome_mismatch")
        record = {
            "session_id": proposal["session_id"],
            "prediction_id": proposal["prediction_id"],
            "requested": bool(requested),
            "shown": bool(shown),
            "explicit_outcome": explicit_outcome,
            "outcome": outcome,
            "outcome_source": resolution["payload"].get("outcome_source") if resolution else None,
            "request_event_id": requested["event_id"] if requested else None,
            "display_event_id": shown["event_id"] if shown else None,
            "resolution_event_id": resolution["event_id"] if resolution else None,
            "review_event_id": reviews[0]["event_id"] if verified_review else None,
            "review_event_count": len(reviews),
            "human_review_confirmed": bool(verified_review),
            "request_id": payload.get("request_id"),
            "pre_state_sequence": payload.get("pre_state_sequence"),
            "raw_response_hash": generated["payload"].get("raw_response_hash")
            if generated
            else None,
            "action_blob_hash": generated["payload"].get("action_blob_hash") if generated else None,
            "runtime_config_hash": payload.get("runtime_config_hash"),
            "context_policy_version": payload.get("context_policy_version"),
            "context_layout": payload.get("context_layout"),
            "model_alias": payload.get("model_alias"),
            "model_protocol": payload.get("model_protocol")
            or generated_payload.get("model_protocol"),
            "model_gguf_sha256": payload.get("model_gguf_sha256")
            or generated_payload.get("model_gguf_sha256")
            or generated_payload.get("model_sha256"),
            "artifact_manifest_sha256": payload.get("artifact_manifest_sha256")
            or generated_payload.get("artifact_manifest_sha256"),
            "tokenizer_id": payload.get("tokenizer_id") or generated_payload.get("tokenizer_id"),
            "tokenizer_revision": payload.get("tokenizer_revision")
            or generated_payload.get("tokenizer_revision"),
            "tokenizer_sha256": payload.get("tokenizer_sha256")
            or generated_payload.get("tokenizer_sha256"),
            "tokenizer_contract_sha256": payload.get("tokenizer_contract_sha256")
            or generated_payload.get("tokenizer_contract_sha256"),
            "tokenizer_vocab_ids_sha256": payload.get("tokenizer_vocab_ids_sha256")
            or generated_payload.get("tokenizer_vocab_ids_sha256"),
            "tokenizer_vocab_size": payload.get("tokenizer_vocab_size")
            or generated_payload.get("tokenizer_vocab_size"),
            "context_hash": payload.get("context_hash") or generated_payload.get("context_hash"),
            "context_blob_hash": payload.get("context_blob_hash")
            or generated_payload.get("context_blob_hash"),
            "pre_state_hash": payload.get("pre_state_hash"),
            "file_sha256": hashlib.sha256(file_path.encode()).hexdigest()
            if isinstance(file_path, str)
            else None,
            "region_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "start": shown["payload"].get("proposed_start"),
                        "end": shown["payload"].get("proposed_end"),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if shown
            and shown["payload"].get("proposed_start") is not None
            and shown["payload"].get("proposed_end") is not None
            else None,
            **proposal_action,
            "proposed_byte_range": _proposed_byte_range(shown),
            "accepted_action_byte_range": accepted_action_range,
            "display_timestamp_ms": shown["timestamp_ms"] if shown else None,
            "dismissal_timestamp_ms": dismissed["timestamp_ms"] if dismissed else None,
            "visible_duration_ms": dismissed["payload"].get("visible_duration_ms")
            if dismissed
            else None,
            "ended_by_event_id": dismissed["payload"].get("ended_by_event_id")
            if dismissed
            else None,
            "active_buffer_at_display": shown["payload"].get("active_buffer") if shown else None,
            "focused_at_display": shown["payload"].get("focused") if shown else None,
            "raw_observation_event_ids": [event["event_id"] for event in events],
            "applied_delta_event_ids": [row["event_id"] for row in applied_rows],
            "linked_subsequent_delta_event_ids": linked_deltas,
            "projection": projection,
            "model": payload.get("model"),
            "model_revision": payload.get("model_revision"),
            "precision": payload.get("precision"),
            "adapter_identity": payload.get("adapter_identity"),
            "lifecycle": [
                {
                    "event_id": event["event_id"],
                    "kind": event["payload"]["prediction_lifecycle"],
                    "reason": event["payload"].get("reason"),
                    "timestamp_ms": event["timestamp_ms"],
                }
                for event in lifecycle_events
            ],
            "five_second_outcome": observation_window(anchor, later, file_path, 5_000),
            "thirty_second_outcome": observation_window(anchor, later, file_path, 30_000),
            "next_save_event_id": next_save["event_id"] if next_save else None,
            "next_save_censored": next_save is None,
            "undo_event_id": undo_event_id,
            "undo_semantics_observable": undo_event_id is not None,
            "ambiguity_flags": flags,
        }
        evidence.append(record)

    connection.close()

    # Candidate comparisons require the same state, file, region and distinct
    # displayed actions. Replay verification remains a separate promotion gate.
    by_state: dict[tuple, list[dict]] = defaultdict(list)
    for record in evidence:
        if "human_provenance_unverified" in record["ambiguity_flags"]:
            continue
        if not all(
            record.get(key)
            for key in (
                "context_hash",
                "pre_state_hash",
                "file_sha256",
                "region_sha256",
                "proposal_identity_sha256",
            )
        ):
            continue
        state_key = (
            record["session_id"],
            record["context_hash"],
            record["pre_state_hash"],
            record["file_sha256"],
            record["region_sha256"],
        )
        by_state[state_key].append(record)
    for peers in by_state.values():
        accepted = [row for row in peers if row["outcome"] == "accepted"]
        rejected = [
            row
            for row in peers
            if row["outcome"] == "rejected_explicit" or row["outcome"] == "prediction_rejected"
        ]
        for preferred in accepted:
            for dispreferred in rejected:
                if (
                    preferred["proposal_identity_sha256"]
                    == dispreferred["proposal_identity_sha256"]
                    or preferred["proposal_protocol_version"]
                    != dispreferred["proposal_protocol_version"]
                    or abs(preferred["display_timestamp_ms"] - dispreferred["display_timestamp_ms"])
                    > 300_000
                ):
                    continue
                flags = sorted(
                    set(preferred["ambiguity_flags"] + dispreferred["ambiguity_flags"])
                    | {"replay_not_verified"}
                )
                defensible_pairs.append(
                    {
                        "session_id": preferred["session_id"],
                        "context_hash": preferred["context_hash"],
                        "pre_state_hash": preferred["pre_state_hash"],
                        "preferred_prediction_id": preferred["prediction_id"],
                        "dispreferred_prediction_id": dispreferred["prediction_id"],
                        "preferred_proposal_identity_sha256": preferred[
                            "proposal_identity_sha256"
                        ],
                        "dispreferred_proposal_identity_sha256": dispreferred[
                            "proposal_identity_sha256"
                        ],
                        "proposal_protocol_version": preferred[
                            "proposal_protocol_version"
                        ],
                        "preferred_resolution_event_id": preferred["resolution_event_id"],
                        "dispreferred_resolution_event_id": dispreferred["resolution_event_id"],
                        "ambiguity_flags": flags,
                    }
                )

    evidence.sort(key=lambda row: (row["session_id"], row["prediction_id"]))
    distinct_sessions = len(
        {pair["session_id"] for pair in defensible_pairs if not pair["ambiguity_flags"]}
    )
    readiness = {
        "enabled": False,
        "defensible_preference_pairs": sum(
            not pair["ambiguity_flags"] for pair in defensible_pairs
        ),
        "distinct_verified_sessions": distinct_sessions,
        "minimum_pairs": 256,
        "minimum_sessions": 3,
        "held_out_temporal_session_split": False,
        "generic_regression_set_verified": False,
        "training_memory_verified": False,
        "inference_contention_cleared": False,
        "rollback_artifact_verified": False,
        "ready": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at": datetime.now(UTC).isoformat(),
        "source_event_snapshot_sha256": snapshot_hash.hexdigest(),
        "sessions_seen": len(sessions),
        "prediction_event_counts": dict(counts),
        "sequence_gaps": sequence_gaps,
        "evidence": evidence,
        "accepted_actions": [
            row["prediction_id"] for row in evidence if row["outcome"] == "accepted"
        ],
        "dismissals": [
            {
                "prediction_id": row["prediction_id"],
                "outcome": row["outcome"],
                "ended_by_event_id": row["ended_by_event_id"],
            }
            for row in evidence
            if row["outcome"]
            in {
                "rejected_explicit",
                "rejected_implicit_typing",
                "typed_match",
                "typed_partial_match",
                "dismissed_navigation",
                "expired",
                "dismissed_editor_change",
            }
        ],
        "candidate_preference_pairs": defensible_pairs,
        "readiness": readiness,
        "content_exported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("/mnt/ssd/collector-data/collector.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "sessions": result["sessions_seen"],
                "proposals": len(result["evidence"]),
                "candidate_pairs": len(result["candidate_preference_pairs"]),
                "defensible_pairs": result["readiness"]["defensible_preference_pairs"],
                "sequence_gaps": len(result["sequence_gaps"]),
                "ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
