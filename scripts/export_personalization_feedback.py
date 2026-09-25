"""Read-only, conservative export of collector proposal evidence.

This never turns missing, stale, shadow, or unverified outcomes into dislikes.
It emits no code or proposed text; a later reviewed data builder may retrieve
content-addressed payloads after verifying replay and authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "personalization-feedback-evidence-v3"
PREDICTION_TYPES = {
    "prediction_requested",
    "prediction_shown",
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
    "prediction_generated",
    "prediction_dismissed",
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
    blob_hashes = (
        {row[0] for row in connection.execute("SELECT sha256 FROM blobs")}
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blobs'"
        ).fetchone()
        else set()
    )
    connection.close()

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
        if payload.get("synthetic") is not False or payload.get("human_verified") is not True:
            flags.append("human_provenance_unverified")
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
            "request_id": payload.get("request_id"),
            "pre_state_sequence": payload.get("pre_state_sequence"),
            "context_blob_hash": payload.get("context_blob_hash"),
            "raw_response_hash": generated["payload"].get("raw_response_hash")
            if generated
            else None,
            "action_blob_hash": generated["payload"].get("action_blob_hash") if generated else None,
            "runtime_config_hash": payload.get("runtime_config_hash"),
            "context_policy_version": payload.get("context_policy_version"),
            "context_hash": payload.get("context_hash"),
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
            "proposal_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "action": shown["payload"].get("action"),
                        "text": shown["payload"].get("proposed_text"),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if shown
            and isinstance(shown["payload"].get("action"), str)
            and isinstance(shown["payload"].get("proposed_text"), str)
            else None,
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
                "proposal_sha256",
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
                    preferred["proposal_sha256"] == dispreferred["proposal_sha256"]
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
