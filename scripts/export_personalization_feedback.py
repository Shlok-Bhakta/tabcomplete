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

SCHEMA_VERSION = "personalization-feedback-evidence-v2"
PREDICTION_TYPES = {
    "prediction_requested",
    "prediction_shown",
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
}
LIFECYCLE_EVENT_TYPE = "heartbeat"
RESOLVED = {
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
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
    observed = any(row["timestamp_ms"] >= limit for row in later)
    edits = [
        row["event_id"]
        for row in later
        if row["event_type"] == "edit_delta"
        and row["timestamp_ms"] <= limit
        and same_file(row, file_path)
    ]
    return {"observed": observed, "censored": not observed, "edit_event_ids": edits}


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
        lifecycle_events = [event for event in events if event["type"] == LIFECYCLE_EVENT_TYPE]
        flags = []
        if not requested:
            flags.append("missing_request")
        if not shown:
            flags.append("never_shown")
        if len(resolutions) != 1:
            flags.append("missing_or_multiple_explicit_resolutions")
        if requested and shown and requested["sequence"] >= shown["sequence"]:
            flags.append("bad_request_display_order")
        if shown and resolutions and shown["sequence"] >= resolutions[0]["sequence"]:
            flags.append("bad_display_resolution_order")
        payload = requested["payload"] if requested else {}
        if payload.get("synthetic") is not False or payload.get("human_verified") is not True:
            flags.append("human_provenance_unverified")
        if not payload.get("context_hash") or not payload.get("pre_state_hash"):
            flags.append("pre_state_unverified")
        if any(gap["session_id"] == proposal["session_id"] for gap in sequence_gaps):
            flags.append("session_sequence_gap")
        outcome = resolutions[0]["type"] if len(resolutions) == 1 else None
        anchor = resolutions[0] if len(resolutions) == 1 else shown
        file_path = payload.get("file")
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
        record = {
            "session_id": proposal["session_id"],
            "prediction_id": proposal["prediction_id"],
            "requested": bool(requested),
            "shown": bool(shown),
            "explicit_outcome": outcome,
            "request_event_id": requested["event_id"] if requested else None,
            "display_event_id": shown["event_id"] if shown else None,
            "resolution_event_id": resolutions[0]["event_id"] if len(resolutions) == 1 else None,
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
            "undo_event_id": None,
            "undo_semantics_observable": False,
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
        key = (
            record["session_id"],
            record["context_hash"],
            record["pre_state_hash"],
            record["file_sha256"],
            record["region_sha256"],
        )
        by_state[key].append(record)
    for peers in by_state.values():
        accepted = [row for row in peers if row["explicit_outcome"] == "prediction_accepted"]
        rejected = [row for row in peers if row["explicit_outcome"] == "prediction_rejected"]
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
