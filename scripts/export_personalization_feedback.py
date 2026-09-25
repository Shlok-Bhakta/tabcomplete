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

SCHEMA_VERSION = "personalization-feedback-evidence-v1"
PREDICTION_TYPES = {
    "prediction_requested",
    "prediction_shown",
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
}
RESOLVED = {
    "prediction_accepted",
    "prediction_partially_accepted",
    "prediction_rejected",
}


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
            if kind not in PREDICTION_TYPES:
                continue
            counts[kind] += 1
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                counts["malformed_payload"] += 1
                continue
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
            "model": payload.get("model"),
            "model_revision": payload.get("model_revision"),
            "precision": payload.get("precision"),
            "adapter_identity": payload.get("adapter_identity"),
            "ambiguity_flags": flags,
        }
        evidence.append(record)
        # A preference pair needs two alternatives for the same verified state.
        # One accept or reject event alone does not supply that comparison.

    evidence.sort(key=lambda row: (row["session_id"], row["prediction_id"]))
    distinct_sessions = len({r["session_id"] for r in evidence if not r["ambiguity_flags"]})
    readiness = {
        "enabled": False,
        "defensible_preference_pairs": len(defensible_pairs),
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
                "defensible_pairs": len(result["candidate_preference_pairs"]),
                "sequence_gaps": len(result["sequence_gaps"]),
                "ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
