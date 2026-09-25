import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_personalization_feedback import extract  # noqa: E402


def test_explicit_and_ambiguous_feedback_stay_distinct(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    rows = [
        (
            "e1",
            "s1",
            1,
            "prediction_requested",
            100,
            {"prediction_id": "p1", "context_hash": "h", "synthetic": True},
        ),
        ("e2", "s1", 2, "prediction_shown", 110, {"prediction_id": "p1"}),
        ("e3", "s1", 3, "prediction_rejected", 120, {"prediction_id": "p1"}),
        ("e4", "s1", 5, "prediction_requested", 130, {"prediction_id": "p2"}),
    ]
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?)",
        [
            (eid, sid, seq, kind, at, json.dumps(payload))
            for eid, sid, seq, kind, at, payload in rows
        ],
    )
    conn.commit()
    conn.close()
    result = extract(db)
    assert result["prediction_event_counts"]["prediction_rejected"] == 1
    assert result["evidence"][0]["explicit_outcome"] == "prediction_rejected"
    assert result["evidence"][1]["explicit_outcome"] is None
    assert result["candidate_preference_pairs"] == []
    assert result["readiness"]["enabled"] is False
    assert len(result["sequence_gaps"]) == 1
