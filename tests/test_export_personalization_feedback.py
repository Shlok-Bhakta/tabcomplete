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
        (
            "e5",
            "s1",
            6,
            "heartbeat",
            140,
            {"prediction_id": "p2", "prediction_lifecycle": "transport_failure"},
        ),
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
    assert result["evidence"][1]["lifecycle"][0]["kind"] == "transport_failure"
    assert result["prediction_event_counts"]["lifecycle_transport_failure"] == 1
    assert result["candidate_preference_pairs"] == []
    assert result["readiness"]["enabled"] is False
    assert len(result["sequence_gaps"]) == 1


def test_delayed_edits_and_save_are_evidence_not_preference_pairs(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    rows = [
        (
            "r",
            "s",
            1,
            "prediction_requested",
            0,
            {
                "prediction_id": "p",
                "file": "/tmp/example.py",
                "context_hash": "c",
                "pre_state_hash": "h",
                "synthetic": False,
                "human_verified": True,
            },
        ),
        ("d", "s", 2, "prediction_shown", 10, {"prediction_id": "p"}),
        ("x", "s", 3, "prediction_rejected", 20, {"prediction_id": "p"}),
        ("e", "s", 4, "edit_delta", 1000, {"path": "/tmp/example.py"}),
        ("other", "s", 5, "edit_delta", 2000, {"path": "/tmp/other.py"}),
        ("tick5", "s", 6, "heartbeat", 6000, {}),
        ("save", "s", 7, "buffer_write", 20000, {"path": "/tmp/example.py"}),
        ("tick30", "s", 8, "heartbeat", 31000, {}),
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
    evidence = result["evidence"][0]
    assert evidence["explicit_outcome"] == "prediction_rejected"
    assert evidence["five_second_outcome"] == {
        "observed": True,
        "censored": False,
        "edit_event_ids": ["e"],
    }
    assert evidence["thirty_second_outcome"]["edit_event_ids"] == ["e"]
    assert evidence["next_save_event_id"] == "save"
    assert evidence["undo_semantics_observable"] is False
    assert result["candidate_preference_pairs"] == []


def test_same_state_opposite_decisions_are_ambiguous_candidate_pair(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    base = {
        "file": "/tmp/example.py",
        "context_hash": "same-context",
        "pre_state_hash": "same-state",
        "synthetic": False,
        "human_verified": True,
    }
    region = {"proposed_start": {"row": 0, "col": 0}, "proposed_end": {"row": 0, "col": 1}}
    rows = [
        ("r1", 1, "prediction_requested", 0, {**base, "prediction_id": "a"}),
        (
            "d1",
            2,
            "prediction_shown",
            1,
            {**region, "prediction_id": "a", "action": "replace", "proposed_text": "x",
             "active_buffer": True, "focused": True},
        ),
        ("a1", 3, "prediction_accepted", 2, {"prediction_id": "a"}),
        ("v1", 4, "prediction_reviewed", 3, {
            "prediction_id": "a", "synthetic": False, "human_verified": True,
            "review_source": "explicit_editor_confirmation",
            "resolution_event_id": "a1", "outcome": "accepted",
        }),
        ("r2", 5, "prediction_requested", 4, {**base, "prediction_id": "b"}),
        (
            "d2",
            6,
            "prediction_shown",
            5,
            {**region, "prediction_id": "b", "action": "replace", "proposed_text": "y",
             "active_buffer": True, "focused": True},
        ),
        ("x2", 7, "prediction_dismissed", 6, {
            "prediction_id": "b", "outcome": "rejected_explicit",
        }),
        ("v2", 8, "prediction_reviewed", 7, {
            "prediction_id": "b", "synthetic": False, "human_verified": True,
            "review_source": "explicit_editor_confirmation",
            "resolution_event_id": "x2", "outcome": "rejected_explicit",
        }),
    ]
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?)",
        [(eid, "s", seq, kind, at, json.dumps(payload)) for eid, seq, kind, at, payload in rows],
    )
    conn.commit()
    conn.close()
    result = extract(db)
    assert len(result["candidate_preference_pairs"]) == 1
    assert result["candidate_preference_pairs"][0]["ambiguity_flags"] == [
        "missing_pre_state_anchor", "replay_not_verified",
    ]
    assert result["readiness"]["defensible_preference_pairs"] == 0


def test_legacy_request_claim_and_wrong_review_link_do_not_verify(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    rows = [
        ("r", 1, "prediction_requested", {"prediction_id": "p", "synthetic": False,
             "human_verified": True}),
        ("s", 2, "prediction_shown", {"prediction_id": "p"}),
        ("a", 3, "prediction_accepted", {"prediction_id": "p"}),
        ("v", 4, "prediction_reviewed", {"prediction_id": "p", "synthetic": False,
             "human_verified": True, "review_source": "explicit_editor_confirmation",
             "resolution_event_id": "wrong", "outcome": "accepted"}),
    ]
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?)",
        [(eid, "s", seq, kind, seq * 100, json.dumps(payload))
         for eid, seq, kind, payload in rows],
    )
    conn.commit()
    conn.close()
    record = extract(db)["evidence"][0]
    assert record["human_review_confirmed"] is False
    assert record["review_event_id"] is None
    assert "human_provenance_unverified" in record["ambiguity_flags"]


def test_review_needs_visible_display_and_contiguous_outcome_link(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    rows = [
        ("r", 1, "prediction_requested", {"prediction_id": "p", "synthetic": False}),
        ("a", 2, "prediction_accepted", {"prediction_id": "p"}),
        ("v", 4, "prediction_reviewed", {"prediction_id": "p", "synthetic": False,
             "human_verified": True, "review_source": "explicit_editor_confirmation",
             "resolution_event_id": "a", "outcome": "accepted"}),
    ]
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?)",
        [(eid, "s", seq, kind, seq * 100, json.dumps(payload))
         for eid, seq, kind, payload in rows],
    )
    conn.commit()
    conn.close()
    record = extract(db)["evidence"][0]
    assert record["human_review_confirmed"] is False
    assert "never_shown" in record["ambiguity_flags"]
    assert "review_interval_sequence_gap" in record["ambiguity_flags"]
