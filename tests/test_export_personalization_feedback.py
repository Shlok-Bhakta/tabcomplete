import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_personalization_feedback import extract  # noqa: E402


@pytest.mark.parametrize("neutral_outcome", [
    "dismissed_navigation", "dismissed_editor_change", "expired",
    "cancelled_by_mode", "rejected_implicit_typing",
])
def test_navigation_and_ambiguous_dismissals_are_not_negative_pairs(
    tmp_path: Path, neutral_outcome: str,
) -> None:
    """Explicit same-state rejection is a control; neutral observations never join it."""
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    base = {"file": "/tmp/example.py", "context_hash": "same-context",
            "pre_state_hash": "same-state", "synthetic": False}
    sequence = 0
    for prediction, text, outcome in [
        ("accepted", "x", "accepted"),
        ("explicit", "y", "rejected_explicit"),
        ("neutral", "z", neutral_outcome),
    ]:
        rows = [
            ("request", "prediction_requested", {**base, "prediction_id": prediction}),
            ("shown", "prediction_shown", {
                "prediction_id": prediction, "action": "replace", "proposed_text": text,
                "proposed_start": {"row": 0, "col": 0},
                "proposed_end": {"row": 0, "col": 1},
                "active_buffer": True, "focused": True,
            }),
            ("decision", "prediction_accepted" if outcome == "accepted"
             else "prediction_dismissed", {
                 "prediction_id": prediction, "outcome": outcome,
                 "outcome_source": "editor_observation",
             }),
        ]
        if prediction != "neutral":
            rows.append(("review", "prediction_reviewed", {
                "prediction_id": prediction, "synthetic": False, "human_verified": True,
                "review_source": "explicit_editor_confirmation",
                "resolution_event_id": prediction + "-decision", "outcome": outcome,
            }))
        for suffix, kind, payload in rows:
            sequence += 1
            conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?)", (
                prediction + "-" + suffix, "s", sequence, kind, sequence, json.dumps(payload),
            ))
    conn.commit()
    conn.close()
    result = extract(db)
    assert len(result["candidate_preference_pairs"]) == 1
    assert result["candidate_preference_pairs"][0]["dispreferred_prediction_id"] == "explicit"
    neutral = next(row for row in result["evidence"] if row["prediction_id"] == "neutral")
    assert neutral["outcome"] == neutral_outcome
    assert result["readiness"]["enabled"] is False


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


def test_v1_actions_use_verified_blobs_and_keep_legacy_hash_semantics(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE blobs (sha256 TEXT PRIMARY KEY, original_bytes INTEGER, "
        "stored_bytes INTEGER, compression TEXT, content BLOB)"
    )
    actions = [
        ("replace", "replace_line", "λ = 2", {"row": 4, "col": 0}, {"row": 4, "col": 6},
         300, 306),
        ("delete", "delete_line", None, {"row": 5, "col": 0}, {"row": 6, "col": 0},
         400, 410),
        ("insert", "insert_before", "new_call()", {"row": 6, "col": 0}, {"row": 6, "col": 0},
         500, 500),
        ("noedit", "keep", None, None, None, None, None),
    ]
    sequence = 0
    for prediction_id, kind, text, start, end, start_byte, end_byte in actions:
        sequence += 1
        requested = {
            "prediction_id": prediction_id,
            "file": "repo:demo:src/example.py",
            "context_hash": "context-hash",
            "pre_state_hash": "state-hash",
            "synthetic": True,
        }
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (f"{prediction_id}-request", "s-v1", sequence, "prediction_requested", sequence,
             json.dumps(requested)),
        )
        action_bytes = json.dumps(
            {"kind": kind, "text": text}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        action_hash = hashlib.sha256(action_bytes).hexdigest()
        if kind == "keep":
            raw_bytes = b"N"
        elif kind == "delete_line":
            raw_bytes = b"D"
        else:
            raw_bytes = ("R\t" + (text or "")).encode("utf-8")
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        for blob_hash, content in ((action_hash, action_bytes), (raw_hash, raw_bytes)):
            stored = gzip.compress(content)
            conn.execute(
                "INSERT OR IGNORE INTO blobs VALUES (?,?,?,?,?)",
                (blob_hash, len(content), len(stored), "gzip", stored),
            )
        sequence += 1
        generated = {
            "prediction_id": prediction_id,
            "action_blob_hash": action_hash,
            "raw_response_hash": raw_hash,
            "canonical_action": kind,
            "wire_version": "single-line-edit-v1",
        }
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (f"{prediction_id}-generated", "s-v1", sequence, "prediction_generated", sequence,
             json.dumps(generated)),
        )
        if kind == "keep":
            sequence += 1
            conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (f"{prediction_id}-noedit", "s-v1", sequence, "heartbeat", sequence,
                 json.dumps({"prediction_id": prediction_id,
                             "prediction_lifecycle": "model_no_edit"})),
            )
            continue
        sequence += 1
        shown = {
            "prediction_id": prediction_id,
            "action": kind,
            "proposed_text": "" if kind == "delete_line" else text,
            "proposed_start": start,
            "proposed_end": end,
            "proposed_start_byte": start_byte,
            "proposed_end_byte": end_byte,
            "proposed_range_end_exclusive": True,
            "proposed_range_includes_terminator": kind == "delete_line",
            "wire_version": "single-line-edit-v1",
        }
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (f"{prediction_id}-shown", "s-v1", sequence, "prediction_shown", sequence,
             json.dumps(shown)),
        )
        sequence += 1
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (
                f"{prediction_id}-accepted",
                "s-v1",
                sequence,
                "prediction_accepted",
                sequence,
                json.dumps({
                    "prediction_id": prediction_id,
                    "wire_version": "single-line-edit-v1",
                    "action": kind,
                    "editable_range": {
                        "start_row": start["row"],
                        "start_col": start["col"],
                        "end_row": end["row"],
                        "end_col": end["col"],
                        "start_byte": start_byte,
                        "end_byte": end_byte,
                        "end_exclusive": True,
                        "includes_terminator": kind == "delete_line",
                    },
                }),
            ),
        )
    conn.commit()
    conn.close()

    result = extract(db)
    assert result["schema_version"] == "personalization-feedback-evidence-v6"
    evidence = {row["prediction_id"]: row for row in result["evidence"]}
    for prediction_id, kind, text, start, end, start_byte, end_byte in actions:
        row = evidence[prediction_id]
        assert row["proposal_protocol_version"] == "single-line-edit-v1"
        assert row["canonical_action_kind"] == kind
        assert row["proposal_action_source"] == "verified_action_blob"
        if kind == "keep":
            assert row["shown"] is False
            assert row["proposal_sha256"] is None
            assert row["proposal_identity_sha256"] is None
            assert row["proposed_byte_range"] is None
        else:
            assert row["shown"] is True
            assert row["proposal_sha256"] is not None
            assert row["proposal_identity_sha256"] is not None
            expected_compatibility_hash = hashlib.sha256(
                json.dumps(
                    {"action": "replace", "text": "" if kind == "delete_line" else text},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            assert row["proposal_sha256"] == expected_compatibility_hash
            assert row["proposed_byte_range"] == {
                "start": {"row": start["row"], "byte_col": start["col"]},
                "end": {"row": end["row"], "byte_col": end["col"]},
                "start_byte": start_byte,
                "end_byte": end_byte,
                "byte_interval_end_exclusive": True,
                "includes_terminator": kind == "delete_line",
                "row_base": 0,
                "column_unit": "utf8_byte",
            }
            assert row["proposal_sha256_semantics"] == (
                "legacy_replacement_equivalence_comparison_only"
            )
            assert row["accepted_action_byte_range"] == row["proposed_byte_range"]
            assert "accepted_action_range_mismatch" not in row["ambiguity_flags"]
        assert "proposed_text" not in row
    encoded_result = json.dumps(result, ensure_ascii=False)
    assert "new_call()" not in encoded_result
    assert "λ = 2" not in encoded_result


def test_v1_blob_mismatch_does_not_fall_back_to_display_text(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
        "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE blobs (sha256 TEXT PRIMARY KEY, original_bytes INTEGER, "
        "stored_bytes INTEGER, compression TEXT, content BLOB)"
    )
    action_bytes = b'{"kind":"delete_line","text":null}'
    action_hash = hashlib.sha256(action_bytes).hexdigest()
    raw = b"D"
    raw_hash = hashlib.sha256(raw).hexdigest()
    conn.executemany(
        "INSERT INTO blobs VALUES (?,?,?,?,?)",
        [
            (action_hash, len(action_bytes), len(action_bytes), "raw", action_bytes),
            (raw_hash, len(raw), len(raw), "raw", raw),
        ],
    )
    events = [
        ("r", "prediction_requested", {"prediction_id": "p"}),
        ("g", "prediction_generated", {
            "prediction_id": "p", "wire_version": "single-line-edit-v1",
            "canonical_action": "delete_line", "action_blob_hash": action_hash,
            "raw_response_hash": raw_hash,
        }),
        ("s", "prediction_shown", {
            "prediction_id": "p", "wire_version": "single-line-edit-v1",
            "action": "replace_line", "proposed_text": "forged display text",
            "proposed_start": {"row": 0, "col": 0},
            "proposed_end": {"row": 0, "col": 4},
        }),
        ("a", "prediction_accepted", {
            "prediction_id": "p", "wire_version": "single-line-edit-v1",
            "editable_range": {
                "start_row": 0, "start_col": 0, "end_row": 0, "end_col": 4,
                "start_byte": 0, "end_byte": 4, "end_exclusive": True,
                "includes_terminator": False,
            },
        }),
    ]
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?)",
        [
            (event_id, "s", index, event_type, index, json.dumps(payload))
            for index, (event_id, event_type, payload) in enumerate(events, start=1)
        ],
    )
    conn.commit()
    conn.close()

    row = extract(db)["evidence"][0]
    assert row["proposal_sha256"] is None
    assert row["proposal_identity_sha256"] is None
    assert "action_blob_display_mismatch" in row["ambiguity_flags"]
    assert "accepted_action_range_mismatch" in row["ambiguity_flags"]


def test_automatic_display_suppression_is_not_a_human_rejection(tmp_path: Path) -> None:
    path = tmp_path / 'collector.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE events (event_id TEXT,session_id TEXT,sequence_number INTEGER,'
                   'event_type TEXT,timestamp_ms INTEGER,payload_json TEXT)')
        db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?)', [
            ('request', 's', 1, 'prediction_requested', 100,
             json.dumps({'prediction_id': 'p', 'synthetic': True})),
            ('suppressed', 's', 2, 'heartbeat', 110,
             json.dumps({'prediction_id': 'p',
                         'prediction_lifecycle': 'automatic_policy_suppressed'})),
        ])
    result = extract(path)
    assert result['evidence'][0]['outcome'] == 'automatic_policy_suppressed'
    assert result['evidence'][0]['explicit_outcome'] is None
    assert result['candidate_preference_pairs'] == []
    assert result['readiness']['enabled'] is False


def test_rust_feedback_retains_actual_model_and_context_identity(tmp_path: Path) -> None:
    db = tmp_path / "collector.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE events (event_id TEXT, session_id TEXT, sequence_number INTEGER, "
            "event_type TEXT, timestamp_ms INTEGER, payload_json TEXT)"
        )
        payload = {
            "prediction_id": "rust-prediction",
            "synthetic": True,
            "model_alias": "q25",
            "model_protocol": "single-line-edit-v1",
            "model_gguf_sha256": "a" * 64,
            "context_layout": "cursor-last-v1",
            "context_policy_version": "single-line-cursor-last-context-v1",
            "runtime_config_hash": "b" * 64,
        }
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            ("request", "synthetic-session", 1, "prediction_requested", 1, json.dumps(payload)),
        )
    result = extract(db)
    evidence = result["evidence"][0]
    for key in (
        "model_alias",
        "model_protocol",
        "model_gguf_sha256",
        "context_layout",
        "context_policy_version",
        "runtime_config_hash",
    ):
        assert evidence[key] == payload[key]
    assert result["readiness"]["enabled"] is False
    assert result["candidate_preference_pairs"] == []
