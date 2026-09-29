"""Boundary checks for the private CS1 sequence qualification rules."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_keystroke_bursts import (  # noqa: E402
    audit,
    changed_spans,
    length_bucket,
    serving_replacement,
)
from audit_keystroke_qualification import position, replay_edit  # noqa: E402


def test_replay_requires_exact_deleted_text_and_handles_unicode() -> None:
    source = "print('é')\n"
    offset = source.index("é")
    assert replay_edit(source, offset, "é", "🙂") == "print('🙂')\n"
    assert replay_edit(source, offset, "e", "🙂") is None
    assert replay_edit(source, len(source) + 1, "", "x") is None


def test_source_location_must_be_a_nonnegative_integer_index() -> None:
    assert position("7.0") == 7
    assert position("7.5") is None
    assert position("nan") is None
    assert position("-1") is None


def test_burst_diff_preserves_cursor_guard_and_deletion() -> None:
    assert changed_spans("abcXYZend", "abc123end", known_prefix=3) == ("XYZ", "123")
    assert changed_spans("abcXYZend", "abc123end", known_prefix=4) is None
    assert changed_spans("abcX", "abc", known_prefix=3) == ("X", "")
    assert changed_spans("abc", "abc", known_prefix=3) is None
    assert length_bucket(0) == "0"
    assert length_bucket(4) == "4-7"
    assert length_bucket(96) == "64-96"


def test_serving_region_replays_exact_suffix_without_touching_other_lines() -> None:
    before = "value = pri\nother = 1\n"
    after = "value = print(x)\nother = 1\n"
    cursor = len("value = pr")
    assert serving_replacement(before, after, cursor) == "int(x)"
    assert serving_replacement(before, "value = print(x)\nother = 2\n", cursor) is None
    assert serving_replacement(before, "value = print(x)\nother = 1", cursor) is None
    assert serving_replacement(before, "VALUE = print(x)\nother = 1\n", cursor) is None


def test_run_event_breaks_burst_and_suppresses_the_next_target(tmp_path: Path) -> None:
    source = tmp_path / "events.csv"
    fields = [
        "EventID",
        "SubjectID",
        "AssignmentID",
        "CodeStateSection",
        "EventType",
        "InsertText",
        "DeleteText",
        "SourceLocation",
        "ClientTimestamp",
        "EditType",
    ]
    events = [
        ("File.Edit", "a", "0.0", 0),
        ("File.Edit", "b", "1.0", 300),
        ("Run.Program", "", "", 350),
        ("File.Edit", "c", "2.0", 400),
        ("File.Edit", "d", "3.0", 700),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event_id, (event_type, inserted, position_text, timestamp) in enumerate(events):
            writer.writerow(
                {
                    "EventID": event_id,
                    "SubjectID": "example",
                    "AssignmentID": "p4s",
                    "CodeStateSection": "task0.py",
                    "EventType": event_type,
                    "InsertText": inserted,
                    "DeleteText": "",
                    "SourceLocation": position_text,
                    "ClientTimestamp": timestamp,
                    "EditType": "Insert" if event_type == "File.Edit" else "",
                }
            )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "tabcomplete-keystroke-qualification-plan-v1",
                "plan_revision": 4,
                "source": {
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
            }
        )
    )
    result = audit(source, plan)
    assert result["counts"]["nonedit_burst_boundaries"] == 1
    assert result["counts"]["serving_region_exact"] == 2
    assert result["counts"]["unknown_input_cursor"] == 2
