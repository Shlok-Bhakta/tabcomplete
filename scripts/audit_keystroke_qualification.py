"""Private, aggregate-only replay audit of the pinned 2019 CS1 edit log.

This script never writes source text or participant identifiers. Its output is
an eligibility diagnostic, not a training set or a privacy clearance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_FIELDS = {
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
}
SAFE_EDIT_TYPES = {"Insert", "Delete", "Undo", "Paste", "Cut", "Redo", "Drag", "X-Checkpoint"}
SAFE_EVENT_TYPES = {"File.Edit", "Run.Program", "X-SwitchTask", "Submit"}
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
URL = re.compile(r"https?://[^\s'\"]+", re.IGNORECASE)


@dataclass
class BufferState:
    text: str = ""
    valid: bool = True
    edits: int = 0
    last_time: int | None = None
    last_event_id: int | None = None
    privacy_flagged: bool = False
    privacy_watch: int = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def position(value: str) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer() or numeric < 0 or numeric > 1_000_000:
        return None
    return int(numeric)


def replay_edit(source: str, offset: int, removed: str, inserted: str) -> str | None:
    """Apply one event only if its removed text matches the current state."""
    if offset > len(source) or source[offset : offset + len(removed)] != removed:
        return None
    return source[:offset] + inserted + source[offset + len(removed) :]


def audit(csv_path: Path, plan_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "tabcomplete-keystroke-qualification-plan-v1":
        raise ValueError("qualification plan schema mismatch")
    if (
        csv_path.stat().st_size != plan["source"]["bytes"]
        or sha256(csv_path) != plan["source"]["sha256"]
    ):
        raise ValueError("keystroke source identity mismatch")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")

    # Run.Program output can be much larger than the Python CSV default. It is
    # never copied into reconstructed source or aggregate output.
    csv.field_size_limit(16 * 1024 * 1024)
    states: dict[tuple[str, str, str], BufferState] = {}
    subjects: set[str] = set()
    assignments: Counter[str] = Counter()
    events: Counter[str] = Counter()
    edit_types: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    valid_types: Counter[str] = Counter()
    pause_buckets: Counter[str] = Counter()
    row_count = 0
    file_edits = 0
    valid_edits = 0
    one_line_events = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not EXPECTED_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("keystroke CSV schema mismatch")
        for row in reader:
            row_count += 1
            if max_rows is not None and row_count > max_rows:
                row_count -= 1
                break
            subject = row["SubjectID"]
            assignment = row["AssignmentID"]
            section = row["CodeStateSection"]
            subjects.add(subject)
            assignments[assignment] += 1
            event_type = row["EventType"]
            edit_type = row["EditType"]
            events[event_type if event_type in SAFE_EVENT_TYPES else "other"] += 1
            if event_type != "File.Edit":
                continue
            file_edits += 1
            edit_types[edit_type if edit_type in SAFE_EDIT_TYPES else "other"] += 1
            state = states.setdefault((subject, assignment, section), BufferState())
            if not state.valid:
                reasons["after_unreconstructable_state"] += 1
                continue
            offset = position(row["SourceLocation"])
            if offset is None:
                reasons["invalid_position"] += 1
                state.valid = False
                continue
            try:
                timestamp = int(row["ClientTimestamp"])
                event_id = int(row["EventID"])
            except ValueError:
                reasons["invalid_time_or_event_id"] += 1
                state.valid = False
                continue
            if state.last_time is not None:
                if (timestamp, event_id) < (state.last_time, state.last_event_id or 0):
                    reasons["out_of_order"] += 1
                    state.valid = False
                    continue
                pause = timestamp - state.last_time
                if pause >= 250:
                    pause_buckets["at_least_250ms"] += 1
                if pause >= 1000:
                    pause_buckets["at_least_1000ms"] += 1
            removed = row["DeleteText"]
            inserted = row["InsertText"]
            new_text = replay_edit(state.text, offset, removed, inserted)
            if new_text is None:
                reasons["delete_text_mismatch_or_missing_initial_state"] += 1
                state.valid = False
                continue
            if len(new_text) > 1_000_000:
                reasons["buffer_limit"] += 1
                state.valid = False
                continue
            valid_edits += 1
            valid_types[edit_type if edit_type in SAFE_EDIT_TYPES else "other"] += 1
            if "\n" not in removed and "\n" not in inserted:
                one_line_events += 1
            state.text = new_text
            state.edits += 1
            state.last_time = timestamp
            state.last_event_id = event_id
            if "@" in inserted or "http" in inserted.lower():
                state.privacy_watch = 100
            if state.privacy_watch or state.edits % 100 == 0:
                state.privacy_flagged |= bool(EMAIL.search(new_text) or URL.search(new_text))
                state.privacy_watch = max(0, state.privacy_watch - 1)

    for state in states.values():
        if state.valid:
            state.privacy_flagged |= bool(EMAIL.search(state.text) or URL.search(state.text))
    return {
        "schema": "tabcomplete-keystroke-qualification-result-v1",
        "plan_sha256": sha256(plan_path),
        "source_sha256": plan["source"]["sha256"],
        "sampled_prefix_rows": max_rows,
        "rows_read": row_count,
        "subjects": len(subjects),
        "subject_assignment_section_groups": len(states),
        "reconstructable_groups_at_end": sum(state.valid for state in states.values()),
        "groups_with_email_or_url_pattern": sum(state.privacy_flagged for state in states.values()),
        "assignment_counts": dict(sorted(assignments.items())),
        "event_type_counts": dict(sorted(events.items())),
        "file_edit_count": file_edits,
        "valid_replayed_edits": valid_edits,
        "valid_one_line_events": one_line_events,
        "edit_type_counts": dict(sorted(edit_types.items())),
        "valid_edit_type_counts": dict(sorted(valid_types.items())),
        "pause_counts_between_valid_edits": dict(sorted(pause_buckets.items())),
        "replay_failure_counts": dict(sorted(reasons.items())),
        "limitations": [
            "A replay match proves event consistency, not next-edit inferability.",
            "Privacy pattern counts are screening signals, not clearance.",
            "Individual keystrokes are not ready-made one-line completion targets.",
            "No no-edit labels are derived from idle periods.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("qualification output already exists")
    result = audit(args.csv, args.plan, max_rows=args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows_read": result["rows_read"],
                "valid_replayed_edits": result["valid_replayed_edits"],
                "reconstructable_groups_at_end": result["reconstructable_groups_at_end"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
