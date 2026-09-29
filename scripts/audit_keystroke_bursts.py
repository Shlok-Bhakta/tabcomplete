"""Count paused, cursor-aligned one-line targets in a private keystroke log.

Only aggregate counts leave the process. This does not create training labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_keystroke_qualification import position, replay_edit, sha256

EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
URL = re.compile(r"https?://[^\s'\"]+", re.IGNORECASE)


def length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length == 1:
        return "1"
    for floor, ceiling in ((2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 96)):
        if length <= ceiling:
            return f"{floor}-{ceiling}"
    return "over_96"


@dataclass
class State:
    text: str = ""
    valid: bool = True
    last_time: int | None = None
    last_event_id: int | None = None
    cursor: int | None = None
    burst_before: str | None = None
    burst_cursor: int | None = None
    burst_first_offset: int | None = None
    burst_edits: int = 0
    suppress_next_burst: bool = False


def changed_spans(before: str, after: str, *, known_prefix: int) -> tuple[str, str] | None:
    if before == after:
        return None
    if before[:known_prefix] != after[:known_prefix]:
        return None
    prefix = known_prefix
    while prefix < min(len(before), len(after)) and before[prefix] == after[prefix]:
        prefix += 1
    # Compare suffix slices in C instead of looping over thousands of source
    # characters in Python for each of millions of bursts.
    low, high = 0, min(len(before) - prefix, len(after) - prefix)
    while low < high:
        middle = (low + high + 1) // 2
        if before[-middle:] == after[-middle:]:
            low = middle
        else:
            high = middle - 1
    suffix = low
    old_end = len(before) - suffix if suffix else len(before)
    new_end = len(after) - suffix if suffix else len(after)
    return before[prefix:old_end], after[prefix:new_end]


def serving_replacement(before: str, after: str, cursor: int) -> str | None:
    """Return the exact replacement for the installed cursor-line suffix region."""
    if cursor < 0 or cursor > len(before) or before[:cursor] != after[:cursor]:
        return None
    before_line_end = before.find("\n", cursor)
    after_line_end = after.find("\n", cursor)
    if before_line_end < 0:
        before_line_end = len(before)
    if after_line_end < 0:
        after_line_end = len(after)
    replacement = after[cursor:after_line_end]
    if (
        "\r" in before[cursor:before_line_end]
        or "\r" in replacement
        or before[before_line_end:] != after[after_line_end:]
        or before[:cursor] + replacement + before[before_line_end:] != after
    ):
        return None
    return replacement


def audit(csv_path: Path, plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "tabcomplete-keystroke-qualification-plan-v1" or plan.get(
        "plan_revision"
    ) not in (2, 3, 4):
        raise ValueError("burst qualification requires a frozen plan revision")
    if (
        csv_path.stat().st_size != plan["source"]["bytes"]
        or sha256(csv_path) != plan["source"]["sha256"]
    ):
        raise ValueError("keystroke source identity mismatch")
    csv.field_size_limit(16 * 1024 * 1024)
    states: dict[tuple[str, str, str], State] = {}
    assignment_states: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    assignment_candidates: Counter[str] = Counter()
    candidate_subjects: set[str] = set()
    candidate_length_buckets: Counter[str] = Counter()
    candidate_source_size_buckets: Counter[str] = Counter()
    candidate_length_four_subjects: set[str] = set()
    serving_length_buckets: Counter[str] = Counter()
    serving_subjects: set[str] = set()

    def finish(group: tuple[str, str, str], state: State) -> None:
        before = state.burst_before
        if before is None:
            return
        counts["bursts"] += 1
        if state.burst_cursor is None:
            counts["unknown_input_cursor"] += 1
            return
        counts["known_input_cursor"] += 1
        if state.burst_first_offset != state.burst_cursor:
            counts["first_edit_not_at_input_cursor"] += 1
            return
        counts["first_edit_at_input_cursor"] += 1
        if before == state.text:
            counts["net_no_change_bursts"] += 1
            return
        spans = changed_spans(before, state.text, known_prefix=state.burst_cursor)
        if spans is None:
            counts["prior_input_prefix_changed"] += 1
            return
        old, new = spans
        if "\n" in old or "\n" in new:
            counts["multiline_diff"] += 1
            return
        counts["one_line_diff"] += 1
        if len(new.encode("utf-8")) > 96:
            counts["response_over_96_utf8_bytes"] += 1
            return
        counts["bounded_one_line_targets"] += 1
        if (
            EMAIL.search(before)
            or EMAIL.search(state.text)
            or URL.search(before)
            or URL.search(state.text)
        ):
            counts["candidate_state_with_email_or_url_pattern"] += 1
            return
        counts["pattern_clean_candidates"] += 1
        output_length = len(new.encode("utf-8"))
        candidate_length_buckets[length_bucket(output_length)] += 1
        source_length = len(before.encode("utf-8"))
        candidate_source_size_buckets["at_most_4096" if source_length <= 4096 else "over_4096"] += 1
        if output_length >= 4:
            counts["pattern_clean_length_at_least_four"] += 1
            candidate_length_four_subjects.add(group[0])
        kind = "replace" if old and new else "delete" if old else "insert"
        action_counts[kind] += 1
        candidate_subjects.add(group[0])
        assignment_candidates[group[1]] += 1
        if plan["plan_revision"] >= 4:
            replacement = serving_replacement(before, state.text, state.burst_cursor)
            if replacement is None:
                counts["serving_region_mismatch"] += 1
                return
            counts["serving_region_exact"] += 1
            serving_length_buckets[length_bucket(len(replacement.encode("utf-8")))] += 1
            serving_subjects.add(group[0])

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts["rows"] += 1
            if row["EventType"] != "File.Edit":
                if plan["plan_revision"] >= 4 and row["EventType"] in (
                    "Run.Program",
                    "Submit",
                    "X-SwitchTask",
                ):
                    session = (row["SubjectID"], row["AssignmentID"])
                    for group in assignment_states.get(session, ()):
                        state = states[group]
                        if state.valid and state.burst_before is not None:
                            finish(group, state)
                            state.burst_before = None
                            state.suppress_next_burst = True
                            counts["nonedit_burst_boundaries"] += 1
                continue
            counts["file_edits"] += 1
            group = (row["SubjectID"], row["AssignmentID"], row["CodeStateSection"])
            if group not in states:
                states[group] = State()
                assignment_states.setdefault(group[:2], set()).add(group)
            state = states[group]
            if not state.valid:
                counts["events_after_invalid_group"] += 1
                continue
            offset = position(row["SourceLocation"])
            try:
                timestamp = int(row["ClientTimestamp"])
                event_id = int(row["EventID"])
            except ValueError:
                timestamp = event_id = -1
            if offset is None or timestamp < 0 or event_id < 0:
                counts["invalid_coordinates"] += 1
                state.valid = False
                continue
            if state.last_time is not None and (timestamp, event_id) < (
                state.last_time,
                state.last_event_id or 0,
            ):
                counts["out_of_order"] += 1
                state.valid = False
                continue
            if (
                state.burst_before is None
                or state.last_time is None
                or timestamp - state.last_time >= 250
            ):
                finish(group, state)
                state.burst_before = state.text
                state.burst_cursor = None if state.suppress_next_burst else state.cursor
                state.burst_first_offset = offset
                state.burst_edits = 0
                state.suppress_next_burst = False
            new_text = replay_edit(state.text, offset, row["DeleteText"], row["InsertText"])
            if new_text is None or len(new_text) > 1_000_000:
                counts["replay_failure"] += 1
                state.valid = False
                continue
            state.text = new_text
            state.cursor = offset + len(row["InsertText"])
            state.last_time = timestamp
            state.last_event_id = event_id
            state.burst_edits += 1
            counts["replayed_edits"] += 1
    for group, state in states.items():
        if state.valid:
            finish(group, state)
    return {
        "schema": "tabcomplete-keystroke-burst-audit-v1",
        "plan_sha256": sha256(plan_path),
        "source_sha256": plan["source"]["sha256"],
        "counts": dict(sorted(counts.items())),
        "pattern_clean_candidate_actions": dict(sorted(action_counts.items())),
        "pattern_clean_candidate_subjects": len(candidate_subjects),
        "pattern_clean_length_at_least_four_subjects": len(candidate_length_four_subjects),
        "pattern_clean_candidate_length_buckets": dict(sorted(candidate_length_buckets.items())),
        "pattern_clean_candidate_source_size_buckets": dict(
            sorted(candidate_source_size_buckets.items())
        ),
        "pattern_clean_candidate_assignments": dict(sorted(assignment_candidates.items())),
        "serving_region_replacement_length_buckets": dict(sorted(serving_length_buckets.items())),
        "serving_region_candidate_subjects": len(serving_subjects),
        "limitations": [
            "Inferred cursor is the end of the prior edit; unlogged movement may still exist.",
            "A natural next burst is an observed action, not proof that a model could infer it.",
            "Email/URL pattern screening does not clear privacy or source rights.",
            "No idle pause is labeled no-edit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("burst audit output already exists")
    result = audit(args.csv, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({k: result["counts"].get(k, 0) for k in ("bursts", "pattern_clean_candidates")})
    )


if __name__ == "__main__":
    main()
