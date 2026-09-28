"""Independent, fixed 32-case development calibration suite.

This module does not alter the shared evaluator or the earlier 16-case suite.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from tinycomplete.one_line.context import serialize_state_bounded
from tinycomplete.one_line.contract import EditAction, EditState, physical_lines
from tinycomplete.one_line.evaluate import EvaluationCase, Prediction, evaluate_cases, summarize

SUITE = "independent-synthetic-calibration-fresh32-v1"
SUITE_REVISION = 3


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lines(source: str) -> tuple[str, ...]:
    return tuple(line.content.decode("utf-8") for line in physical_lines(source.encode("utf-8")))


def _objective(state: EditState, spec: dict) -> Callable[[str], bool]:
    from tree_sitter_language_pack import get_parser

    parser = get_parser(state.filetype)
    before = _lines(state.source)
    row = state.target_row
    if row >= len(before):
        raise ValueError("fresh32 objective requires an existing target row")
    kind = spec["kind"]

    if kind == "history_transfer":
        if len(state.history) != 1 or state.history[0].row == row:
            raise ValueError("transfer must be grounded in a sibling's prior edit")
        history = state.history[0]
        old = spec["history_old_fragment"]
        new = spec["history_new_fragment"]
        target_old = spec["target_old_fragment"]
        target_new = spec["target_new_fragment"]
        if (
            old == new
            or old not in history.old_text
            or new not in history.new_text
            or target_old not in before[row]
            or target_new in before[row]
        ):
            raise ValueError("transfer objective lacks visible source/history evidence")

        def property_check(source: str) -> bool:
            lines = _lines(source)
            return row < len(lines) and target_new in lines[row] and target_old not in lines[row]

    elif kind == "latest_reversal":
        if (
            len(state.history) != 1
            or state.history[0].row != row
            or state.history[0].new_text != before[row]
            or state.history[0].old_text == before[row]
        ):
            raise ValueError("keep case lacks a visible latest target reversal")

        def property_check(source: str) -> bool:
            lines = _lines(source)
            return row < len(lines) and lines[row] == before[row]

    elif kind == "ordered_insert":
        intent = spec["visible_intent"]
        if intent not in state.source:
            raise ValueError("insertion intent is absent from visible source")
        pattern = re.compile(spec["inserted_line_regex"])
        if any(pattern.search(line) for line in before):
            raise ValueError("insertion operation already exists")

        def property_check(source: str) -> bool:
            lines = _lines(source)
            return (
                len(lines) == len(before) + 1
                and row + 1 < len(lines)
                and lines[row + 1] == before[row]
                and bool(pattern.search(lines[row]))
            )

    elif kind == "remove_stale":
        intent = spec["visible_intent"]
        if intent not in state.source or before.count(before[row]) != 1:
            raise ValueError("deletion lacks visible intent or a unique target")

        def property_check(source: str) -> bool:
            lines = _lines(source)
            return len(lines) == len(before) - 1 and before[row] not in lines

    else:
        raise ValueError("unknown fresh32 objective")

    def checked(source: str) -> bool:
        if parser.parse(source.encode("utf-8")).root_node.has_error:
            return False
        return property_check(source)

    return checked


def load_fresh32_cases(fixture_path: Path, manifest_path: Path) -> list[EvaluationCase]:
    """Reject changes to the fixture, individual cases, or development split."""
    payload = fixture_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != manifest.get("fixture_sha256"):
        raise ValueError("fresh32 fixture hash mismatch")
    fixture = json.loads(payload)
    rows = fixture.get("cases")
    if fixture.get("suite") != SUITE or fixture.get("suite_revision") != SUITE_REVISION:
        raise ValueError("unexpected fresh32 suite or contract revision")
    if not isinstance(rows, list) or len(rows) != 32 or manifest.get("case_count") != 32:
        raise ValueError("fresh32 suite must contain exactly 32 cases")
    if {row["id"]: canonical_hash(row) for row in rows} != manifest.get("case_hashes"):
        raise ValueError("fresh32 per-case hash mismatch")
    cases = []
    for row in rows:
        if row["split"] != "development" or row["source_type"] != "synthetic":
            raise ValueError("fresh32 calibration is synthetic development only")
        state = EditState.from_mapping(row["state"])
        current = _lines(state.source)
        for edit in state.history:
            if edit.row >= len(current) or current[edit.row] != edit.new_text:
                raise ValueError("fresh32 history does not reconstruct current source")
            if edit.old_text == edit.new_text:
                raise ValueError("fresh32 history contains a no-op")
        action = EditAction(**row["gold_action"])
        objective = _objective(state, row["objective"])
        cases.append(
            EvaluationCase(
                id=row["id"],
                state=state,
                gold_action=action,
                after_source=row["after_source"],
                split="development",
                source_repo=row["source_repo"],
                generator_family=row["generator_family"],
                mechanism=row["mechanism"],
                source_type="synthetic",
                objective_check=objective,
                objective_name=row["objective"]["kind"],
            )
        )
    if len({case.id for case in cases}) != 32:
        raise ValueError("fresh32 has duplicate case IDs")
    return cases


def fresh32_prompts(cases: Sequence[EvaluationCase], tokenizer: object) -> list[dict]:
    """Expose only the canonical answer-free state prompt and case ID."""
    if len(cases) != 32 or any(case.split != "development" for case in cases):
        raise ValueError("expected the fixed 32-case development suite")
    output = []
    for case in cases:
        context = serialize_state_bounded(case.state, tokenizer, max_input_tokens=1024)
        output.append(
            {"case_id": case.id, "prompt": context.text, "input_tokens": context.input_tokens}
        )
    return output


def score_fresh32_predictions(
    cases: Sequence[EvaluationCase], predictions: Sequence[Prediction]
) -> dict:
    """Score one complete pass with the shared v1 action evaluator."""
    if len(cases) != 32 or any(case.split != "development" for case in cases):
        raise ValueError("expected the fixed 32-case development suite")
    scores = evaluate_cases(cases, predictions, max_tokens=64)
    return {"suite": SUITE, "summary": summarize(scores), "cases": [row.__dict__ for row in scores]}
