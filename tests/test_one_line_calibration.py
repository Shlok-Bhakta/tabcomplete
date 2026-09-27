from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditAction, apply_action, physical_lines
from tinycomplete.one_line.evaluate import (
    Prediction,
    calibration_prompts,
    control_predictions,
    load_calibration_cases,
    score_calibration_predictions,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1] / "reports/research/one_line_r1"
FIXTURE = ROOT / "calibration_cases.json"
MANIFEST = ROOT / "calibration_manifest.json"


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8"))


def test_frozen_cases_gold_and_controls() -> None:
    cases = load_calibration_cases(FIXTURE, MANIFEST)
    assert len(cases) == 16
    assert {case.split for case in cases} == {"development"}
    assert {case.source_type for case in cases} == {"synthetic"}
    assert {case.state.filetype for case in cases} == {"python", "typescript", "rust", "go"}
    assert {
        kind: sum(case.gold_action.kind == kind for case in cases)
        for kind in ("replace_line", "insert_before", "delete_line", "keep")
    } == {"replace_line": 4, "insert_before": 4, "delete_line": 4, "keep": 4}

    gold_rows = score_calibration_predictions(cases, control_predictions(cases, "gold"))
    summary = gold_rows["summary"]
    assert summary["valid_action"] == 16
    assert summary["edit_success"] == 12
    assert summary["keep_correct"] == 4
    assert summary["false_positive_edits"] == 0

    keep_rows = score_calibration_predictions(cases, control_predictions(cases, "keep"))
    assert keep_rows["summary"]["edit_success"] == 0
    assert keep_rows["summary"]["keep_correct"] == 4


def test_objectives_reject_unchanged_and_wrong_edits() -> None:
    cases = load_calibration_cases(FIXTURE, MANIFEST)
    for case in cases:
        check = case.objective_check
        assert check is not None
        assert check(case.after_source), case.id
        if case.gold_action.kind != "keep":
            assert not check(case.state.source), case.id
        assert not check(apply_action(case.state, EditAction("replace_line", "BROKEN"))), case.id


def test_counterfactual_pairs_change_only_visible_history() -> None:
    by_id = {case.id: case for case in load_calibration_cases(FIXTURE, MANIFEST)}
    for edit_id, keep_id in (("c01", "c02"), ("c05", "c06"), ("c09", "c10"), ("c13", "c14")):
        edit, keep = by_id[edit_id], by_id[keep_id]
        assert replace(edit.state, history=keep.state.history) == keep.state
        assert edit.gold_action.kind == "replace_line"
        assert keep.gold_action.kind == "keep"
        assert keep.state.history[-1].row == keep.state.target_row
        assert keep.state.history[-1].new_text == physical_lines(keep.state.source.encode("utf-8"))[
            keep.state.target_row
        ].content.decode("utf-8")
        assert keep.state.history[-1].old_text != keep.state.history[-1].new_text


def test_prompt_contains_visible_history_without_scoring_metadata() -> None:
    cases = load_calibration_cases(FIXTURE, MANIFEST)
    prompts = calibration_prompts(cases, CharacterTokenizer())
    assert len(prompts) == 16
    for case, row in zip(cases, prompts, strict=True):
        text = row["prompt"]
        assert row["case_id"] == case.id
        assert row["input_tokens"] <= 1024
        assert "old=" in text if case.state.history else "Recent edits" in text
        for forbidden in (
            "after_source",
            "gold_action",
            "objective",
            "generator_family",
            "mechanism",
        ):
            assert forbidden not in text


def test_manifest_and_split_tampering_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.json"
    fixture.write_bytes(FIXTURE.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(MANIFEST.read_bytes())
    assert len(load_calibration_cases(fixture, manifest)) == 16
    data = json.loads(fixture.read_text())
    data["cases"][0]["state"]["source"] += "# tampered\n"
    fixture.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_calibration_cases(fixture, manifest)
    frozen = json.loads(manifest.read_text())
    frozen["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="case hashes"):
        load_calibration_cases(fixture, manifest)


def test_scorer_requires_complete_predictions() -> None:
    cases = load_calibration_cases(FIXTURE, MANIFEST)
    with pytest.raises(ValueError, match="cover every case"):
        score_calibration_predictions(cases, [Prediction(cases[0].id, "N", True)])
    gold = control_predictions(cases, "gold")
    rows = score_calibration_predictions(cases, gold)
    assert summarize([])["cases"] == 0
    assert rows["summary"]["cases"] == 16
