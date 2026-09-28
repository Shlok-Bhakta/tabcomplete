from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

from tinycomplete.one_line.calibration_fresh32 import (
    fresh32_prompts,
    load_fresh32_cases,
    score_fresh32_predictions,
)
from tinycomplete.one_line.contract import EditAction, apply_action, encode_action
from tinycomplete.one_line.evaluate import Prediction, control_predictions, score_case

ROOT = Path(__file__).resolve().parents[1] / "reports/research/one_line_r1"
FIXTURE = ROOT / "calibration_fresh32_cases.json"
MANIFEST = ROOT / "calibration_fresh32_manifest.json"


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode("utf-8"))


def test_balance_provenance_and_distinct_family() -> None:
    cases = load_fresh32_cases(FIXTURE, MANIFEST)
    existing = json.loads((ROOT / "calibration_cases.json").read_text())
    assert len(cases) == 32
    assert Counter(case.state.filetype for case in cases) == {
        "python": 8,
        "typescript": 8,
        "rust": 8,
        "go": 8,
    }
    assert Counter(case.gold_action.kind for case in cases) == {
        "keep": 8,
        "replace_line": 8,
        "insert_before": 8,
        "delete_line": 8,
    }
    assert {case.generator_family for case in cases}.isdisjoint(
        {row["generator_family"] for row in existing["cases"]}
    )
    assert {case.state.source for case in cases}.isdisjoint(
        {row["state"]["source"] for row in existing["cases"]}
    )
    assert {case.split for case in cases} == {"development"}
    assert {case.source_type for case in cases} == {"synthetic"}


def test_gold_keep_and_wrong_controls() -> None:
    cases = load_fresh32_cases(FIXTURE, MANIFEST)
    gold = score_fresh32_predictions(cases, control_predictions(cases, "gold"))["summary"]
    assert gold["valid_action"] == 32
    assert gold["edit_success"] == 24
    assert gold["keep_correct"] == 8
    keep = score_fresh32_predictions(cases, control_predictions(cases, "keep"))["summary"]
    assert keep["edit_success"] == 0
    assert keep["keep_correct"] == 8
    for case in cases:
        check = case.objective_check
        assert check is not None
        assert check(case.after_source)
        if case.gold_action.kind != "keep":
            assert not check(case.state.source), case.id
        wrong = apply_action(case.state, EditAction("replace_line", "BROKEN"))
        assert not check(wrong), case.id
        parser = get_parser(case.state.filetype)
        assert not parser.parse(case.state.source.encode()).root_node.has_error, case.id
        assert not parser.parse(case.after_source.encode()).root_node.has_error, case.id


def test_counterfactual_pairs_differ_only_in_real_history() -> None:
    by_id = {case.id: case for case in load_fresh32_cases(FIXTURE, MANIFEST)}
    for number in (1, 3, 9, 11, 17, 19, 25, 27):
        edit, keep = by_id[f"f{number:02}"], by_id[f"f{number + 1:02}"]
        assert replace(edit.state, history=keep.state.history) == keep.state
        assert edit.gold_action.kind == "replace_line"
        assert keep.gold_action.kind == "keep"
        assert edit.state.history[-1].row != edit.state.target_row
        assert keep.state.history[-1].row == keep.state.target_row
        assert keep.state.history[-1].old_text != keep.state.history[-1].new_text


def test_answer_free_prompts_and_alternative_validity() -> None:
    cases = load_fresh32_cases(FIXTURE, MANIFEST)
    prompts = fresh32_prompts(cases, CharacterTokenizer())
    assert len(prompts) == 32
    for case, item in zip(cases, prompts, strict=True):
        assert item["case_id"] == case.id
        assert item["input_tokens"] <= 1024
        assert "Target row (zero based)" in item["prompt"]
        assert case.generator_family not in item["prompt"]
        for forbidden in ("after_source", "gold_action", "objective", "mechanism"):
            assert forbidden not in item["prompt"]
    first = cases[0]
    alternative = EditAction("replace_line", "def net(x): return (round(x))")
    row = score_case(first, Prediction(first.id, encode_action(alternative), True, 16))
    assert row.exact_after is False
    assert row.alternative_valid is True
    assert row.edit_success == "pass"


def test_hash_guard_and_complete_pass(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    fixture.write_bytes(FIXTURE.read_bytes())
    manifest.write_bytes(MANIFEST.read_bytes())
    assert len(load_fresh32_cases(fixture, manifest)) == 32
    with pytest.raises(ValueError, match="cover every case"):
        score_fresh32_predictions(
            load_fresh32_cases(fixture, manifest), [Prediction("f01", "N", True)]
        )
    payload = json.loads(fixture.read_text())
    payload["cases"][0]["state"]["source"] += "# tampered\n"
    fixture.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fixture hash"):
        load_fresh32_cases(fixture, manifest)
    frozen = json.loads(manifest.read_text())
    frozen["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="per-case hash"):
        load_fresh32_cases(fixture, manifest)
