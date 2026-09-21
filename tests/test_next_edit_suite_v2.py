from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinycomplete.eval.next_edit_benchmark import load_next_edit_suite
from tinycomplete.eval.next_edit_protocol import build_next_edit_action_prompt

SUITE = Path("data/benchmarks/next_edit_v2.jsonl")
MANIFEST = Path("data/benchmarks/next_edit_v2.manifest.json")
CONTROL_ROOT = Path("reports/code_cpt/research_r1/next_edit_controls")


def test_v2_suite_matches_versioned_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == manifest["sha256"]
    cases = load_next_edit_suite(SUITE)
    assert len(cases) == manifest["case_count"] == 200
    assert sum(case.action == "noop" for case in cases) == 40
    assert sum(case.action == "replace" for case in cases) == 160


def test_v2_prompt_declares_delete_and_no_edit_as_different_actions() -> None:
    case = next(case for case in load_next_edit_suite(SUITE) if case.action == "replace")
    prompt = build_next_edit_action_prompt(case)

    assert '{"action":"no_edit"}' in prompt
    assert '{"action":"replace","text":"..."}' in prompt
    assert "An empty text deletes the marked region" in prompt
    assert case.expected not in prompt


def test_v2_saved_negative_controls_are_strict() -> None:
    gold = json.loads((CONTROL_ROOT / "gold" / "summary.json").read_text(encoding="utf-8"))
    current = json.loads(
        (CONTROL_ROOT / "current" / "summary.json").read_text(encoding="utf-8")
    )
    malformed = json.loads(
        (CONTROL_ROOT / "malformed" / "summary.json").read_text(encoding="utf-8")
    )

    assert gold["edit_no_edit_choice_rate"] == 1.0
    assert gold["functional_success_edit_required_rate"] == 1.0
    assert current["no_edit_recall"] == 1.0
    assert current["functional_success_edit_required_rate"] == 0.0
    assert current["false_positive_edits"] == 0
    assert malformed["malformed_count"] == 100
    assert malformed["truncation_count"] == 100
