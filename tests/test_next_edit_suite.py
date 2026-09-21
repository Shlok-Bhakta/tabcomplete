"""Frozen next-edit suite integrity and distribution checks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tinycomplete.eval.next_edit_benchmark import load_next_edit_suite

SUITE = Path("data/benchmarks/next_edit_v1.jsonl")
MANIFEST = Path("data/benchmarks/next_edit_v1.manifest.json")


def test_frozen_next_edit_suite_matches_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == manifest["sha256"]

    cases = load_next_edit_suite(SUITE)
    assert len(cases) == manifest["cases"] == 200
    assert len({case.id for case in cases}) == len(cases)
    assert Counter(case.language for case in cases) == manifest["languages"]
    assert Counter(case.action for case in cases) == manifest["actions"]
    assert sum(case.repository_context for case in cases) == manifest["repository_context_cases"]


def test_no_gold_or_hidden_checks_leak_into_prompts() -> None:
    from tinycomplete.eval.next_edit_benchmark import build_next_edit_prompt

    for case in load_next_edit_suite(SUITE):
        prompt = build_next_edit_prompt(case)
        assert case.expected not in prompt or case.expected in case.current
        for hidden_content in case.check.files.values():
            assert hidden_content not in prompt or hidden_content in case.context_files.values()
