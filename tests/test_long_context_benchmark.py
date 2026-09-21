"""Long-context dependency-use benchmark generation."""

from __future__ import annotations

import re
from pathlib import Path

from tinycomplete.eval.code_benchmark import Prediction, evaluate_prediction
from tinycomplete.eval.code_generation import build_causal_prompt
from tinycomplete.eval.long_context_benchmark import (
    LongContextSuiteSpec,
    materialize_long_context_case,
    materialize_long_context_suite,
    suite_metadata,
    summarize_long_context_results,
)


class RegexTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"[A-Za-z_]+|\d+|[^\w\s]", text))))


def small_spec() -> LongContextSuiteSpec:
    return LongContextSuiteSpec(
        model_id="test/tokenizer",
        tokenizer_revision="fixed",
        target_tokens=[768, 1536],
        needle_depths=[0.2, 0.8],
        dependency_kinds=["constant", "signature"],
        seed=17,
        tolerance_fraction=0.1,
    )


def test_materialization_is_deterministic_and_calibrated():
    tokenizer = RegexTokenizer()
    first = materialize_long_context_suite(small_spec(), tokenizer)
    second = materialize_long_context_suite(small_spec(), tokenizer)
    assert suite_metadata(small_spec(), first) == suite_metadata(small_spec(), second)
    assert len(first) == 4
    for item in first:
        assert abs(item.prompt_tokens - item.target_tokens) <= max(
            64, round(item.target_tokens * 0.1)
        )
        assert abs(item.actual_needle_depth - item.requested_needle_depth) < 0.15
        assert item.dependency_path in item.case.context_files
        assert item.case.repository_context is True


def test_prompt_has_real_repository_code_without_gold_or_fim_tokens():
    item = materialize_long_context_case(
        tokenizer=RegexTokenizer(),
        target_tokens=1024,
        needle_depth=0.5,
        kind="field",
        seed=19,
        tolerance_fraction=0.1,
    )
    prompt = build_causal_prompt(item.case)
    assert "class EndpointPolicy" in prompt
    assert "class Component" in prompt
    assert item.case.expected not in prompt
    assert "<fim_" not in prompt.lower()
    assert "[[EDIT]]" not in prompt
    assert prompt.endswith(item.case.prefix)


def test_every_dependency_kind_has_an_executable_gold_completion(tmp_path: Path):
    tokenizer = RegexTokenizer()
    for index, kind in enumerate(("constant", "enum", "signature", "field", "config")):
        item = materialize_long_context_case(
            tokenizer=tokenizer,
            target_tokens=768,
            needle_depth=0.5,
            kind=kind,
            seed=index,
            tolerance_fraction=0.1,
        )
        result = evaluate_prediction(
            item.case,
            Prediction(case_id=item.case.id, completion=item.case.expected),
            work_root=tmp_path / kind,
            execution_backend="trusted-host",
        )
        assert result.parse.status == "pass", kind
        assert result.compile.status == "pass", kind
        assert result.test.status == "pass", kind


def test_dependency_blind_completion_fails_behavior_checks(tmp_path: Path):
    tokenizer = RegexTokenizer()
    for index, kind in enumerate(("constant", "enum", "signature", "field", "config")):
        item = materialize_long_context_case(
            tokenizer=tokenizer,
            target_tokens=768,
            needle_depth=0.5,
            kind=kind,
            seed=index,
            tolerance_fraction=0.1,
        )
        result = evaluate_prediction(
            item.case,
            Prediction(case_id=item.case.id, completion="0"),
            work_root=tmp_path / kind,
            execution_backend="trusted-host",
        )
        assert result.test.status == "fail", kind


def test_dependency_positions_rotate_across_sizes_and_kinds():
    cases = materialize_long_context_suite(small_spec(), RegexTokenizer())
    requested = {(item.target_tokens, item.requested_needle_depth) for item in cases}
    assert requested == {(768, 0.2), (768, 0.8), (1536, 0.2), (1536, 0.8)}


def test_frozen_suite_fingerprint_changes_when_prompt_content_changes():
    spec = small_spec()
    cases = materialize_long_context_suite(spec, RegexTokenizer())
    original = suite_metadata(spec, cases)["suite_sha256"]
    changed_case = cases[0].case.model_copy(
        update={"prefix": cases[0].case.prefix + "# changed\n"}
    )
    cases[0] = cases[0].__class__(
        case=changed_case,
        target_tokens=cases[0].target_tokens,
        prompt_tokens=cases[0].prompt_tokens,
        requested_needle_depth=cases[0].requested_needle_depth,
        actual_needle_depth=cases[0].actual_needle_depth,
        dependency_path=cases[0].dependency_path,
        prompt_sha256="changed",
    )
    assert suite_metadata(spec, cases)["suite_sha256"] != original


def test_summary_slices_results_by_context_length_and_depth(tmp_path: Path):
    spec = small_spec()
    cases = materialize_long_context_suite(spec, RegexTokenizer())
    results = [
        evaluate_prediction(
            item.case,
            Prediction(case_id=item.case.id, completion=item.case.expected),
            work_root=tmp_path / f"case-{index}",
            execution_backend="none",
        )
        for index, item in enumerate(cases)
    ]
    summary = summarize_long_context_results(suite_metadata(spec, cases), results)
    assert summary["overall"]["total"] == 4
    assert summary["by_target_tokens"]["768"]["total"] == 2
    assert summary["by_target_tokens"]["1536"]["total"] == 2
    assert summary["by_requested_needle_depth"]["0.20"]["total"] == 2
    assert summary["by_dependency_kind"]["signature"]["total"] == 2
