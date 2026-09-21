"""Tests for the causal code-output specialization benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinycomplete.eval.code_benchmark import Prediction
from tinycomplete.eval.code_output_benchmark import (
    CodeOutputCase,
    build_code_output_prompt,
    evaluate_code_output,
    load_code_output_suite,
    summarize_code_output,
)


def case(**overrides) -> CodeOutputCase:
    values = {
        "id": "python/function",
        "language": "python",
        "path": "src/example.py",
        "prefix": "def double(value: int) -> int:\n",
        "target": "code",
        "category": "function",
    }
    values.update(overrides)
    return CodeOutputCase.model_validate(values)


def test_prompt_is_pure_causal_context():
    prompt = build_code_output_prompt(case())
    assert prompt == '<file path="src/example.py">\ndef double(value: int) -> int:\n'
    assert "complete" not in prompt.lower()
    assert "instruction" not in prompt.lower()
    assert "fim" not in prompt.lower()


def test_direct_code_passes_behavior_and_parse():
    result = evaluate_code_output(
        case(), Prediction(case_id="python/function", completion="    return value * 2\n")
    )
    assert result.immediate_code is True
    assert result.behavior_pass is True
    assert result.conversational_preamble is False
    assert result.markdown_fence is False
    assert result.parse.status == "pass"


@pytest.mark.parametrize(
    "completion",
    [
        "    # Keep arithmetic explicit.\n    return value * 2\n",
        '    """Return twice the input."""\n    return value * 2\n',
    ],
)
def test_natural_comments_and_docstrings_count_as_immediate_code(completion: str):
    result = evaluate_code_output(
        case(), Prediction(case_id="python/function", completion=completion)
    )
    assert result.immediate_code is True
    assert result.behavior_pass is True
    assert result.parse.status == "pass"


def test_conversational_preamble_and_fence_are_separate_failures():
    preamble = evaluate_code_output(
        case(),
        Prediction(
            case_id="python/function",
            completion="Sure! Here's the implementation:\n    return value * 2\n",
        ),
    )
    fenced = evaluate_code_output(
        case(),
        Prediction(case_id="python/function", completion="```python\n    return value * 2\n```"),
    )
    assert preamble.conversational_preamble is True
    assert preamble.immediate_code is False
    assert fenced.markdown_fence is True
    assert fenced.behavior_pass is False


def test_markdown_context_allows_prose_and_fences_without_code_penalty():
    markdown = case(
        id="markdown/readme",
        language="markdown",
        path="README.md",
        prefix="# Usage\n\nThe client connects",
        target="markdown",
        category="documentation",
    )
    result = evaluate_code_output(
        markdown,
        Prediction(
            case_id=markdown.id, completion=" to the local server.\n\n```sh\ncurl /health\n```"
        ),
    )
    assert result.nonempty is True
    assert result.immediate_code is None
    assert result.behavior_pass is None
    assert result.parse.status == "not_run"


def test_summary_excludes_markdown_from_code_rates():
    code_result = evaluate_code_output(
        case(), Prediction(case_id="python/function", completion="    return value * 2\n")
    )
    markdown_case = case(
        id="markdown/readme",
        language="markdown",
        path="README.md",
        prefix="# Usage\n",
        target="markdown",
        category="documentation",
    )
    markdown_result = evaluate_code_output(
        markdown_case, Prediction(case_id=markdown_case.id, completion="Some prose.")
    )
    summary = summarize_code_output([code_result, markdown_result])
    assert summary["code_cases"] == 1
    assert summary["markdown_cases"] == 1
    assert summary["behavior_pass_rate"] == 1.0
    assert summary["code_output_pass_rate"] == 1.0
    assert summary["markdown_nonempty_rate"] == 1.0


def test_suite_rejects_duplicates_and_bad_target_language_pairs(tmp_path: Path):
    record = case().model_dump(mode="json")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate benchmark id"):
        load_code_output_suite(suite)
    with pytest.raises(ValueError, match="markdown targets"):
        case(language="python", target="markdown")
    with pytest.raises(ValueError, match="safe and relative"):
        case(path="../escape.py")
