"""Frozen behavioral benchmark for code-first causal continuations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .code_benchmark import CheckResult, Prediction

CodeLanguage = Literal[
    "python", "typescript", "javascript", "java", "cpp", "rust", "go", "c", "csharp"
]


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("benchmark paths must be safe and relative")
    return str(path)


class CodeOutputCase(BaseModel):
    id: str
    language: CodeLanguage | Literal["markdown"]
    path: str
    prefix: str = Field(min_length=1)
    target: Literal["code", "markdown"]
    category: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_path(value)

    def model_post_init(self, __context: object) -> None:
        if self.target == "markdown" and self.language != "markdown":
            raise ValueError("markdown targets must use the markdown language")
        if self.target == "code" and self.language == "markdown":
            raise ValueError("code targets must use a programming language")


class CodeOutputResult(BaseModel):
    case_id: str
    language: str
    target: str
    category: str
    nonempty: bool
    conversational_preamble: bool
    markdown_fence: bool
    immediate_code: bool | None
    behavior_pass: bool | None
    parse: CheckResult
    latency_seconds: float | None = None
    generated_tokens: int | None = None


_PREAMBLE = re.compile(
    r"^(?:sure\b|certainly\b|of course\b|here(?:'s| is| are)\b|below is\b|"
    r"to (?:implement|complete|solve)\b|this (?:function|method|code)\b|"
    r"the (?:function|method|code)\b|i(?:'ll| will| can)\b)",
    flags=re.IGNORECASE,
)


def load_code_output_suite(path: Path) -> list[CodeOutputCase]:
    cases: list[CodeOutputCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = CodeOutputCase.model_validate_json(line)
            if case.id in seen:
                raise ValueError(f"duplicate benchmark id at line {line_number}: {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("code-output benchmark suite is empty")
    return cases


def build_code_output_prompt(case: CodeOutputCase) -> str:
    """Return a pure causal prompt. It contains no request or gold answer."""
    return f'<file path="{case.path}">\n{case.prefix}'


def _parse(source: str, language: str) -> CheckResult:
    grammar = {"csharp": "csharp"}.get(language, language)
    try:
        from tree_sitter_language_pack import get_parser

        root = get_parser(grammar).parse(source.encode()).root_node
    except Exception as exc:
        return CheckResult(status="unavailable", stderr=type(exc).__name__)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            return CheckResult(status="fail")
        stack.extend(node.children)
    return CheckResult(status="pass")


def evaluate_code_output(case: CodeOutputCase, prediction: Prediction) -> CodeOutputResult:
    if prediction.case_id != case.id:
        raise ValueError(f"prediction id {prediction.case_id!r} does not match {case.id!r}")
    completion = prediction.completion
    stripped = completion.lstrip()
    nonempty = bool(stripped)
    conversational = bool(_PREAMBLE.match(stripped))
    fenced = "```" in completion or "~~~" in completion

    if case.target == "code":
        immediate = nonempty and not conversational and not stripped.startswith(("```", "~~~"))
        behavior_pass = immediate and not fenced
        parse = _parse(case.prefix + completion, case.language)
    else:
        immediate = None
        behavior_pass = None
        parse = CheckResult(status="not_run")

    return CodeOutputResult(
        case_id=case.id,
        language=case.language,
        target=case.target,
        category=case.category,
        nonempty=nonempty,
        conversational_preamble=conversational,
        markdown_fence=fenced,
        immediate_code=immediate,
        behavior_pass=behavior_pass,
        parse=parse,
        latency_seconds=prediction.latency_seconds,
        generated_tokens=prediction.generated_tokens,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_code_output(results: list[CodeOutputResult]) -> dict[str, object]:
    if not results:
        raise ValueError("cannot summarize empty results")
    code = [result for result in results if result.target == "code"]
    markdown = [result for result in results if result.target == "markdown"]
    parse_available = [result for result in code if result.parse.status != "unavailable"]

    def code_breakdown(items: list[CodeOutputResult]) -> dict[str, object]:
        return {
            "total": len(items),
            "immediate_code_rate": _rate(
                sum(item.immediate_code is True for item in items), len(items)
            ),
            "conversational_preamble_rate": _rate(
                sum(item.conversational_preamble for item in items), len(items)
            ),
            "markdown_fence_rate": _rate(sum(item.markdown_fence for item in items), len(items)),
            "behavior_pass_rate": _rate(
                sum(item.behavior_pass is True for item in items), len(items)
            ),
        }

    languages = sorted({result.language for result in code})
    return {
        "total": len(results),
        "code_cases": len(code),
        "markdown_cases": len(markdown),
        **code_breakdown(code),
        "parse_available": len(parse_available),
        "parse_pass_rate": _rate(
            sum(item.parse.status == "pass" for item in parse_available), len(parse_available)
        ),
        "code_output_pass_rate": _rate(
            sum(
                item.behavior_pass is True and item.parse.status == "pass"
                for item in parse_available
            ),
            len(parse_available),
        ),
        "markdown_nonempty_rate": _rate(sum(item.nonempty for item in markdown), len(markdown)),
        "by_language": {
            language: code_breakdown([item for item in code if item.language == language])
            for language in languages
        },
        "parse_statuses": dict(sorted(Counter(item.parse.status for item in code).items())),
    }


def load_predictions(path: Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            prediction = Prediction.model_validate_json(line)
            if prediction.case_id in predictions:
                raise ValueError(
                    f"duplicate prediction at line {line_number}: {prediction.case_id}"
                )
            predictions[prediction.case_id] = prediction
    return predictions


def write_results(path: Path, results: list[CodeOutputResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
