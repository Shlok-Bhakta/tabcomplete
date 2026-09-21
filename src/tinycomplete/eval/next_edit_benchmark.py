"""Executable benchmark contract for marked-region next-edit prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .code_benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    CheckSpec,
    Prediction,
    _safe_relative_path,
    evaluate_prediction,
    summarize_results,
)


class NextEditCase(BaseModel):
    """One editor state with a fixed region whose next replacement is predicted."""

    id: str
    language: Literal[
        "python", "typescript", "javascript", "java", "cpp", "rust", "go", "c", "csharp"
    ]
    path: str
    current: str
    region_start: int = Field(ge=0)
    region_end: int = Field(ge=0)
    expected: str = ""
    action: Literal["replace", "noop"] = "replace"
    recent_edits: tuple[str, ...] = ()
    context_files: dict[str, str] = Field(default_factory=dict)
    check: CheckSpec = Field(default_factory=CheckSpec)
    category: str = "unspecified"
    repository_context: bool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @field_validator("context_files")
    @classmethod
    def validate_context_files(cls, value: dict[str, str]) -> dict[str, str]:
        return {_safe_relative_path(path): content for path, content in value.items()}

    @model_validator(mode="after")
    def validate_region_and_action(self) -> NextEditCase:
        raw = self.current.encode("utf-8")
        if self.region_start > self.region_end or self.region_end > len(raw):
            raise ValueError("next-edit region is outside current file")
        try:
            raw[: self.region_start].decode("utf-8")
            raw[self.region_start : self.region_end].decode("utf-8")
            raw[self.region_end :].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("next-edit region must lie on UTF-8 boundaries") from exc
        if self.action == "noop" and self.expected:
            raise ValueError("noop cases must have an empty expected replacement")
        if self.action == "replace" and not self.expected:
            raise ValueError("replace cases need a non-empty expected replacement")
        if self.action == "replace" and self.expected == self.region_text:
            raise ValueError("replace cases must change the marked region")
        visible_paths = set(self.context_files)
        hidden_paths = set(self.check.files)
        if self.path in visible_paths or self.path in hidden_paths:
            raise ValueError("target path must not collide with context or check files")
        collisions = visible_paths & hidden_paths
        if collisions:
            raise ValueError(f"context and check file paths collide: {sorted(collisions)[0]}")
        return self

    @property
    def region_text(self) -> str:
        raw = self.current.encode("utf-8")
        return raw[self.region_start : self.region_end].decode("utf-8")

    def as_completion_case(self) -> BenchmarkCase:
        raw = self.current.encode("utf-8")
        return BenchmarkCase(
            id=self.id,
            language=self.language,
            path=self.path,
            prefix=raw[: self.region_start].decode("utf-8"),
            suffix=raw[self.region_end :].decode("utf-8"),
            expected=self.expected or self.region_text,
            context_files=self.context_files,
            check=self.check,
            category=self.category,
            repository_context=self.repository_context,
        )


class NextEditResult(BenchmarkResult):
    gold_action: Literal["replace", "noop"]
    predicted_action: Literal["replace", "noop"]
    action_correct: bool
    character_edit_distance: int
    valid_patch: bool


def _edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance without allocating a full matrix."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def load_next_edit_suite(path: Path) -> list[NextEditCase]:
    cases: list[NextEditCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = NextEditCase.model_validate_json(line)
            if case.id in seen:
                raise ValueError(f"duplicate benchmark id at line {line_number}: {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("next-edit benchmark suite is empty")
    return cases


def build_next_edit_prompt(case: NextEditCase) -> str:
    """Serialize exactly the marked-region contract intended for Stage 2."""
    raw = case.current.encode("utf-8")
    before = raw[: case.region_start].decode("utf-8")
    region = raw[case.region_start : case.region_end].decode("utf-8")
    after = raw[case.region_end :].decode("utf-8")
    pieces = [f"<repo {case.path}>\n"]
    pieces.extend(f"{edit}\n" for edit in case.recent_edits)
    for path, content in sorted(case.context_files.items()):
        pieces.append(f"<context {path}>\n{content}\n</context>\n")
    pieces.append(
        f"<file {case.path}>\n{before}[[EDIT]]{region}[[/EDIT]]{after}\n</file>\n"
        f"<P {case.path} {case.region_start}>\n"
    )
    return "".join(pieces)


def evaluate_next_edit_prediction(
    case: NextEditCase,
    prediction: Prediction,
    *,
    work_root: Path,
    execution_backend: Literal["none", "container", "trusted-host"] = "none",
) -> NextEditResult:
    predicted_action: Literal["replace", "noop"] = (
        "noop" if prediction.completion == "" else "replace"
    )
    applied_prediction = prediction
    if predicted_action == "noop":
        applied_prediction = prediction.model_copy(update={"completion": case.region_text})
    base = evaluate_prediction(
        case.as_completion_case(),
        applied_prediction,
        work_root=work_root,
        execution_backend=execution_backend,
    )
    valid_patch = (
        base.parse.status == "pass"
        and (not base.compile_configured or base.compile.status == "pass")
        and (not base.test_configured or base.test.status == "pass")
    )
    return NextEditResult(
        **base.model_copy(
            update={
                "exact_match": prediction.completion == case.expected,
                "normalized_exact_match": prediction.completion.strip() == case.expected.strip(),
                "nonempty": bool(prediction.completion.strip()),
            }
        ).model_dump(),
        gold_action=case.action,
        predicted_action=predicted_action,
        action_correct=predicted_action == case.action,
        character_edit_distance=_edit_distance(prediction.completion, case.expected),
        valid_patch=valid_patch,
    )


def summarize_next_edit_results(results: list[NextEditResult]) -> dict:
    summary = summarize_results(results)  # type: ignore[arg-type]
    summary["action_accuracy"] = (
        sum(row.action_correct for row in results) / len(results) if results else None
    )
    true_noop = sum(
        row.gold_action == "noop" and row.predicted_action == "noop" for row in results
    )
    false_noop = sum(
        row.gold_action == "replace" and row.predicted_action == "noop" for row in results
    )
    missed_noop = sum(
        row.gold_action == "noop" and row.predicted_action == "replace" for row in results
    )
    noop_precision = true_noop / (true_noop + false_noop) if true_noop + false_noop else None
    noop_recall = true_noop / (true_noop + missed_noop) if true_noop + missed_noop else None
    noop_f1 = (
        2 * noop_precision * noop_recall / (noop_precision + noop_recall)
        if noop_precision is not None and noop_recall is not None and noop_precision + noop_recall
        else None
    )
    summary["action_confusion"] = {
        "gold_noop_predicted_noop": true_noop,
        "gold_noop_predicted_replace": missed_noop,
        "gold_replace_predicted_noop": false_noop,
        "gold_replace_predicted_replace": sum(
            row.gold_action == "replace" and row.predicted_action == "replace" for row in results
        ),
    }
    summary["noop_precision"] = noop_precision
    summary["noop_recall"] = noop_recall
    summary["noop_f1"] = noop_f1
    summary["valid_patch_rate"] = (
        sum(row.valid_patch for row in results) / len(results) if results else None
    )
    summary["mean_character_edit_distance"] = (
        sum(row.character_edit_distance for row in results) / len(results) if results else None
    )
    summary["by_gold_action"] = {
        action: summarize_results(  # type: ignore[arg-type]
            [row for row in results if row.gold_action == action]
        )
        for action in ("replace", "noop")
    }
    return summary


def write_next_edit_results(path: Path, results: list[NextEditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
