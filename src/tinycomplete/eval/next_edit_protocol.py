"""Version 2 JSON action contract for next-edit model output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from tinycomplete.observability.hooks import observed_next_edit
from tinycomplete.protocol.events import byte_splice

PROTOCOL_VERSION = "marked-region-next-edit-action-v2"
MAX_NEW_TOKENS = 96


class NextEditAction(BaseModel):
    """Canonical action. Empty replacement text means deletion, never no-op."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["no_edit", "replace"]
    text: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> NextEditAction:
        if self.action == "no_edit" and "text" in self.model_fields_set:
            raise ValueError("no_edit must omit text")
        if self.action == "replace" and "text" not in self.model_fields_set:
            raise ValueError("replace requires text, including an empty string for deletion")
        if self.action == "replace" and self.text is None:
            raise ValueError("replace text must be a string")
        return self


class ParsedNextEditAction(BaseModel):
    status: Literal["ok", "malformed", "truncated", "overgeneration"]
    action: NextEditAction | None = None
    finish_reason: str | None = None
    generated_tokens: int | None = None
    hit_token_cap: bool = False
    error: str | None = None


def serialize_next_edit_action(action: NextEditAction) -> str:
    """Serialize the protocol independently of prompt/tokenizer formatting."""
    value: dict[str, str | None]
    if action.action == "no_edit":
        value = {"action": "no_edit"}
    else:
        value = {"action": "replace", "text": action.text}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_next_edit_action(
    raw: str,
    *,
    finish_reason: str | None = None,
    generated_tokens: int | None = None,
    max_tokens: int = MAX_NEW_TOKENS,
) -> ParsedNextEditAction:
    """Parse one JSON action while retaining stop and token-cap evidence."""
    if max_tokens <= 0 or max_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_tokens must be between 1 and {MAX_NEW_TOKENS}")
    hit_token_cap = finish_reason in {"length", "max_tokens"} or (
        generated_tokens is not None and generated_tokens >= max_tokens
    )
    leading = len(raw) - len(raw.lstrip())
    try:
        value, end = json.JSONDecoder().raw_decode(raw, leading)
    except (json.JSONDecodeError, TypeError) as exc:
        return ParsedNextEditAction(
            status="truncated" if hit_token_cap else "malformed",
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
            hit_token_cap=hit_token_cap,
            error=f"{type(exc).__name__}: {exc}",
        )
    if raw[end:].strip():
        return ParsedNextEditAction(
            status="overgeneration",
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
            hit_token_cap=hit_token_cap,
            error="non-whitespace output followed the JSON action",
        )
    try:
        action = NextEditAction.model_validate(value)
    except Exception as exc:
        return ParsedNextEditAction(
            status="malformed",
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
            hit_token_cap=hit_token_cap,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ParsedNextEditAction(
        status="ok",
        action=action,
        finish_reason=finish_reason,
        generated_tokens=generated_tokens,
        hit_token_cap=hit_token_cap,
    )


def apply_next_edit_action(
    current: str,
    region_start: int,
    region_end: int,
    action: NextEditAction,
) -> str:
    """Apply byte-precise action semantics without normalizing replacement text."""
    if action.action == "no_edit":
        byte_splice(
            current,
            region_start,
            region_end,
            current.encode("utf-8")[region_start:region_end].decode("utf-8"),
        )
        return current
    assert action.text is not None
    return byte_splice(current, region_start, region_end, action.text)


def build_next_edit_action_prompt(case) -> str:
    """Serialize the v2 action prompt without changing the v1 frozen protocol."""
    from tinycomplete.eval.next_edit_benchmark import build_next_edit_prompt

    return (
        build_next_edit_prompt(case)
        + "Return exactly one JSON object. Use "
        + '{"action":"no_edit"} to leave the region unchanged, or '
        + '{"action":"replace","text":"..."}. An empty text deletes the marked region.\n'
    )


class NextEditActionResult(BaseModel):
    case_id: str
    gold_action: Literal["no_edit", "replace"]
    predicted_action: Literal["no_edit", "replace"] | None
    parse_status: Literal["ok", "malformed", "truncated", "overgeneration"]
    action_correct: bool
    replacement_exact: bool
    gold_deletion: bool
    deletion_success: bool
    functional_success: bool
    valid_patch: bool
    wrong_action: bool
    wrong_code: bool
    finish_reason: str | None = None
    generated_tokens: int | None = None
    hit_token_cap: bool
    parse_check: str
    compile_check: str
    test_check: str


@observed_next_edit
def evaluate_next_edit_action_prediction(
    case,
    prediction,
    *,
    work_root: Path,
    execution_backend: Literal["none", "container", "trusted-host"] = "none",
    max_tokens: int = MAX_NEW_TOKENS,
) -> NextEditActionResult:
    """Parse, apply, and functionally score one v2 action response."""
    from tinycomplete.eval.code_benchmark import Prediction, evaluate_prediction

    parsed = parse_next_edit_action(
        prediction.completion,
        finish_reason=prediction.finish_reason,
        generated_tokens=prediction.generated_tokens,
        max_tokens=max_tokens,
    )
    gold_action: Literal["no_edit", "replace"] = "no_edit" if case.action == "noop" else "replace"
    predicted_action = parsed.action.action if parsed.action is not None else None
    if parsed.action is None:
        return NextEditActionResult(
            case_id=case.id,
            gold_action=gold_action,
            predicted_action=None,
            parse_status=parsed.status,
            action_correct=False,
            replacement_exact=False,
            gold_deletion=gold_action == "replace" and case.expected == "",
            deletion_success=False,
            functional_success=False,
            valid_patch=False,
            wrong_action=False,
            wrong_code=False,
            finish_reason=parsed.finish_reason,
            generated_tokens=parsed.generated_tokens,
            hit_token_cap=parsed.hit_token_cap,
            parse_check="fail",
            compile_check="not_run",
            test_check="not_run",
        )
    if parsed.action.action == "no_edit":
        replacement = case.region_text
    else:
        assert parsed.action.text is not None
        replacement = parsed.action.text
    base = evaluate_prediction(
        case.as_completion_case(),
        Prediction(
            case_id=case.id,
            completion=replacement,
            latency_seconds=prediction.latency_seconds,
            generated_tokens=prediction.generated_tokens,
            finish_reason=prediction.finish_reason,
            hit_token_cap=prediction.hit_token_cap,
        ),
        work_root=work_root,
        execution_backend=execution_backend,
    )
    valid_patch = (
        base.parse.status == "pass"
        and (not base.compile_configured or base.compile.status == "pass")
        and (not base.test_configured or base.test.status == "pass")
    )
    action_correct = predicted_action == gold_action
    expected_replacement = case.region_text if gold_action == "no_edit" else case.expected
    replacement_exact = parsed.action is not None and replacement == expected_replacement
    functional_success = action_correct and valid_patch
    return NextEditActionResult(
        case_id=case.id,
        gold_action=gold_action,
        predicted_action=predicted_action,
        parse_status=parsed.status,
        action_correct=action_correct,
        replacement_exact=replacement_exact,
        gold_deletion=gold_action == "replace" and case.expected == "",
        deletion_success=(
            gold_action == "replace"
            and case.expected == ""
            and predicted_action == "replace"
            and replacement == ""
            and valid_patch
        ),
        functional_success=functional_success,
        valid_patch=valid_patch,
        wrong_action=parsed.status == "ok" and not action_correct,
        wrong_code=parsed.status == "ok" and action_correct and not functional_success,
        finish_reason=parsed.finish_reason,
        generated_tokens=parsed.generated_tokens,
        hit_token_cap=parsed.hit_token_cap,
        parse_check=base.parse.status,
        compile_check=base.compile.status,
        test_check=base.test.status,
    )


def summarize_next_edit_action_results(results: list[NextEditActionResult]) -> dict:
    total = len(results)
    edit_required = [row for row in results if row.gold_action == "replace"]
    no_edit = [row for row in results if row.gold_action == "no_edit"]
    deletions = [row for row in edit_required if row.gold_deletion]
    return {
        "protocol": PROTOCOL_VERSION,
        "total": total,
        "edit_no_edit_choice_rate": (
            sum(row.action_correct for row in results) / total if total else None
        ),
        "no_edit_recall": (
            sum(row.predicted_action == "no_edit" for row in no_edit) / len(no_edit)
            if no_edit
            else None
        ),
        "replacement_success_rate": (
            sum(row.replacement_exact for row in edit_required) / len(edit_required)
            if edit_required
            else None
        ),
        "deletion_success_count": sum(row.deletion_success for row in deletions),
        "deletion_success_rate": (
            sum(row.deletion_success for row in deletions) / len(deletions) if deletions else None
        ),
        "functional_success_edit_required_rate": (
            sum(row.functional_success for row in edit_required) / len(edit_required)
            if edit_required
            else None
        ),
        "false_positive_edits": sum(
            row.gold_action == "no_edit" and row.predicted_action == "replace" for row in results
        ),
        "wrong_action_count": sum(row.wrong_action for row in results),
        "wrong_code_count": sum(row.wrong_code for row in results),
        "overgeneration_count": sum(row.parse_status == "overgeneration" for row in results),
        "truncation_count": sum(row.parse_status == "truncated" for row in results),
        "truncation_rate": (
            sum(row.parse_status == "truncated" for row in results) / total if total else None
        ),
        "malformed_count": sum(row.parse_status == "malformed" for row in results),
        "token_cap_count": sum(row.hit_token_cap for row in results),
        "finish_reasons": {
            reason: sum(row.finish_reason == reason for row in results)
            for reason in sorted({row.finish_reason for row in results if row.finish_reason})
        },
    }
