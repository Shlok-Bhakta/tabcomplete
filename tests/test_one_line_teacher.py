"""No network or provider invocation occurs in these policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinycomplete.one_line import teacher


def test_opencode_route_is_closed_even_for_public_synthetic_input() -> None:
    for source_class in ("public", "synthetic"):
        with pytest.raises(teacher.TeacherPolicyError, match="output-use terms"):
            teacher.assert_opencode_request_allowed(
                model_id=teacher.MODEL_ID,
                purpose="student_label",
                source_class=source_class,
                prompt="def f(x): return x + 1",
            )


def test_private_sealed_and_secret_input_rejected_before_terms() -> None:
    for source_class in ("private", "sealed"):
        with pytest.raises(teacher.TeacherPolicyError, match="private or sealed"):
            teacher.assert_opencode_request_allowed(
                model_id=teacher.MODEL_ID,
                purpose="student_label",
                source_class=source_class,
                prompt="def harmless(): pass",
            )
    for payload in (
        "api_key=do-not-send",
        "-----BEGIN PRIVATE KEY-----\nredacted\n-----END PRIVATE KEY-----",
        "sk-1234567890abcdefghij",
    ):
        with pytest.raises(teacher.TeacherPolicyError, match="credential"):
            teacher.assert_opencode_request_allowed(
                model_id=teacher.MODEL_ID,
                purpose="student_label",
                source_class="synthetic",
                prompt=payload,
            )


def test_unapproved_model_and_purpose_rejected() -> None:
    with pytest.raises(teacher.TeacherPolicyError, match="unapproved teacher model"):
        teacher.assert_opencode_request_allowed(
            model_id="another-model", purpose="student_label",
            source_class="synthetic", prompt="fixture",
        )
    with pytest.raises(teacher.TeacherPolicyError, match="unapproved teacher purpose"):
        teacher.assert_opencode_request_allowed(
            model_id=teacher.MODEL_ID, purpose="other",  # type: ignore[arg-type]
            source_class="synthetic", prompt="fixture",
        )


def test_ledger_reservations_persist_and_missing_usage_fails(tmp_path: Path) -> None:
    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    assert ledger.totals() == teacher.UsageTotals()
    ledger.reserve("r1", input_tokens=10, max_output_tokens=5)
    assert teacher.TeacherUsageLedger(ledger.path).totals() == teacher.UsageTotals(1, 10, 5)
    with pytest.raises(teacher.TeacherBudgetError, match="missing"):
        ledger.settle("r1", input_tokens=None, output_tokens=2)
    assert ledger.totals() == teacher.UsageTotals(1, 10, 5)
    ledger.settle("r1", input_tokens=8, output_tokens=2)
    assert ledger.totals() == teacher.UsageTotals(1, 8, 2)
    with pytest.raises(teacher.TeacherBudgetError, match="duplicate"):
        ledger.reserve("r1", input_tokens=1, max_output_tokens=1)
    with pytest.raises(teacher.TeacherBudgetError, match="already settled"):
        ledger.settle("r1", input_tokens=8, output_tokens=2)


def test_ledger_caps_and_corruption_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(teacher, "MAX_CALLS", 1)
    monkeypatch.setattr(teacher, "MAX_INPUT_TOKENS", 10)
    monkeypatch.setattr(teacher, "MAX_OUTPUT_TOKENS", 4)
    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    ledger.reserve("r1", input_tokens=10, max_output_tokens=4)
    with pytest.raises(teacher.TeacherBudgetError, match="cap exceeded"):
        ledger.reserve("r2", input_tokens=0, max_output_tokens=0)
    with pytest.raises(teacher.TeacherBudgetError, match="reserved limit"):
        ledger.settle("r1", input_tokens=11, output_tokens=1)
    assert ledger.totals() == teacher.UsageTotals(1, 10, 4)
    ledger.path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(teacher.TeacherBudgetError, match="invalid teacher usage ledger"):
        ledger.reserve("r2", input_tokens=0, max_output_tokens=0)
