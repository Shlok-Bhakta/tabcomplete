"""No network or provider invocation occurs in these policy tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tinycomplete.one_line import teacher


def test_opencode_route_requires_recorded_approval_basis() -> None:
    for source_class in ("public", "synthetic"):
        with pytest.raises(teacher.TeacherPolicyError, match="approval basis missing"):
            teacher.assert_opencode_request_allowed(
                model_id=teacher.MODEL_ID,
                purpose="student_label",
                source_class=source_class,
                prompt="def f(x): return x + 1",
                authorization_basis="",
            )
        teacher.assert_opencode_request_allowed(
            model_id=teacher.MODEL_ID,
            purpose="student_label",
            source_class=source_class,
            prompt="def f(x): return x + 1",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
        )


def test_private_sealed_and_secret_input_rejected_before_terms() -> None:
    for source_class in ("private", "sealed"):
        with pytest.raises(teacher.TeacherPolicyError, match="private or sealed"):
            teacher.assert_opencode_request_allowed(
                model_id=teacher.MODEL_ID,
                purpose="student_label",
                source_class=source_class,
                prompt="def harmless(): pass",
                authorization_basis=teacher.AUTHORIZATION_BASIS,
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
                authorization_basis=teacher.AUTHORIZATION_BASIS,
            )


def test_unapproved_model_and_purpose_rejected() -> None:
    with pytest.raises(teacher.TeacherPolicyError, match="unapproved teacher model"):
        teacher.assert_opencode_request_allowed(
            model_id="another-model",
            purpose="student_label",
            source_class="synthetic",
            prompt="fixture",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
        )
    with pytest.raises(teacher.TeacherPolicyError, match="unapproved teacher purpose"):
        teacher.assert_opencode_request_allowed(
            model_id=teacher.MODEL_ID,
            purpose="other",  # type: ignore[arg-type]
            source_class="synthetic",
            prompt="fixture",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
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


def test_reasoning_accounting_correction_is_append_only(tmp_path: Path) -> None:
    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    ledger.reserve("old-smoke", input_tokens=100, max_output_tokens=50)
    ledger.settle("old-smoke", input_tokens=20, output_tokens=3)
    ledger.correct_output_accounting(
        "old-smoke",
        previous_output_tokens=3,
        generated_tokens_including_reasoning=15,
    )
    assert ledger.totals() == teacher.UsageTotals(1, 20, 15)
    assert [json.loads(line)["kind"] for line in ledger.path.read_text().splitlines()] == [
        "reserve",
        "settle",
        "correct",
    ]
    with pytest.raises(teacher.TeacherBudgetError, match="invalid usage correction"):
        ledger.correct_output_accounting(
            "old-smoke",
            previous_output_tokens=3,
            generated_tokens_including_reasoning=15,
        )


def test_strict_candidate_parser_rejects_narration_and_bad_shapes() -> None:
    assert teacher.parse_candidate_action('{"action":"keep"}') == {"action": "keep"}
    assert teacher.parse_candidate_action('{"action":"replace_line","text":"λ  "}') == {
        "action": "replace_line",
        "text": "λ  ",
    }
    for value in (
        'Keep requested. {"action":"keep"}',
        '```json\n{"action":"keep"}\n```',
        '{"action":"keep"}{"action":"keep"}',
        '{"action":"delete_line","text":""}',
        '{"action":"insert_before","text":"a\\nb"}',
        '{"action":"other"}',
    ):
        with pytest.raises(teacher.TeacherCandidateError):
            teacher.parse_candidate_action(value)


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def test_v6_final_action_block_allows_prose_but_one_exact_terminal_wire() -> None:
    tokenizer = ByteTokenizer()
    for content, wire, kind in (
        ("<FINAL_ACTION>\nN\n</FINAL_ACTION>", "N", "keep"),
        ("Reasoning here.\n<FINAL_ACTION>\nD\n</FINAL_ACTION>", "D", "delete_line"),
        ("One line of prose.\n<FINAL_ACTION>\nR\tλ  \n</FINAL_ACTION>", "R\tλ  ", "replace_line"),
        ("<FINAL_ACTION>\nI\t\n</FINAL_ACTION>", "I\t", "insert_before"),
    ):
        parsed = teacher.extract_final_action_block(
            content,
            provider_complete=True,
            tokenizer=tokenizer,
        )
        assert parsed.wire == wire
        assert parsed.action.kind == kind
        assert parsed.wire_plus_q25_eos_tokens == len(wire.encode()) + 1


def test_v6_final_action_block_rejects_malformed_multiple_and_unterminated() -> None:
    tokenizer = ByteTokenizer()
    for content in (
        "Reasoning N",
        "text<FINAL_ACTION>\nN\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nN",
        "<FINAL_ACTION>\nN\n</FINAL_ACTION>extra",
        "<FINAL_ACTION>\nN\n</FINAL_ACTION>\n",
        "<FINAL_ACTION>\nN\nD\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nN\r\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nN\n</FINAL_ACTION>\n<FINAL_ACTION>\nD\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nR text\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nR\tfirst\nsecond\n</FINAL_ACTION>",
    ):
        with pytest.raises(teacher.TeacherCandidateError):
            teacher.extract_final_action_block(
                content,
                provider_complete=True,
                tokenizer=tokenizer,
            )
    with pytest.raises(teacher.TeacherCandidateError, match="incomplete"):
        teacher.extract_final_action_block(
            "<FINAL_ACTION>\nN\n</FINAL_ACTION>",
            provider_complete=False,
            tokenizer=tokenizer,
        )
    with pytest.raises(teacher.TeacherCandidateError, match="incomplete"):
        teacher.extract_final_action_block(
            "<FINAL_ACTION>\nR\t" + "a" * 64 + "\n</FINAL_ACTION>",
            provider_complete=True,
            tokenizer=tokenizer,
        )


def test_role_wrapper_uses_exact_model_and_reports_usage(tmp_path: Path) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "session-1"})
        assert request.url.path == "/session/session-1/message"
        body = json.loads(request.content)
        assert body["model"] == {
            "providerID": "opencode-go",
            "modelID": "muse-spark-1.3-contributor",
        }
        assert body["tools"] == {}
        assert body["system"] == "Fixed protocol instruction."
        return httpx.Response(
            200,
            json={
                "info": {
                    "id": "message-1",
                    "providerID": "opencode-go",
                    "modelID": "muse-spark-1.3-contributor",
                    "finish": "stop",
                    "tokens": {
                        "input": 27,
                        "output": 9,
                        "reasoning": 4,
                        "total": 40,
                        "cache": {"read": 3, "write": 0},
                    },
                    "cost": 0.0,
                },
                "parts": [{"type": "text", "text": '{"action":"keep"}'}],
            },
        )

    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    client = teacher.OpenCodeTeacherClient(ledger)
    client._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(response)
    )
    try:
        result = client.run_role(
            request_id="r1",
            prompt="synthetic fixture",
            purpose="calibration",
            source_class="synthetic",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
            system_instruction="Fixed protocol instruction.",
        )
    finally:
        client.__exit__(None, None, None)
    assert result.content == '{"action":"keep"}'
    assert result.session_id == "session-1"
    assert result.input_tokens == 27 and result.output_tokens == 9
    assert result.reasoning_tokens == 4 and result.total_tokens_reported == 40
    assert ledger.totals() == teacher.UsageTotals(1, 27, 13)
