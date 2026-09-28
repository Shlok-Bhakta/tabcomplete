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


def test_persisted_usage_overrun_is_append_only_and_single_use(tmp_path: Path) -> None:
    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    digest = "a" * 64
    ledger.reserve("failed-public", input_tokens=10, max_output_tokens=8)
    with pytest.raises(teacher.TeacherBudgetError, match="reserved limit"):
        ledger.settle("failed-public", input_tokens=14, output_tokens=3)
    ledger.settle_overrun(
        "failed-public", input_tokens=14, output_tokens=3, evidence_sha256=digest
    )
    assert ledger.totals() == teacher.UsageTotals(1, 14, 3)
    assert [json.loads(line)["kind"] for line in ledger.path.read_text().splitlines()] == [
        "reserve", "settle_overrun"
    ]
    with pytest.raises(teacher.TeacherBudgetError, match="already settled"):
        ledger.settle_overrun(
            "failed-public", input_tokens=14, output_tokens=3, evidence_sha256=digest
        )
    with pytest.raises(teacher.TeacherBudgetError, match="duplicate"):
        ledger.reserve("failed-public", input_tokens=10, max_output_tokens=8)


def test_overrun_rejects_missing_evidence_nonoverrun_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    ledger.reserve("r", input_tokens=10, max_output_tokens=8)
    with pytest.raises(teacher.TeacherBudgetError, match="evidence hash"):
        ledger.settle_overrun("r", input_tokens=11, output_tokens=2, evidence_sha256="bad")
    with pytest.raises(teacher.TeacherBudgetError, match="did not exceed"):
        ledger.settle_overrun("r", input_tokens=9, output_tokens=2, evidence_sha256="b" * 64)
    monkeypatch.setattr(teacher, "MAX_INPUT_TOKENS", 12)
    with pytest.raises(teacher.TeacherBudgetError, match="cap exceeded"):
        ledger.settle_overrun("r", input_tokens=13, output_tokens=2, evidence_sha256="b" * 64)
    assert ledger.totals() == teacher.UsageTotals(1, 10, 8)
    assert [json.loads(line)["kind"] for line in ledger.path.read_text().splitlines()] == [
        "reserve"
    ]


def test_overrun_replay_rejects_tampered_or_duplicate_events(tmp_path: Path) -> None:
    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    ledger.reserve("r", input_tokens=10, max_output_tokens=8)
    ledger.settle_overrun("r", input_tokens=14, output_tokens=3, evidence_sha256="c" * 64)
    valid = ledger.path.read_text()
    event = json.loads(valid.splitlines()[-1])
    for bad in (
        {**event, "input_tokens": 9},
        {**event, "evidence_sha256": "wrong"},
        {**event, "reason": "other"},
    ):
        ledger.path.write_text(valid.splitlines()[0] + "\n" + json.dumps(bad) + "\n")
        with pytest.raises(teacher.TeacherBudgetError, match="invalid teacher usage ledger"):
            ledger.totals()
    ledger.path.write_text(valid + valid.splitlines()[-1] + "\n")
    with pytest.raises(teacher.TeacherBudgetError, match="invalid teacher usage ledger"):
        ledger.totals()
    ledger.path.write_text(valid)
    assert ledger.totals() == teacher.UsageTotals(1, 14, 3)


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


def test_v7_final_action_block_allows_exact_unique_sentinel_after_prose() -> None:
    tokenizer = ByteTokenizer()
    content = "Some prose.<FINAL_ACTION>\nR\tλ  \n</FINAL_ACTION>"
    parsed = teacher.extract_final_action_block_v7(
        content, provider_complete=True, tokenizer=tokenizer
    )
    assert parsed.wire == "R\tλ  "
    assert parsed.action.kind == "replace_line"
    assert parsed.wire_plus_q25_eos_tokens == len(parsed.wire.encode()) + 1


def test_v7_final_action_block_rejects_duplicates_incomplete_and_overcap() -> None:
    tokenizer = ByteTokenizer()
    bad = (
        "<FINAL_ACTION>\nN\n</FINAL_ACTION>extra",
        "<FINAL_ACTION>\nN\n</FINAL_ACTION>\n",
        "<FINAL_ACTION>\nN\nD\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nN\r\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nN\n</FINAL_ACTION><FINAL_ACTION>\nD\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nN\n</FINAL_ACTION></FINAL_ACTION>",
        "<FINAL_ACTION>N\n</FINAL_ACTION>",
        "<FINAL_ACTION>\nR\t" + "a" * 64 + "\n</FINAL_ACTION>",
    )
    for content in bad:
        with pytest.raises(teacher.TeacherCandidateError):
            teacher.extract_final_action_block_v7(
                content, provider_complete=True, tokenizer=tokenizer
            )
    with pytest.raises(teacher.TeacherCandidateError, match="incomplete"):
        teacher.extract_final_action_block_v7(
            "<FINAL_ACTION>\nN\n</FINAL_ACTION>",
            provider_complete=False,
            tokenizer=tokenizer,
        )


def test_author_candidate_terminal_block_accepts_one_strict_object() -> None:
    content = (
        'Reasoning.<AUTHOR_CANDIDATE>\n{"action":{"kind":"N","text":null}}'
        "\n</AUTHOR_CANDIDATE>"
    )
    extracted = teacher.extract_author_candidate_block(content, provider_complete=True)
    assert extracted.value == {"action": {"kind": "N", "text": None}}
    assert extracted.json_text == '{"action":{"kind":"N","text":null}}'


def test_author_candidate_terminal_block_rejects_rescue_and_incomplete() -> None:
    examples = (
        'answer {"action":1}',
        '<AUTHOR_CANDIDATE>{"action":1}\n</AUTHOR_CANDIDATE>',
        '<AUTHOR_CANDIDATE>\n{"action":1}\n</AUTHOR_CANDIDATE> extra',
        '<AUTHOR_CANDIDATE>\n{"action":1}\n</AUTHOR_CANDIDATE>\n',
        '<AUTHOR_CANDIDATE>\n{"action":1}\n</AUTHOR_CANDIDATE>' * 2,
        '<AUTHOR_CANDIDATE>\n{"action":1,"action":2}\n</AUTHOR_CANDIDATE>',
        '<AUTHOR_CANDIDATE>\n{"action":NaN}\n</AUTHOR_CANDIDATE>',
        '<AUTHOR_CANDIDATE>\n[1]\n</AUTHOR_CANDIDATE>',
        '<AUTHOR_CANDIDATE>\n{"action":1}\r\n</AUTHOR_CANDIDATE>',
        '<AUTHOR_CANDIDATE>\n{"action":1}\n</AUTHOR_CANDIDATE>\n' + "x" * 16_385,
    )
    for content in examples:
        with pytest.raises(teacher.TeacherCandidateError):
            teacher.extract_author_candidate_block(content, provider_complete=True)
    with pytest.raises(teacher.TeacherCandidateError, match="incomplete"):
        teacher.extract_author_candidate_block(
            '<AUTHOR_CANDIDATE>\n{"action":1}\n</AUTHOR_CANDIDATE>',
            provider_complete=False,
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


def test_role_wrapper_reads_v11831_structured_field(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
        "additionalProperties": False,
    }

    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "session-structured"})
        body = json.loads(request.content)
        assert body["format"] == {"type": "json_schema", "schema": schema}
        return httpx.Response(
            200,
            json={
                "info": {
                    "id": "message-structured",
                    "providerID": "opencode-go",
                    "modelID": "muse-spark-1.3-contributor",
                    "finish": "stop",
                    "structured": {"action": "keep"},
                    "tokens": {
                        "input": 10,
                        "output": 5,
                        "reasoning": 2,
                        "total": 17,
                        "cache": {"read": 0, "write": 0},
                    },
                    "cost": 0.0,
                },
                "parts": [{"type": "text", "text": "unstructured narration"}],
            },
        )

    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    client = teacher.OpenCodeTeacherClient(ledger)
    client._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(response)
    )
    try:
        result = client.run_role(
            request_id="structured-1",
            prompt="synthetic fixture",
            purpose="calibration",
            source_class="synthetic",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
            output_schema=schema,
        )
    finally:
        client.__exit__(None, None, None)
    assert result.content == '{"action":"keep"}'
    assert ledger.totals() == teacher.UsageTotals(1, 10, 7)


def test_role_wrapper_does_not_rescue_missing_structured_with_text(tmp_path: Path) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "session-missing"})
        return httpx.Response(
            200,
            json={
                "info": {
                    "id": "message-missing",
                    "providerID": "opencode-go",
                    "modelID": "muse-spark-1.3-contributor",
                    "finish": "stop",
                    "structured_output": {"action": "keep"},
                    "tokens": {
                        "input": 10,
                        "output": 5,
                        "reasoning": 2,
                        "total": 17,
                        "cache": {"read": 0, "write": 0},
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
        with pytest.raises(teacher.TeacherTransportError):
            client.run_role(
                request_id="structured-missing",
                prompt="synthetic fixture",
                purpose="calibration",
                source_class="synthetic",
                authorization_basis=teacher.AUTHORIZATION_BASIS,
                output_schema={"type": "object"},
            )
    finally:
        client.__exit__(None, None, None)
    assert ledger.totals() == teacher.UsageTotals(1, 100_000, 4_096)


def test_structured_mode_is_explicit_and_allows_only_output_tool(tmp_path: Path) -> None:
    default = json.loads(teacher._isolated_env()["OPENCODE_CONFIG_CONTENT"])
    assert default["permission"] == "deny"
    assert default["tools"] == {"*": False}
    isolated = teacher._isolated_env(
        structured_output_only=True, config_home=tmp_path / "config"
    )
    config = json.loads(isolated["OPENCODE_CONFIG_CONTENT"])
    assert list(config["permission"].items()) == [
        ("*", "deny"),
        ("StructuredOutput", "allow"),
    ]
    assert "tools" not in config
    assert config["plugin"] == [] and config["mcp"] == {}
    assert isolated["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert isolated["XDG_CONFIG_HOME"] == str(tmp_path / "config")
    with pytest.raises(teacher.TeacherPolicyError, match="isolated config home"):
        teacher._isolated_env(structured_output_only=True)


def test_structured_mode_request_has_no_per_message_tool_override(tmp_path: Path) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "session-isolated"})
        body = json.loads(request.content)
        assert "tools" not in body
        assert body["format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "info": {
                    "id": "message-isolated",
                    "providerID": "opencode-go",
                    "modelID": "muse-spark-1.3-contributor",
                    "finish": "stop",
                    "structured": {"action": "keep"},
                    "tokens": {
                        "input": 10,
                        "output": 5,
                        "reasoning": 2,
                        "total": 17,
                        "cache": {"read": 0, "write": 0},
                    },
                    "cost": 0.0,
                },
                "parts": [],
            },
        )

    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    client = teacher.OpenCodeTeacherClient(ledger, structured_output_only=True)
    client._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(response)
    )
    try:
        with pytest.raises(teacher.TeacherPolicyError, match="requires a JSON schema"):
            client.run_role(
                request_id="no-schema",
                prompt="synthetic fixture",
                purpose="calibration",
                source_class="synthetic",
                authorization_basis=teacher.AUTHORIZATION_BASIS,
            )
        result = client.run_role(
            request_id="with-schema",
            prompt="synthetic fixture",
            purpose="calibration",
            source_class="synthetic",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
            output_schema={"type": "object"},
        )
    finally:
        client.__exit__(None, None, None)
    assert result.content == '{"action":"keep"}'
    assert ledger.totals() == teacher.UsageTotals(1, 10, 7)


def test_api_error_without_optional_total_is_fail_closed(tmp_path: Path) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "session-error"})
        return httpx.Response(
            200,
            json={
                "info": {
                    "id": "message-error",
                    "providerID": "opencode-go",
                    "modelID": "muse-spark-1.3-contributor",
                    "error": {"name": "APIError", "data": {"statusCode": 400}},
                    "tokens": {
                        "input": 0,
                        "output": 0,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                    "cost": 0,
                },
                "parts": [],
            },
        )

    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    client = teacher.OpenCodeTeacherClient(ledger, structured_output_only=True)
    client._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(response)
    )
    try:
        with pytest.raises(teacher.TeacherTransportError, match="failed or usage was invalid"):
            client.run_role(
                request_id="api-error",
                prompt="synthetic fixture",
                purpose="calibration",
                source_class="synthetic",
                authorization_basis=teacher.AUTHORIZATION_BASIS,
                output_schema={"type": "object"},
            )
    finally:
        client.__exit__(None, None, None)
    assert ledger.totals() == teacher.UsageTotals(1, 100_000, 4_096)


def test_success_without_optional_total_uses_reported_components(tmp_path: Path) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "session-success"})
        return httpx.Response(
            200,
            json={
                "info": {
                    "id": "message-success",
                    "providerID": "opencode-go",
                    "modelID": "muse-spark-1.3-contributor",
                    "finish": "stop",
                    "structured": {"action": "keep"},
                    "tokens": {
                        "input": 10,
                        "output": 5,
                        "reasoning": 2,
                        "cache": {"read": 0, "write": 0},
                    },
                    "cost": 0.0,
                },
                "parts": [],
            },
        )

    ledger = teacher.TeacherUsageLedger(tmp_path / "usage.jsonl")
    client = teacher.OpenCodeTeacherClient(ledger, structured_output_only=True)
    client._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(response)
    )
    try:
        result = client.run_role(
            request_id="no-total",
            prompt="synthetic fixture",
            purpose="calibration",
            source_class="synthetic",
            authorization_basis=teacher.AUTHORIZATION_BASIS,
            output_schema={"type": "object"},
        )
    finally:
        client.__exit__(None, None, None)
    assert result.total_tokens_reported is None
    assert ledger.totals() == teacher.UsageTotals(1, 10, 7)
