"""Fail-closed plan-7 calibration runner checks without provider calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tinycomplete.one_line.teacher import TeacherResponse

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_one_line_teacher_calibration.py"
_SPEC = importlib.util.spec_from_file_location("one_line_teacher_calibration_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class ByteTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(value.encode("utf-8"))


def test_frozen_plan7_files_verify() -> None:
    protocol = runner.verify_frozen()
    assert protocol["version"] == "one-line-teacher-protocol-v3"
    assert "<FINAL_ACTION>" in protocol["calibration"]["fixed_instruction"]


def test_runner_records_complete_response_and_raw_hash(tmp_path) -> None:
    response = TeacherResponse(
        content="Brief reasoning.<FINAL_ACTION>\nD\n</FINAL_ACTION>",
        session_id="session",
        response_id="response",
        model_id="opencode-go/muse-spark-1.3-contributor",
        input_tokens=100,
        output_tokens=8,
        reasoning_tokens=20,
        total_tokens_reported=128,
        cached_read_tokens=0,
        cached_write_tokens=0,
        finish_reason="stop",
        cost_usd_reported=0.001,
    )
    row = runner.make_row("f01", "synthetic prompt", 10, response, ByteTokenizer())
    assert row["extracted_wire"] == "D"
    assert row["generated_output_tokens"] == 28
    path = tmp_path / "raw.jsonl"
    runner.append_row(path, row)
    assert runner.completed_rows(path, {"f01"})["f01"]["raw_output_sha256"] == runner.sha_text(
        response.content
    )
    with pytest.raises(ValueError, match="duplicate"):
        runner.completed_rows(path, set())
    row["raw_content"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="content hash"):
        runner.completed_rows(path, {"f01"})


def test_runner_rejects_incomplete_provider_finish() -> None:
    response = TeacherResponse(
        content="<FINAL_ACTION>\nN\n</FINAL_ACTION>",
        session_id="session",
        response_id="response",
        model_id="opencode-go/muse-spark-1.3-contributor",
        input_tokens=100,
        output_tokens=8,
        reasoning_tokens=20,
        total_tokens_reported=128,
        cached_read_tokens=0,
        cached_write_tokens=0,
        finish_reason="length",
        cost_usd_reported=0.001,
    )
    row = runner.make_row("f01", "synthetic prompt", 10, response, ByteTokenizer())
    assert row["extracted_status"] == "invalid"
    assert row["extracted_wire"] is None
    assert row["provider_message_complete"] is False
