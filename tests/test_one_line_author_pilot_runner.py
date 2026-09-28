"""Author pilot identity and restart guards without provider calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_one_line_author_pilot.py"
_SPEC = importlib.util.spec_from_file_location("one_line_author_pilot_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class ByteTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        return list(value.encode("utf-8")) + ([0] if add_special_tokens else [])


def test_frozen_plan9_and_full_source_preflight() -> None:
    protocol = runner.verify_frozen()
    assert protocol["version"] == "one-line-author-text-v1"
    assert len(runner.load_inputs("smoke")) == 4
    assert len(runner.load_inputs("public")) == 100
    assert runner.run("smoke", execute=False)["preflight_sources"] == 4


def test_author_result_shape_and_terminal_validation() -> None:
    source = runner.load_inputs("smoke")[0]
    answer = {
        "prior_edit": {
            "row": 2,
            "old_text": '    return "a-" + value',
            "new_text": '    return "a:" + value',
        },
        "target_row": 5,
        "action": {"kind": "R", "text": '    return "a:" + value'},
        "intent_evidence": "visible sibling edit",
        "objective": {"kind": "shared_rule", "description": "match sibling", "checks": []},
    }
    content = "Reasoning.<AUTHOR_CANDIDATE>\n" + json.dumps(answer) + "\n</AUTHOR_CANDIDATE>"
    status, reason, candidate = runner.evaluate_response(
        "smoke", source, content, "stop", ByteTokenizer()
    )
    assert status == "candidate_preflight" and reason is None
    assert candidate is not None and candidate["validation"]["accepted_training"] is False
    status, reason, candidate = runner.evaluate_response(
        "smoke", source, content, "length", ByteTokenizer()
    )
    assert status == "rejected" and "incomplete" in reason and candidate is None
    status, reason, candidate = runner.evaluate_response(
        "smoke", source, "prefix " + json.dumps(answer), "stop", ByteTokenizer()
    )
    assert status == "rejected" and candidate is None
    answer["prior_edit"]["old_text"] = "missing line"
    impossible = "<AUTHOR_CANDIDATE>\n" + json.dumps(answer) + "\n</AUTHOR_CANDIDATE>"
    status, reason, candidate = runner.evaluate_response(
        "smoke", source, impossible, "stop", ByteTokenizer()
    )
    assert status == "rejected" and candidate is None


def test_resume_rejects_tampering_and_public_requires_smoke_gate(
    tmp_path: Path, monkeypatch
) -> None:
    source = runner.load_inputs("smoke")[0]
    prompt = runner.source_preflight(source, synthetic=True)
    row = {
        "source_id": source["id"],
        "request_id": runner.request_id("smoke", source["id"]),
        "plan_sha256": runner.PLAN_SHA,
        "protocol_sha256": runner.PROTOCOL_SHA,
        "prompt_sha256": runner.sha_bytes(prompt.encode()),
        "raw_content": "answer",
        "raw_output_sha256": runner.sha_bytes(b"answer"),
    }
    path = tmp_path / "raw.jsonl"
    runner.append_row(path, row)
    assert len(runner.completed_rows(path, "smoke", [source])) == 1
    row["raw_content"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="raw hash"):
        runner.completed_rows(path, "smoke", [source])
    monkeypatch.setitem(runner.RESULT, "smoke", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="has not completed"):
        runner.assert_smoke_gate()
