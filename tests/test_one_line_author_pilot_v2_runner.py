"""Plan-10 indexed author pilot guards without provider calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_one_line_author_pilot_v2.py"
_SPEC = importlib.util.spec_from_file_location("one_line_author_pilot_v2_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class ByteTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        return list(value.encode("utf-8")) + ([0] if add_special_tokens else [])


def test_v2_frozen_identities_preflight_and_no_old_output_reuse() -> None:
    protocol = runner.verify_frozen()
    assert protocol["version"] == "one-line-author-text-v2"
    assert len(runner.load_inputs("smoke")) == 4
    assert len(runner.load_inputs("public")) == 100
    dry_run = runner.run("smoke", execute=False)
    assert dry_run["phase"] == "smoke" and dry_run["preflight_sources"] == 4
    assert dry_run["completed"] == len(
        runner.completed_rows(runner.RAW["smoke"], "smoke", runner.load_inputs("smoke"))
    )
    assert runner.RAW["smoke"].name == "author_indexed_smoke_plan10.jsonl"
    assert runner.RAW["smoke"].name != "author_format_smoke_plan9.jsonl"
    assert runner.request_id("smoke", "author-indexed-smoke/python").startswith("author10-")


def test_v2_exact_zero_based_replay_is_preflight_only() -> None:
    source = runner.load_inputs("smoke")[0]
    answer = {
        "prior_edit": {
            "row": 2,
            "old_text": "    return value",
            "new_text": "    return value.strip()",
        },
        "target_row": 5,
        "action": {"kind": "R", "text": "    return value.strip()"},
        "intent_evidence": "The two status functions follow the same visible rule.",
        "objective": {"kind": "shared_rule", "description": "Both trim spaces.", "checks": []},
    }
    content = "<AUTHOR_CANDIDATE>\n" + json.dumps(answer) + "\n</AUTHOR_CANDIDATE>"
    status, reason, candidate = runner.evaluate_response(
        "smoke", source, content, "stop", ByteTokenizer()
    )
    assert status == "candidate_preflight" and reason is None
    assert candidate is not None and candidate["validation"]["accepted_training"] is False
    answer["prior_edit"]["row"] = 3
    shifted = "<AUTHOR_CANDIDATE>\n" + json.dumps(answer) + "\n</AUTHOR_CANDIDATE>"
    status, reason, candidate = runner.evaluate_response(
        "smoke", source, shifted, "stop", ByteTokenizer()
    )
    assert status == "rejected" and "pre-edit line" in reason and candidate is None


def test_v2_resume_rejects_plan9_rows(tmp_path: Path) -> None:
    source = runner.load_inputs("smoke")[0]
    old = {
        "source_id": source["id"],
        "request_id": "author9-smoke-old",
        "plan_sha256": "dce75fba6703b907022184b68266f81bb534823581089266a39e74c19d99aae9",
        "protocol_sha256": "1862934521d0530d3b8878a7121f27e057d27f97be92f4365db9cf62f62787c72",
        "prompt_sha256": "old",
        "raw_content": "old",
        "raw_output_sha256": runner.sha_bytes(b"old"),
    }
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(old) + "\n")
    with pytest.raises(ValueError, match="frozen identity"):
        runner.completed_rows(path, "smoke", [source])
