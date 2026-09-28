"""Frozen continuation checks never invoke the external teacher."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "one_line_author_continuation_runner", SCRIPTS / "run_one_line_author_continuation.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_plan11_frozen_preflight_skips_all_historical_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, protocol, sources = runner.verify_frozen()
    assert spec["remaining_source_count"] == 92
    assert protocol["version"] == "one-line-author-text-v2"
    assert len(sources) == 100
    monkeypatch.setattr(runner, "RAW", tmp_path / "new_rows.jsonl")
    assert runner.run(execute=False) == {
        "phase": "public_continuation",
        "preflight_sources": 92,
        "completed": 0,
        "skipped_historical": 8,
    }
    assert {source["id"] for source in sources[:8]}.isdisjoint(
        {source["id"] for source in sources[8:]}
    )
    assert all(
        runner.request_id(source["id"]).startswith("author11-public-") for source in sources[8:]
    )
    assert all(
        runner.request_id(source["id"]) != runner.previous_request_id("public", source["id"])
        for source in sources
    )


def test_continuation_rejects_plan10_and_tampered_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = runner.load_inputs("public")
    path = tmp_path / "rows.jsonl"
    monkeypatch.setattr(runner, "RAW", path)
    old = json.loads(runner.PREVIOUS_RAW["public"].read_text().splitlines()[0])
    path.write_text(json.dumps(old) + "\n")
    with pytest.raises(ValueError, match="unknown or duplicate"):
        runner.completed_rows(sources)
    source = sources[8]
    content = "<AUTHOR_CANDIDATE>\n{}\n</AUTHOR_CANDIDATE>"
    row = {
        "source_id": source["id"],
        "plan_sha256": runner.PLAN_SHA,
        "spec_sha256": runner.SPEC_SHA,
        "request_id": runner.request_id(source["id"]),
        "prompt_sha256": runner.sha_bytes(
            runner.source_preflight(source, synthetic=False).encode()
        ),
        "raw_content": content,
        "raw_output_sha256": runner.sha_bytes(content.encode()),
        "accepted_training": False,
    }
    path.write_text(json.dumps(row) + "\n")
    assert list(runner.completed_rows(sources)) == [source["id"]]
    for bad in (
        {**row, "request_id": old["request_id"]},
        {**row, "raw_output_sha256": "0" * 64},
        {**row, "accepted_training": True},
    ):
        path.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match="identity changed"):
            runner.completed_rows(sources)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        runner.completed_rows(sources)


def test_combined_summary_has_auditable_source_and_raw_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = runner.load_inputs("public")
    raw = tmp_path / "continuation.jsonl"
    raw.write_text("synthetic test artifact\n")
    previous = tmp_path / "previous.jsonl"
    previous.write_text("synthetic historical artifact\n")
    combined = tmp_path / "combined.json"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RAW", raw)
    monkeypatch.setattr(runner, "PREVIOUS_RAW", {"public": previous})
    monkeypatch.setattr(runner, "COMBINED", combined)
    rows = {
        source["id"]: {
            "status": "rejected",
            "input_tokens": 1,
            "generated_tokens": 1,
            "reported_cost_usd": 0.0,
        }
        for source in sources[8:]
    }
    result = runner.summarize(rows, sources)
    assert result["completed_new_calls"] == 92
    summary = json.loads(combined.read_text())
    assert summary["durable_rows"] == 99
    assert summary["failed_source_id"] == sources[7]["id"]
    assert summary["source_sha256"] == runner.SOURCE_SHA
    assert summary["protocol_sha256"] == runner.PROTOCOL_SHA
    assert summary["plan10_raw_artifact_sha256"] == runner.sha_file(runner.PREVIOUS_RAW["public"])
    assert summary["plan11_raw_artifact_sha256"] == runner.sha_file(raw)
    assert summary["accepted_training"] == 0
