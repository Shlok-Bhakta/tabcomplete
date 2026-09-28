"""CPU checks for the complete-output guard and fixed audit sample."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_one_line_author_pilot_plan10 as audit  # noqa: E402


def test_sample_uses_fixed_sha_ranking() -> None:
    source_ids = [f"source-{index:03d}" for index in range(100)]
    expected = sorted(
        source_ids,
        key=lambda item: hashlib.sha256(
            b"plan10-human-audit-v1\0" + item.encode("utf-8")
        ).hexdigest(),
    )[:12]
    assert audit.selected_source_ids(source_ids) == expected
    assert audit.selected_source_ids(list(reversed(source_ids))) == expected
    with pytest.raises(ValueError, match="duplicate eligible source ID"):
        audit.selected_source_ids(["same", "same"])


def test_audit_refuses_to_open_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_summary = tmp_path / "not-finished.json"
    monkeypatch.setattr(audit, "COMBINED_SUMMARY", missing_summary)
    with pytest.raises(ValueError, match="do not inspect partial results"):
        audit.audit()
    missing_summary.write_text('{"durable_rows": 98, "accepted_training": 0}')
    with pytest.raises(ValueError, match="incomplete"):
        audit.audit()


def test_raw_artifact_path_stays_in_pilot_directory() -> None:
    assert (
        audit._artifact_path("artifacts/research/one_line_r1/author_pilot_plan11.jsonl").name
        == "author_pilot_plan11.jsonl"
    )
    with pytest.raises(ValueError, match="outside the pilot directory"):
        audit._artifact_path("../private.jsonl")


def test_partial_refuses_wrong_incident_before_raw_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({"plan_revision": 10, "durable_candidates": 8}))
    monkeypatch.setattr(audit, "PARTIAL_INCIDENT", incident)
    with pytest.raises(ValueError, match="incident identity or denominator changed"):
        audit.audit_partial()
