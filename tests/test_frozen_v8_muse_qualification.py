from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tinycomplete.one_line import candidate_acceptance

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/export_frozen_v8_muse_acceptance.py"
SPEC = importlib.util.spec_from_file_location("frozen_v8_muse_qualification", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.chmod(path, 0o600)
    os.chmod(path.parent, 0o700)


def _self_hashed_document(path: Path, key: str, body: dict[str, Any]) -> tuple[str, str]:
    identity = MODULE._digest(body)
    payload = json.dumps({**body, key: identity}, sort_keys=True, indent=2).encode() + b"\n"
    _write_private(path, payload)
    return hashlib.sha256(payload).hexdigest(), identity


def _frozen_plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    private = tmp_path / "private"
    packet = tmp_path / "packet"
    private.mkdir(mode=0o700)
    packet.mkdir(mode=0o700)
    old_plan_file, old_plan_id = _self_hashed_document(
        private / "qualification-plan.json", "plan_sha256", {"revision": 1}
    )
    old_result_file, old_result_id = _self_hashed_document(
        private / "qualification-result.json", "result_sha256", {"status": "complete"}
    )
    v2_plan_file, v2_plan_id = _self_hashed_document(
        private / "qualification-plan-v2.json", "plan_sha256", {"revision": 2}
    )
    v3_plan_file, v3_plan_id = _self_hashed_document(
        private / "qualification-plan-v3.json",
        "plan_sha256",
        {"schema": "frozen-v8-muse-cpu-qualification-plan-v3"},
    )
    v3_result_file, v3_result_id = _self_hashed_document(
        private / "qualification-result-v3.json", "result_sha256", {"status": "complete"}
    )
    v4_plan_file, v4_plan_id = _self_hashed_document(
        private / "qualification-plan-v4.json",
        "plan_sha256",
        {"schema": "frozen-v8-muse-cpu-qualification-plan-v4"},
    )
    v4_result_file, v4_result_id = _self_hashed_document(
        private / "qualification-result-v4.json", "result_sha256", {"status": "complete"}
    )

    packet_plan = {"schema": "test-packet", "plan_sha256": "a" * 64}
    artifact = {"plan_canonical_sha256": packet_plan["plan_sha256"]}
    _write_private(packet / "plan.json", json.dumps(packet_plan).encode())
    _write_private(packet / "artifact_manifest.json", json.dumps(artifact).encode())

    hashes = {"inputs_sha256": "b" * 64, "oracles_sha256": "c" * 64}
    code_hashes = {"runner": "d" * 64}
    preflight = {
        "cases": [
            {
                "case_id": "synthetic-a",
                "source_seed_sha256": "e" * 64,
                "source_transform_sha256": "f" * 64,
                "source_transform_rows": [
                    {"row": 4, "classification": "synthetic_prestate_signature"},
                    {"row": 5, "classification": "privacy_redaction"},
                    {"row": 16, "classification": "privacy_redaction"},
                ],
            }
        ]
    }
    reconciliation = {
        "author_seed_sha256": "e" * 64,
        "parent_to_seed_transform_sha256": "f" * 64,
    }
    plan: dict[str, Any] = {
        "schema": "frozen-v8-muse-cpu-qualification-plan-v5",
        "status": "frozen_before_cpu_acceptance",
        "revision": 5,
        "supersedes": {
            "plan_sha256": old_plan_id,
            "plan_file_sha256": old_plan_file,
            "result_sha256": old_result_id,
            "result_file_sha256": old_result_file,
            "failed_plan_sha256": v2_plan_id,
            "failed_plan_file_sha256": v2_plan_file,
        },
        "earlier_successful_revision": {
            "plan_sha256": v3_plan_id,
            "plan_file_sha256": v3_plan_file,
            "result_sha256": v3_result_id,
            "result_file_sha256": v3_result_file,
        },
        "previous_successful_revision": {
            "plan_sha256": v4_plan_id,
            "plan_file_sha256": v4_plan_file,
            "result_sha256": v4_result_id,
            "result_file_sha256": v4_result_file,
        },
        "row_export": {
            "schema": "one-line-muse-accepted-rows-manifest-v1",
            "rows_path": str(private / "packages-v5" / "accepted_rows.jsonl"),
            "manifest_path": str(private / "packages-v5" / "accepted_rows_manifest.json"),
            "package_root": str(private / "packages-v5"),
            "selection": "objective+role+semantic+portable-verified rows only",
            "maximum_rows": 2,
        },
        "upstream": {"plan_canonical_sha256": packet_plan["plan_sha256"], **hashes},
        "code_sha256": code_hashes,
        "tokenizer": {"tokenizer_json_sha256": "1" * 64},
        "sandbox_policy": {
            "target_path_policy": {
                "python": "solution.py",
                "typescript": "tooltip-view.ts",
                "source_repository_path_remains_provenance_only": True,
            }
        },
        "source_role_preflight": preflight,
        "source_provenance_reconciliation": reconciliation,
    }
    plan["plan_sha256"] = MODULE._digest(plan)
    _write_private(
        private / "qualification-plan-v5.json",
        json.dumps(plan, sort_keys=True, indent=2).encode() + b"\n",
    )
    monkeypatch.setattr(MODULE, "PRIVATE_ROOT", private)
    monkeypatch.setattr(MODULE, "PLAN_PATH", private / "qualification-plan-v5.json")
    package_root = private / "packages-v5"
    monkeypatch.setattr(MODULE, "PACKAGE_ROOT", package_root)
    monkeypatch.setattr(MODULE, "ROWS_PATH", package_root / "accepted_rows.jsonl")
    monkeypatch.setattr(MODULE, "ROWS_MANIFEST_PATH", package_root / "accepted_rows_manifest.json")
    monkeypatch.setattr(MODULE, "PACKET", packet)
    monkeypatch.setattr(MODULE, "PREVIOUS_PLAN_SHA256", old_plan_id)
    monkeypatch.setattr(MODULE, "PREVIOUS_PLAN_FILE_SHA256", old_plan_file)
    monkeypatch.setattr(MODULE, "PREVIOUS_RESULT_SHA256", old_result_id)
    monkeypatch.setattr(MODULE, "PREVIOUS_RESULT_FILE_SHA256", old_result_file)
    monkeypatch.setattr(MODULE, "FAILED_V2_PLAN_SHA256", v2_plan_id)
    monkeypatch.setattr(MODULE, "FAILED_V2_PLAN_FILE_SHA256", v2_plan_file)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V3_PLAN_SHA256", v3_plan_id)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V3_PLAN_FILE_SHA256", v3_plan_file)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V3_RESULT_SHA256", v3_result_id)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V3_RESULT_FILE_SHA256", v3_result_file)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V4_PLAN_SHA256", v4_plan_id)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V4_PLAN_FILE_SHA256", v4_plan_file)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V4_RESULT_SHA256", v4_result_id)
    monkeypatch.setattr(MODULE, "SUCCESSFUL_V4_RESULT_FILE_SHA256", v4_result_file)
    monkeypatch.setattr(MODULE, "_packet_hashes", lambda: hashes)
    monkeypatch.setattr(MODULE, "_file_hashes", lambda: code_hashes)
    monkeypatch.setattr(MODULE, "_tokenizer_hash", lambda: "1" * 64)
    monkeypatch.setattr(MODULE, "_no_output_preflight", lambda _plan: preflight)
    return {"private": private, "plan": plan, "old_plan_file": private / "qualification-plan.json"}


def test_frozen_plan_validator_keeps_file_and_canonical_hashes_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_plan_fixture(tmp_path, monkeypatch)
    verified = MODULE._verify_plan()
    assert verified["plan_sha256"] == fixture["plan"]["plan_sha256"]

    # Whitespace preserves canonical identity, but must still fail the pinned file hash.
    path = fixture["old_plan_file"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file hash mismatch"):
        MODULE._verify_plan()


def test_exporter_objective_path_flows_into_sandbox_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = {
        "runtime_image": "python@sha256:" + "a" * 64,
        "runtime_command": ["python", "oracle.py"],
        "test_source": "pass",
    }
    entry = MODULE._objective_entry(
        candidate_id="case-1", source_id="source-1", filetype="python", fixture=fixture
    )
    seen: dict[str, str] = {}

    def fake_evaluate(case: Any, _prediction: Any, **_kwargs: Any) -> Any:
        seen["path"] = case.path
        return SimpleNamespace(
            parse=SimpleNamespace(status="pass"),
            compile=SimpleNamespace(status="not_run"),
            test=SimpleNamespace(status="pass"),
        )

    monkeypatch.setattr(candidate_acceptance, "evaluate_prediction", fake_evaluate)
    assert candidate_acceptance._sandbox_check("def value(): return 7\n", entry, "python") is True
    assert entry["path"] == "solution.py"
    assert seen["path"] == entry["path"]
