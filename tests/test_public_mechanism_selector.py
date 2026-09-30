"""Offline, deterministic guards for the public-source preflight selector."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/select_public_mechanism_pilot.py"
BANK = ROOT / "artifacts/research/one_line_r1/public_source_authoring_100.jsonl"
SPEC = importlib.util.spec_from_file_location("public_mechanism_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def _license_record(source: str, file_notice: str | None = None) -> dict:
    return {
        "id": "public-source/test",
        "authoring_metadata": {
            "source_license": "MIT",
            "license_path": "LICENSE",
            "license_sha256": "a" * 64,
            "file_spdx_notice": file_notice,
        },
        "student_state_seed": {"source": source},
    }


def test_helper_reuse_uses_source_lines_and_ignores_comments_and_literals() -> None:
    source = """# def ghost():
message = "ghost() ghost()"
def normalize(value):
    return value.strip()
a = normalize(first)
b = normalize(second)
# normalize(third)
"""
    evidence = selector.detect_source_evidence(source, "python")
    selected = evidence["local_helper_reuse"]["evidence"]
    assert [entry["line"] for entry in selected] == [3, 5, 6]
    assert [entry["role"] for entry in selected] == [
        "function declaration",
        "call site",
        "call site",
    ]
    assert all(
        entry["line_sha256"] == hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
        for entry in selected
    )


def test_sort_family_requires_code_ordering_signal_not_comment_or_string() -> None:
    false_signal = """// values.sort(key=item.key < other.key)
message = "items.sort((a,b) => a < b)"
"""
    assert "sort_call_with_ordering_signal" not in selector.detect_source_evidence(
        false_signal, "typescript"
    )
    source = """const ordered = values.sort((left, right) => left.rank < right.rank ? -1 : 1);
"""
    evidence = selector.detect_source_evidence(source, "typescript")[
        "sort_call_with_ordering_signal"
    ]
    assert evidence["evidence"][0]["role"] == "sort-like call"
    assert evidence["evidence"][1]["role"] == "nearby ordering/comparator/key signal"
    assert evidence["evidence"][0]["line"] == evidence["evidence"][1]["line"] == 1


def test_qualified_member_family_requires_two_observed_cross_type_names() -> None:
    source = """enum InputTag { Alpha, Beta }
let a = InputTag::Alpha;
let b = Value::Alpha;
let c = InputTag::Beta;
let d = Value::Beta;
"""
    found = selector.detect_source_evidence(source, "rust")
    evidence = found["qualified_type_member_overlap"]["evidence"]
    assert len(evidence) == 4
    assert [entry["role"] for entry in evidence] == [
        "InputTag::Alpha",
        "Value::Alpha",
        "InputTag::Beta",
        "Value::Beta",
    ]
    one_member = """let a = InputTag::Alpha;
let b = Value::Alpha;
"""
    assert "qualified_type_member_overlap" not in selector.detect_source_evidence(
        one_member, "rust"
    )


def test_license_screen_rejects_conflict_and_marks_unnoted_scope() -> None:
    conflict = _license_record("// SPDX-License-Identifier: Apache-2.0\nfn f() {}\n")
    ok, _, reason = selector._license_review(conflict)
    assert not ok
    assert reason == "file_spdx_conflicts_with_repository_mit"

    unnoted = _license_record("fn f() {}\n")
    ok, review, reason = selector._license_review(unnoted)
    assert ok and reason is None
    assert review["file_scope"] == "unverified_without_file_notice"


def test_pinned_bank_selection_is_repeatable_bounded_and_focus_free() -> None:
    raw = BANK.read_bytes()
    plan, selected = selector.build_selection(raw, SCRIPT.read_bytes())
    repeated_plan, repeated_selected = selector.build_selection(raw, SCRIPT.read_bytes())
    assert plan == repeated_plan
    assert selected == repeated_selected
    assert plan["source_bank_sha256"] == selector.EXPECTED_BANK_SHA256
    assert plan["selector_inputs"] == {
        "authoring_focus_fields_read": False,
        "candidate_split_fields_read": False,
        "source_downloads": 0,
        "provider_calls": 0,
    }
    results = plan["results"]
    assert results["source_rows"] == 100
    assert results["source_rows_passing_provenance_and_license_screen"] == 100
    assert results["structural_candidates"] == 24
    assert 0 < results["selected_source_seeds"] <= 24
    assert set(results["detected_evidence_family_counts"]) == set(selector.FAMILY_ORDER)
    assert all(results["selected_allocation_queue_counts"].values())
    assert len({item["source_provenance"]["source_repo"] for item in selected}) == len(selected)
    assert len({item["seed_id"] for item in selected}) == len(selected)
    assert all(item["training_eligible"] is False for item in selected)
    assert all("authoring_focus" not in json.dumps(item) for item in selected)
    assert all(
        item["source_provenance"]["file_license_scope"] == "unverified_without_file_notice"
        for item in selected
    )
    assert results["remaining_missing_evidence"]["accepted_training_labels"] == 0
    assert plan["teacher_call_gate"].startswith("no call from this script")


def test_bank_hash_mismatch_refuses_to_select() -> None:
    with pytest.raises(ValueError, match="pinned source bank hash mismatch"):
        selector.build_selection(BANK.read_bytes() + b"\n", SCRIPT.read_bytes())


def test_output_is_private_and_frozen(tmp_path: Path) -> None:
    plan, selected = selector.build_selection(BANK.read_bytes(), SCRIPT.read_bytes())
    output = tmp_path / "pilot"
    plan_path, seeds_path = selector.write_outputs(output, plan, selected)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(seeds_path.stat().st_mode) == 0o600
    assert selector.write_outputs(output, plan, selected) == (plan_path, seeds_path)

    changed = copy.deepcopy(plan)
    changed["status"] = "changed"
    with pytest.raises(ValueError, match="frozen output differs"):
        selector.write_outputs(output, changed, selected)
