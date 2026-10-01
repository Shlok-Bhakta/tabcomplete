from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditState, physical_lines
from tinycomplete.one_line.public_prefix_pilot import sha256_bytes
from tinycomplete.one_line.public_synthetic_pilot import (
    read_proof,
    validate_implementation_evidence,
)
from tinycomplete.one_line.synthetic_functional_mix import (
    FUNCTION_CONTRACTS,
    build_synthetic_candidates,
    validate_synthetic_family_split,
)


def test_synthetic_families_are_grouped_and_near_duplicates_do_not_cross():
    candidates = build_synthetic_candidates()
    audit = validate_synthetic_family_split([c.row for c in candidates])
    assert audit["rows"] == 192
    assert audit["families_by_split"] == {"train": 12, "development": 4}
    near = defaultdict(set)
    for candidate in candidates:
        near[candidate.row["near_duplicate_sha256"]].add(candidate.row["split"])
        ast.parse(candidate.row["after_source"])
    assert all(len(splits) == 1 for splits in near.values())


def test_deletions_preserve_the_latest_edit_and_contracts_do_not_name_actions():
    for candidate in build_synthetic_candidates():
        row = candidate.row
        state = EditState.from_mapping(row["state"])
        assert state.file_id == "synthetic/example.py"
        assert len(state.history) == 1
        if row["action"]["kind"] == "delete_line":
            edit = state.history[-1]
            assert edit.row != state.target_row
            assert edit.new_text.encode() in [
                line.content for line in physical_lines(row["after_source"].encode())
            ]
    assert all(
        "insert" not in text.lower() and "delete" not in text.lower()
        for text in FUNCTION_CONTRACTS.values()
    )


def test_proof_reader_rejects_tamper_escape_and_symlink(tmp_path):
    path = tmp_path / "proof.json"
    path.write_bytes(b"{}")
    descriptor = {"path": "proof.json", "sha256": sha256_bytes(b"{}"), "bytes": 2}
    assert read_proof(tmp_path, descriptor) == b"{}"
    path.write_bytes(b"[]")
    with pytest.raises(ValueError):
        read_proof(tmp_path, descriptor)
    with pytest.raises(ValueError):
        read_proof(tmp_path, {**descriptor, "path": "../proof.json"})
    path.unlink()
    path.symlink_to(tmp_path.parent / "other.json")
    with pytest.raises(ValueError):
        read_proof(tmp_path, descriptor)


def test_completed_fixture_with_skipped_update_does_not_authorize_pilot():
    path = (
        Path(__file__).parents[1]
        / "reports/prototype/product_r2/disposable_fixture_gpu_observation_v5.json"
    )
    record = json.loads(path.read_bytes())
    record["token_counts"]["skipped_updates"] = 1
    payload = json.dumps(record).encode()
    descriptor = {
        "sha256": sha256_bytes(payload),
        "fixture_plan_sha256": record["artifact_integrity"]["plan_sha256"],
    }
    with pytest.raises(ValueError):
        validate_implementation_evidence(payload, descriptor)
