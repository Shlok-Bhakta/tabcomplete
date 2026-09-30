from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_public_source_oracle_seeds.py"
SPEC = importlib.util.spec_from_file_location("prepare_public_source_oracle_seeds", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ALLOWED_ACTIONS = MODULE.ALLOWED_ACTIONS
PARENT_IDENTITIES = MODULE.PARENT_IDENTITIES
AuditError = MODULE.AuditError
audit_parent_license_scope = MODULE.audit_parent_license_scope
path_directories = MODULE.path_directories
validate_spec = MODULE.validate_spec
git_blob_sha1 = MODULE.git_blob_sha1


def seed_spec() -> dict[str, Any]:
    rows = []
    for index, candidate_id in enumerate(PARENT_IDENTITIES):
        rows.append(
            {
                "seed_id": f"seed-{index:02d}",
                "candidate_id": candidate_id,
                "split": "train",
                "source_type": "synthetic_public_source_task",
                "student_request": f"Check the explicitly stated synthetic behavior {index}.",
                "expected_action_kind": "keep" if index in (4, 5) else "replace_line",
                "objective_contract": ["before/after behavior is evaluated independently"],
                "control_contract": ["keep and one behaviorally wrong action are tested"],
                "history_policy": "empty_synthetic_history",
                "execution_scope": "isolated test harness",
                "whole_project_claim": False,
                "human_chronology_observed": False,
            }
        )
    return {"schema": "public-source-oracle-seed-spec-v1", "revision": 1, "seeds": rows}


class FixtureApi:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records
        self.bytes_received = 0
        self.endpoints: list[str] = []

    def json(self, repository: str, endpoint: str) -> dict[str, Any]:
        self.endpoints.append(f"{repository}/{endpoint}")
        try:
            result = self.records[endpoint]
            self.bytes_received += len(json.dumps(result).encode())
            return result
        except KeyError as error:
            raise AssertionError(f"unexpected API endpoint: {endpoint}") from error


def _encoded_blob(data: bytes) -> dict[str, Any]:
    return {"encoding": "base64", "content": base64.b64encode(data).decode("ascii")}


def _api_records(
    *,
    root_license: bytes,
    root_blob: str = "a" * 40,
    nested_license: bytes | None = None,
    nested_blob: str = "b" * 40,
    ambiguous: bool = False,
    truncated: bool = False,
    notice: bool = False,
    reuse_dep5: bool = False,
    source_bytes: bytes = b"package example\n",
) -> dict[str, dict[str, Any]]:
    root_entries: list[dict[str, str]] = [
        {"path": "LICENSE", "type": "blob", "sha": root_blob},
        {"path": "src", "type": "tree", "sha": "src-tree"},
    ]
    src_entries: list[dict[str, str]] = [{"path": "pkg", "type": "tree", "sha": "pkg-tree"}]
    pkg_entries: list[dict[str, str]] = [
        {"path": "file.go", "type": "blob", "sha": git_blob_sha1(source_bytes)}
    ]
    blobs = {f"git/blobs/{root_blob}": _encoded_blob(root_license)}
    if nested_license is not None:
        src_entries.append({"path": "COPYING", "type": "blob", "sha": nested_blob})
        blobs[f"git/blobs/{nested_blob}"] = _encoded_blob(nested_license)
    if ambiguous:
        src_entries.append({"path": "COPYING", "type": "blob", "sha": nested_blob})
        src_entries.append({"path": "LICENSE", "type": "blob", "sha": "c" * 40})
        blobs[f"git/blobs/{nested_blob}"] = _encoded_blob(b"SPDX-License-Identifier: MIT\n")
        blobs[f"git/blobs/{'c' * 40}"] = _encoded_blob(b"SPDX-License-Identifier: MIT\n")
    extra_records: dict[str, dict[str, Any]] = {}
    if notice:
        root_entries.append({"path": "NOTICE", "type": "blob", "sha": "d" * 40})
        blobs[f"git/blobs/{'d' * 40}"] = _encoded_blob(b"Additional terms\n")
    if reuse_dep5:
        root_entries.append({"path": ".reuse", "type": "tree", "sha": "reuse-tree"})
        extra_records["git/trees/reuse-tree"] = {
            "truncated": False,
            "tree": [{"path": "dep5", "type": "blob", "sha": "e" * 40}],
        }
    return {
        "git/commits/parent-sha": {"sha": "parent-sha", "tree": {"sha": "root-tree"}},
        "git/trees/root-tree": {"truncated": truncated, "tree": root_entries},
        "git/trees/src-tree": {"truncated": False, "tree": src_entries},
        "git/trees/pkg-tree": {"truncated": False, "tree": pkg_entries},
        **blobs,
        **extra_records,
    }


def audit(
    api: FixtureApi,
    *,
    source_bytes: bytes = b"package example\n",
    root_license: bytes = b"MIT License\n",
    root_blob: str = "a" * 40,
    nested_license: bytes | None = None,
    ambiguous: bool = False,
    truncated: bool = False,
    notice: bool = False,
    reuse_dep5: bool = False,
) -> dict[str, Any]:
    return audit_parent_license_scope(
        api,
        repository="example/project",
        parent_commit="parent-sha",
        expected_tree_sha="root-tree",
        source_path="src/pkg/file.go",
        root_license_path="LICENSE",
        root_license_git_blob_sha=root_blob,
        root_license_sha256=hashlib.sha256(root_license).hexdigest(),
        root_spdx=["mit"],
        parent_source_bytes=source_bytes,
        private_license_dir=None,
    )


def test_spec_freezes_eight_train_groups_without_source_or_gold_bodies() -> None:
    spec = seed_spec()
    validate_spec(spec)
    assert len(spec["seeds"]) == 8
    groups = {PARENT_IDENTITIES[row["candidate_id"]]["source_group_id"] for row in spec["seeds"]}
    assert len(groups) == 8
    assert all(row["expected_action_kind"] in ALLOWED_ACTIONS for row in spec["seeds"])
    assert all(row["whole_project_claim"] is False for row in spec["seeds"])
    assert all(row["human_chronology_observed"] is False for row in spec["seeds"])


@pytest.mark.parametrize(
    ("mutation", "status"),
    [
        ("development", "nontraining_seed_forbidden"),
        ("wrong_source_type", "source_type_invalid"),
        ("chronology", "synthetic_chronology_mislabelled"),
        ("whole_project", "whole_project_claim_forbidden"),
        ("no_controls", "semantic_control_contract_missing"),
        ("source_body", "spec_contains_source_or_gold_body"),
    ],
)
def test_spec_rejects_split_or_provenance_laundering(mutation: str, status: str) -> None:
    spec = seed_spec()
    row = spec["seeds"][0]
    if mutation == "development":
        row["split"] = "development"
    elif mutation == "wrong_source_type":
        row["source_type"] = "observed_human_edit"
    elif mutation == "chronology":
        row["human_chronology_observed"] = True
    elif mutation == "whole_project":
        row["whole_project_claim"] = True
    elif mutation == "no_controls":
        row["control_contract"] = []
    else:
        row["gold_action"] = {"kind": "replace_line", "text": "hidden"}
    with pytest.raises(AuditError, match=status):
        validate_spec(spec)


def test_path_directories_are_exact_ancestors_only() -> None:
    assert path_directories("src/pkg/file.go") == ("", "src", "src/pkg")
    assert path_directories("file.go") == ("",)
    with pytest.raises(AuditError, match="unsafe_source_file_path"):
        path_directories("src/../other.go")


def test_root_scope_uses_exact_parent_tree_and_keeps_root_spdx_separate() -> None:
    license_bytes = b"MIT License\n"
    records = _api_records(root_license=license_bytes)
    api = FixtureApi(records)
    result = audit(api, root_license=license_bytes)
    assert result["status"] == "verified_path_scope"
    assert result["root_license"]["root_spdx"] == ["mit"]
    assert result["path_scope"]["spdx"] == ["mit"]
    assert result["path_scope"]["scope"] == "root"
    assert result["network_bytes"] > 0
    assert all("child" not in endpoint for endpoint in api.endpoints)
    assert any("git/commits/parent-sha" in endpoint for endpoint in api.endpoints)


def test_nearer_different_license_is_reported_and_skipped() -> None:
    root = b"MIT License\n"
    nested = b"SPDX-License-Identifier: Apache-2.0\n"
    api = FixtureApi(_api_records(root_license=root, nested_license=nested))
    result = audit(api, root_license=root, nested_license=nested)
    assert result["root_license"]["root_spdx"] == ["mit"]
    assert result["path_scope"]["spdx"] == ["apache-2.0"]
    assert result["path_scope"]["scope"] == "closest_ancestor"
    assert result["status"] == "skip_path_scope_differs_from_root_spdx"


def test_ambiguous_closest_license_fails_closed() -> None:
    root = b"MIT License\n"
    api = FixtureApi(_api_records(root_license=root, ambiguous=True))
    with pytest.raises(AuditError, match="ambiguous_closest_license"):
        audit(api, root_license=root, ambiguous=True)


def test_truncated_tree_fails_closed() -> None:
    root = b"MIT License\n"
    api = FixtureApi(_api_records(root_license=root, truncated=True))
    with pytest.raises(AuditError, match="parent_tree_truncated"):
        audit(api, root_license=root, truncated=True)


def test_root_blob_mismatch_is_not_hidden_by_spdx_metadata() -> None:
    root = b"MIT License\n"
    api = FixtureApi(_api_records(root_license=root, root_blob="a" * 40))
    with pytest.raises(AuditError, match="root_license_tree_identity_mismatch"):
        audit(api, root_license=root, root_blob="b" * 40)


def test_source_header_conflicting_with_exact_path_license_is_skipped() -> None:
    root = b"MIT License\n"
    source = b"// SPDX-License-Identifier: Apache-2.0\npackage example\n"
    api = FixtureApi(_api_records(root_license=root, source_bytes=source))
    result = audit(api, root_license=root, source_bytes=source)
    assert result["source_header_spdx"] == ["apache-2.0"]
    assert result["root_license"]["root_spdx"] == ["mit"]
    assert result["status"] == "skip_source_header_scope_mismatch"


def test_compound_spdx_expression_never_falls_back_to_root_label() -> None:
    root = b"SPDX-License-Identifier: MIT OR Apache-2.0\n"
    api = FixtureApi(_api_records(root_license=root))
    result = audit(api, root_license=root)
    assert result["root_license"]["root_spdx"] == ["mit"]
    assert result["path_scope"]["spdx_ambiguous"] is True
    assert result["status"] == "skip_ambiguous_spdx_expression"


def test_notice_and_reuse_dep5_are_recorded_and_block_approval() -> None:
    root = b"MIT License\n"
    api = FixtureApi(_api_records(root_license=root, notice=True, reuse_dep5=True))
    result = audit(api, root_license=root)
    assert result["additional_license_references"] == [{"path": "NOTICE", "git_blob_sha": "d" * 40}]
    assert result["reuse_dep5_references"] == [{"path": ".reuse/dep5", "git_blob_sha": "e" * 40}]
    assert result["status"] == "skip_additional_license_notice_requires_review"


def test_ancestor_copyright_and_reuse_dep5_references_are_preserved() -> None:
    root = b"MIT License\n"
    records = _api_records(root_license=root)
    records["git/trees/src-tree"]["tree"].append(
        {"path": "COPYRIGHT", "type": "blob", "sha": "f" * 40}
    )
    records["git/trees/src-tree"]["tree"].append(
        {"path": ".reuse", "type": "tree", "sha": "reuse-src-tree"}
    )
    records["git/trees/reuse-src-tree"] = {
        "truncated": False,
        "tree": [{"path": "dep5", "type": "blob", "sha": "e" * 40}],
    }
    result = audit(FixtureApi(records), root_license=root)
    assert result["additional_license_references"] == [
        {"path": "src/COPYRIGHT", "git_blob_sha": "f" * 40}
    ]
    assert result["reuse_dep5_references"] == [
        {"path": "src/.reuse/dep5", "git_blob_sha": "e" * 40}
    ]
    assert result["status"] == "skip_additional_license_notice_requires_review"


def test_exact_parent_source_blob_must_match_the_local_snapshot() -> None:
    root = b"MIT License\n"
    source = b"package other\n"
    api = FixtureApi(_api_records(root_license=root, source_bytes=b"package example\n"))
    with pytest.raises(AuditError, match="parent_source_tree_identity_mismatch"):
        audit(api, root_license=root, source_bytes=source)
