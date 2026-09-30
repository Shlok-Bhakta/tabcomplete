from __future__ import annotations

import base64
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_commit_sequence_sources.py"
SPEC = importlib.util.spec_from_file_location("commit_sequence_source_verification", SCRIPT)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


PARENT = "1" * 40
CHILD = "2" * 40
LICENSE_TEXT = b"SPDX-License-Identifier: MIT\nPermission is hereby granted.\n"
BEFORE = b"def total(items):\n    return sum(items)\n"
AFTER = b"def total(items):\n    return sum(items) + 1\n"
LICENSE_SHA = "a" * 40


class FakeGitHub:
    def __init__(self, *, parent_bytes: bytes = BEFORE, child_bytes: bytes = AFTER) -> None:
        self.parent_bytes = parent_bytes
        self.child_bytes = child_bytes

    def raw(self, repo: str, path: str, revision: str) -> bytes:
        assert repo == "example/project"
        assert path == "src/math.py"
        if revision == PARENT:
            return self.parent_bytes
        assert revision == CHILD
        return self.child_bytes

    def json(self, repo: str, endpoint: str) -> dict[str, Any]:
        assert repo == "example/project"
        if endpoint == f"git/commits/{CHILD}":
            return {"parents": [{"sha": PARENT}], "tree": {"sha": "child-root"}}
        if endpoint == f"git/commits/{PARENT}":
            return {"parents": [{"sha": "0" * 40}], "tree": {"sha": "parent-root"}}
        if endpoint in (f"license?ref={PARENT}", f"license?ref={CHILD}"):
            return {
                "path": "LICENSE",
                "sha": LICENSE_SHA,
                "license": {"spdx_id": "MIT"},
                "encoding": "base64",
                "content": base64.b64encode(LICENSE_TEXT).decode(),
            }
        if endpoint in ("git/trees/parent-root", "git/trees/child-root"):
            return {
                "tree": [
                    {"path": "LICENSE", "type": "blob", "sha": LICENSE_SHA},
                    {"path": "src", "type": "tree", "sha": "src-tree"},
                ]
            }
        if endpoint == "git/trees/src-tree":
            return {"tree": [{"path": "math.py", "type": "blob", "sha": "b" * 40}]}
        if endpoint == f"git/blobs/{LICENSE_SHA}":
            return {
                "encoding": "base64",
                "content": base64.b64encode(LICENSE_TEXT).decode(),
            }
        raise AssertionError(f"unexpected API identity: {endpoint}")


def candidate(**changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": "commit-sequence/test-case",
        "source_group_id": "group-a",
        "split": "train",
        "source_repo": "example/project",
        "source_revision": CHILD,
        "file_path": "src/math.py",
        "source_before_sha256": verifier.sha256_bytes(BEFORE),
        "committed_child_file_sha256": verifier.sha256_bytes(AFTER),
        "source_license_claim": "mit",
    }
    row.update(changes)
    return row


def test_exact_pair_and_license_are_qualified_only_for_review(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(verifier, "OUTPUT_DIR", tmp_path / "private")
    result = verifier.verify_candidate(candidate(), FakeGitHub())  # type: ignore[arg-type]
    assert result["status"] == "source_and_license_verified_for_human_review"
    assert result["parent_commit"] == PARENT
    assert result["source_pair"]["parent_sha256"] == verifier.sha256_bytes(BEFORE)
    assert result["source_pair"]["child_sha256"] == verifier.sha256_bytes(AFTER)
    assert result["accepted_training"] is False
    assert result["chronology_observed"] is False
    assert result["inferability_reviewed"] is False
    for private_path in (tmp_path / "private").rglob("*"):
        if private_path.is_file():
            assert stat.S_IMODE(private_path.stat().st_mode) == 0o600


def test_source_hash_mismatch_is_skipped_without_persisting_source(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "private"
    monkeypatch.setattr(verifier, "OUTPUT_DIR", private)
    result = verifier.verify_candidate(candidate(source_before_sha256="f" * 64), FakeGitHub())  # type: ignore[arg-type]
    assert result["status"] == "parent_source_hash_mismatch"
    assert not (private / "source_pairs").exists()


def test_merge_commit_is_skipped() -> None:
    class Merge(FakeGitHub):
        def json(self, repo: str, endpoint: str) -> dict[str, Any]:
            if endpoint == f"git/commits/{CHILD}":
                return {
                    "parents": [{"sha": PARENT}, {"sha": "3" * 40}],
                    "tree": {"sha": "child-root"},
                }
            return super().json(repo, endpoint)

    result = verifier.verify_candidate(candidate(), Merge())  # type: ignore[arg-type]
    assert result["status"] == "non_single_parent_commit"


def test_license_expression_is_not_collapsed_to_first_identifier() -> None:
    assert verifier._spdx_from_bytes(b"# SPDX-License-Identifier: MIT\n") == {"mit"}
    assert verifier._spdx_from_bytes(b"# SPDX-License-Identifier: MIT OR Apache-2.0\n") == set()
    assert not verifier._license_claim_matches(
        "mit", verifier._spdx_from_bytes(b"# SPDX-License-Identifier: MIT OR Apache-2.0\n")
    )


def test_bound_client_stops_at_transfer_cap(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "gh"
    executable.write_text("#!/bin/sh\nprintf 'abcdefghij'\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    client = verifier.GitHubClient(byte_cap=3, deadline=verifier.time.monotonic() + 5)
    with pytest.raises(verifier.VerificationError, match="transfer_byte_cap"):
        client.request("repos/example/project/test")
    assert client.bytes_received <= 3


def test_plan_contains_queue_commitment_without_source_bodies(
    tmp_path: Path, monkeypatch
) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / "review_index.jsonl"
    rows = [
        {
            "candidate_id": f"candidate-{index:03d}",
            "source_group_id": f"group-{index:03d}",
            "split": "train" if index < 96 else "development",
            "source_repo": f"example/project-{index:03d}",
            "source_revision": CHILD,
            "file_path": "src/math.py",
            "source_before_sha256": "a" * 64,
            "committed_child_file_sha256": "b" * 64,
            "source_license_claim": "mit",
        }
        for index in range(128)
    ]
    queue_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    selector_plan_path = tmp_path / "selector-plan.json"
    selector_plan_identity = verifier.canonical_sha({"schema": "synthetic-selector-plan-v4"})
    selector_plan_path.write_text(
        json.dumps(
            {
                "schema": "synthetic-selector-plan-v4",
                "plan_identity_sha256": selector_plan_identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (queue_dir / "manifest.json").write_text(
        json.dumps(
            {
                "private_index_sha256": verifier.sha256_file(queue_path),
                "plan_file_sha256": verifier.sha256_file(selector_plan_path),
                "plan_identity_sha256": selector_plan_identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prior_plan_path = tmp_path / "synthetic-plan_v1.json"
    prior_plan_path.write_text('{"schema":"synthetic-prior-plan-v1"}\n', encoding="utf-8")

    monkeypatch.setattr(verifier, "QUEUE_PATH", queue_path)
    monkeypatch.setattr(verifier, "QUEUE_MANIFEST", queue_dir / "manifest.json")
    monkeypatch.setattr(verifier, "QUEUE_PLAN", selector_plan_path)
    monkeypatch.setattr(verifier, "PRIOR_PLAN_PATH", prior_plan_path)
    monkeypatch.setattr(verifier, "OUTPUT_DIR", tmp_path / "private-output")
    gh_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_gh_version(*args: Any, **kwargs: Any) -> SimpleNamespace:
        gh_calls.append((args, kwargs))
        return SimpleNamespace(stdout=b"gh version synthetic\n")

    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        fake_gh_version,
    )

    plan = verifier.make_plan()
    serialized = str(plan)
    assert len(gh_calls) == 1
    assert gh_calls[0][0] == (["gh", "--version"],)
    assert plan["selected_row_count"] == 128
    assert plan["selected_split_counts"] == {"train": 96, "development": 32}
    assert plan["revision"] == 2
    assert plan["supersedes"]["plan_path"].endswith("_v1.json")
    assert plan["caps"]["response_body_bytes"] == 20 * 1024 * 1024
    assert "selected_rows_commitment_sha256" in plan
    assert "source_pair" not in serialized
    assert "old_contents" not in serialized
