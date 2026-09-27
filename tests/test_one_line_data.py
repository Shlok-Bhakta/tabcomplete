"""Deterministic public-Git candidate and split-isolation checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditState
from tinycomplete.one_line.data import (
    AUTHORING_FOCUS,
    PublicGitSource,
    connected_groups,
    discover_r2_public_sources,
    mine_public_git_pair,
    mine_public_git_pair_enriched,
    parse_author_response,
    replay_replacement_history,
    validate_splits,
    verify_live_public_source,
    verify_source,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _source(repo: Path, parent: str, child: str) -> PublicGitSource:
    license_hash = hashlib.sha256(
        subprocess.check_output(["git", "-C", str(repo), "show", f"{child}:LICENSE"])
    ).hexdigest()
    return PublicGitSource(
        checkout=repo,
        repo_id="public/example",
        public_url="https://example.invalid/public/example",
        parent_rev=parent,
        child_rev=child,
        license_spdx="MIT",
        license_path="LICENSE",
        license_sha256=license_hash,
    )


def test_mine_exact_single_line_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "LICENSE").write_text("MIT sample\n")
    (repo / "main.py").write_text("x = 1\nprint(x)\n")
    parent = _commit(repo, "start")
    (repo / "main.py").write_text("x = 2\nprint(x)\n")
    child = _commit(repo, "change one line")
    source = _source(repo, parent, child)
    assert verify_source(source)["child_rev"] == child
    rows = mine_public_git_pair(source)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_type"] == "git_observed_atomic"
    assert row["action"] == {"kind": "replace_line", "text": "x = 2"}
    assert row["after_source"] == "x = 2\nprint(x)\n"
    assert row["validation"]["equals_committed_child"] is True
    assert row["validation"]["inferability_reviewed"] is False


def test_multi_change_is_reconstructed_not_observed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "LICENSE").write_text("MIT sample\n")
    (repo / "main.py").write_text("a = 1\nkeep = 0\nb = 1\n")
    parent = _commit(repo, "start")
    (repo / "main.py").write_text("a = 2\nkeep = 0\nb = 2\n")
    child = _commit(repo, "change two lines")
    rows = mine_public_git_pair(_source(repo, parent, child))
    assert len(rows) == 2
    assert all(row["source_type"] == "git_reconstructed_atomic" for row in rows)
    assert all(row["validation"]["equals_committed_child"] is False for row in rows)
    assert all(row["provenance"]["human_edit_order_observed"] is False for row in rows)
    enriched = mine_public_git_pair_enriched(_source(repo, parent, child))
    assert {row["source_type"] for row in enriched} == {
        "git_reconstructed_order",
        "synthetic_terminal_keep",
    }
    history_row = next(row for row in enriched if row["source_type"] == "git_reconstructed_order")
    state = EditState.from_mapping(history_row["state"])
    assert len(state.history) == 1
    assert (
        replay_replacement_history(
            "a = 1\nkeep = 0\nb = 1\n",
            state.history,
            file_id=state.file_id,
            filetype=state.filetype,
        )
        == state.source
    )
    keep = next(row for row in enriched if row["source_type"] == "synthetic_terminal_keep")
    assert keep["action"] == {"kind": "keep", "text": None}
    assert keep["state"]["source"] == "a = 2\nkeep = 0\nb = 2\n"
    assert keep["provenance"]["not_observed_no_edit"] is True


def test_history_before_insertion_keeps_exact_row_coordinates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "LICENSE").write_text("MIT sample\n")
    (repo / "main.py").write_text("a = 1\nkeep = 0\nend = 1\n")
    parent = _commit(repo, "start")
    (repo / "main.py").write_text("a = 2\nkeep = 0\nnew = 1\nend = 1\n")
    child = _commit(repo, "replace then insert")
    rows = mine_public_git_pair_enriched(_source(repo, parent, child))
    insertion = next(row for row in rows if row["action"]["kind"] == "insert_before")
    state = EditState.from_mapping(insertion["state"])
    assert state.target_row == 2
    assert state.source == "a = 2\nkeep = 0\nend = 1\n"
    assert insertion["after_source"] == "a = 2\nkeep = 0\nnew = 1\nend = 1\n"
    assert len(state.history) == 1


def test_split_validator_connects_repo_alias_and_rejects_duplicates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "LICENSE").write_text("MIT sample\n")
    (repo / "main.py").write_text("a = 1\n")
    parent = _commit(repo, "start")
    (repo / "main.py").write_text("a = 2\n")
    child = _commit(repo, "change")
    first = mine_public_git_pair(_source(repo, parent, child))[0]
    second = {
        **first,
        "id": "other-id",
        "source_repo": "fork/example",
        "source_aliases": ["PUBLIC/EXAMPLE"],
    }
    second["state"] = {**first["state"], "file_id": "fork/example/main.py"}
    first["split"] = "train"
    second["split"] = "test_new_repo"
    assert len(connected_groups([first, second])) == 1
    with pytest.raises(ValueError, match="connected source group"):
        validate_splits([first, second])
    second["split"] = "train"
    assert validate_splits([first, second])["groups"] == 1
    duplicate_state = {**first, "id": "same-input-different-id"}
    with pytest.raises(ValueError, match="duplicate model input state"):
        validate_splits([first, duplicate_state])


def test_bad_license_hash_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "LICENSE").write_text("MIT sample\n")
    (repo / "main.py").write_text("a = 1\n")
    parent = _commit(repo, "start")
    (repo / "main.py").write_text("a = 2\n")
    child = _commit(repo, "change")
    source = PublicGitSource(
        checkout=repo,
        repo_id="public/example",
        public_url="https://example.invalid/public/example",
        parent_rev=parent,
        child_rev=child,
        license_spdx="MIT",
        license_path="LICENSE",
        license_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="license blob hash mismatch"):
        mine_public_git_pair(source)


def test_public_authoring_source_has_pinned_license_and_no_answer_fields(tmp_path: Path) -> None:
    assert len(AUTHORING_FOCUS) == len(set(AUTHORING_FOCUS)) == 25
    pool = tmp_path / "python.jsonl"
    lead = {
        "repository": "public/example",
        "repository_aliases": ["public/example"],
        "path": "src/main.py",
        "licenses": ["MIT"],
        "parse_status": "pass",
        "content": "def old():\n    return 1\n" * 30,
        "content_sha256": "0" * 64,
    }
    pool.write_text(json.dumps(lead) + "\n")
    discovered = discover_r2_public_sources(pool, "python")
    assert len(discovered) == 1
    source = "def compute(value):\n    return value + 1\n" + "# public source\n" * 40
    blobs = {
        "src/main.py": source.encode(),
        "LICENSE": b"MIT License\nPermission is hereby granted, free of charge\n",
    }
    row = verify_live_public_source(
        discovered[0],
        resolve_head=lambda repo: "a" * 40,
        fetch_blob=lambda repo, revision, path: blobs.get(path),
    )
    assert row is not None
    assert row["student_state_seed"]["source"] == source
    assert row["authoring_metadata"]["source_license"] == "MIT"
    assert row["authoring_metadata"]["source_revision"] == "a" * 40
    assert row["authoring_metadata"]["file_license_scope_unverified_without_notice"] is True
    assert "action" not in row and "authoring_focus" not in row["student_state_seed"]
    blobs["src/main.py"] = ("# SPDX-License-Identifier: GPL-3.0\n" + source).encode()
    assert (
        verify_live_public_source(
            discovered[0],
            resolve_head=lambda repo: "a" * 40,
            fetch_blob=lambda repo, revision, path: blobs.get(path),
        )
        is None
    )


class _ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(text.encode()) + ([0] if add_special_tokens else [])


def _authoring_source() -> dict:
    source = "def compute(value):\n    total = value + 1\n    return total\n"
    return {
        "id": "public-source/fixed",
        "student_state_seed": {"file_id": "file_fixed.py", "filetype": "python", "source": source},
        "authoring_metadata": {
            "source_repo": "public/example",
            "source_aliases": ["public/example"],
            "source_revision": "a" * 40,
            "source_license": "MIT",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "license_sha256": "b" * 64,
            "authoring_focus": "call site consistency",
        },
    }


def _valid_author_response() -> dict:
    return {
        "prior_edit": {
            "row": 1,
            "old_text": "    total = value + 1",
            "new_text": "    result = value + 1",
        },
        "target_row": 2,
        "action": {"kind": "R", "text": "    return result"},
        "intent_evidence": "The assignment was renamed, so the return should use its new name.",
        "objective": {"kind": "structural", "description": "The return uses result."},
    }


def test_author_response_replays_one_edit_without_leaking_metadata() -> None:
    source = _authoring_source()
    row = parse_author_response(json.dumps(_valid_author_response()), source, _ByteTokenizer())
    assert row["state"]["history"] == (
        {"row": 1, "old_text": "    total = value + 1", "new_text": "    result = value + 1"},
    )
    assert row["after_source"] == "def compute(value):\n    result = value + 1\n    return result\n"
    assert row["validation"]["history_visible_in_prompt"] is True
    assert row["validation"]["accepted_training"] is False
    from tinycomplete.one_line.context import serialize_state

    prompt = serialize_state(EditState.from_mapping(row["state"]), _ByteTokenizer())
    assert "public/example" not in prompt
    assert "call site consistency" not in prompt
    assert "The return uses result" not in prompt


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"prior_edit": {"row": 1, "old_text": "wrong", "new_text": "new"}}, "history"),
        ({"target_row": 99}, "target row"),
        ({"action": {"kind": "R", "text": "    return total"}}, "unchanged"),
        ({"action": {"kind": "R", "text": "    return x\n"}}, "line"),
        ({"action": {"kind": "R", "text": "x" * 70}}, "64 q25"),
        ({"action": {"kind": "X", "text": None}}, "canonical"),
    ],
)
def test_author_response_rejects_bad_structure(change: dict, message: str) -> None:
    raw = _valid_author_response()
    raw.update(change)
    with pytest.raises(ValueError, match=message):
        parse_author_response(json.dumps(raw), _authoring_source(), _ByteTokenizer())


def test_author_response_rejects_duplicate_keys_and_extra_prose() -> None:
    raw = json.dumps(_valid_author_response())
    with pytest.raises(ValueError, match="strict JSON"):
        parse_author_response(raw + " explanation", _authoring_source(), _ByteTokenizer())
    duplicate = raw.replace('"target_row": 2', '"target_row": 2, "target_row": 2')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_author_response(duplicate, _authoring_source(), _ByteTokenizer())


def test_author_response_keep_is_candidate_not_verified_label() -> None:
    raw = _valid_author_response()
    raw["action"] = {"kind": "N", "text": None}
    raw["objective"] = {"kind": "no_edit", "description": "The target is already correct."}
    row = parse_author_response(json.dumps(raw), _authoring_source(), _ByteTokenizer())
    assert row["action"] == {"kind": "keep", "text": None}
    assert row["after_source"] == row["state"]["source"]
    assert row["validation"]["objective_verified"] is False


def test_author_response_rejects_tampered_public_source() -> None:
    source = _authoring_source()
    source["student_state_seed"]["source"] += "# later unverified edit\n"
    with pytest.raises(ValueError, match="pinned public source hash mismatch"):
        parse_author_response(json.dumps(_valid_author_response()), source, _ByteTokenizer())
