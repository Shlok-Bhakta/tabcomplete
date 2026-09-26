"""Deterministic public-Git candidate and split-isolation checks."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tinycomplete.one_line.contract import EditState
from tinycomplete.one_line.data import (
    PublicGitSource,
    connected_groups,
    mine_public_git_pair,
    mine_public_git_pair_enriched,
    replay_replacement_history,
    validate_splits,
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
