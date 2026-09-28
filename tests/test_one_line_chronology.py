"""Local-only first-parent Git chronology mining tests."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tinycomplete.one_line.chronology import (
    MineLimits,
    PinnedChronologySource,
    mine_cross_commit_candidates,
)
from tinycomplete.one_line.contract import EditAction, EditState, apply_action


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


def _source(
    repo: Path, base: str, tip: str, *, license_sha256: str | None = None
) -> PinnedChronologySource:
    if license_sha256 is None:
        license_sha256 = hashlib.sha256((repo / "LICENSE").read_bytes()).hexdigest()
    return PinnedChronologySource(
        checkout=repo,
        repo_id="public/example",
        public_url="https://example.invalid/public/example",
        base_rev=base,
        tip_rev=tip,
        file_path="main.py",
        license_spdx="MIT",
        license_path="LICENSE",
        license_sha256=license_sha256,
    )


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "LICENSE").write_text("MIT sample\n")
    (repo / "main.py").write_text("count = 1\nprint(count)\n")
    return repo, _commit(repo, "base")


def test_cross_commit_pair_uses_real_chronology_and_exact_blobs(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "main.py").write_text("count = 2\nprint(count)\n")
    prior = _commit(repo, "earlier replacement")
    (repo / "other.txt").write_text("unrelated\n")
    middle = _commit(repo, "other file")
    (repo / "main.py").write_text("count = 2\nprint(count + 1)\n")
    target = _commit(repo, "later replacement")
    rows = mine_cross_commit_candidates(_source(repo, base, target))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "unreviewed"
    assert row["validation"]["accepted"] is False
    assert row["validation"]["inferability_reviewed"] is False
    assert row["source_type"] == "git_cross_commit_chronology"
    assert row["provenance"]["prior_commit"] == prior
    assert row["provenance"]["target_parent_commit"] == middle
    assert row["provenance"]["target_commit"] == target
    assert row["provenance"]["target_after_git_blob"] == _git(
        repo, "rev-parse", f"{target}:main.py"
    )
    assert row["provenance"]["commit_order_observed"] is True
    assert row["provenance"]["editor_edit_order_observed"] is False
    state = EditState.from_mapping(row["state"])
    assert len(state.history) == 1
    assert state.history[0].old_text == "count = 1"
    assert state.history[0].new_text == "count = 2"
    assert apply_action(state, EditAction(**row["action"])) == row["after_source"]


@pytest.mark.parametrize(
    ("target_source", "expected_kind"),
    [
        ("count = 2\nextra = 1\nprint(count)\n", "insert_before"),
        ("count = 2\n", "delete_line"),
    ],
)
def test_target_insertion_and_deletion(
    tmp_path: Path, target_source: str, expected_kind: str
) -> None:
    repo, base = _repo(tmp_path)
    (repo / "main.py").write_text("count = 2\nprint(count)\n")
    _commit(repo, "prior")
    (repo / "main.py").write_text(target_source)
    tip = _commit(repo, "target")
    rows = mine_cross_commit_candidates(_source(repo, base, tip))
    assert len(rows) == 1
    assert rows[0]["action"]["kind"] == expected_kind
    assert rows[0]["after_source"] == target_source


def test_multi_hunk_commit_cannot_become_synthetic_prior_order(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "main.py").write_text("count = 2\nprint(count + 1)\n")
    _commit(repo, "two changed lines")
    (repo / "main.py").write_text("count = 3\nprint(count + 1)\n")
    tip = _commit(repo, "one changed line")
    assert mine_cross_commit_candidates(_source(repo, base, tip)) == []


def test_skipped_intervening_file_change_breaks_history(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "main.py").write_text("count = 2\nprint(count)\n")
    _commit(repo, "prior")
    (repo / "main.py").write_text("count = 2\nprint(count + 1)\nextra = 1\n")
    _commit(repo, "unrepresentable change")
    (repo / "main.py").write_text("count = 2\nprint(count)\n")
    _commit(repo, "restore")
    (repo / "main.py").write_text("count = 3\nprint(count)\n")
    tip = _commit(repo, "target")
    assert mine_cross_commit_candidates(_source(repo, base, tip)) == []


def test_first_parent_and_license_pin_are_required(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "main.py").write_text("count = 2\nprint(count)\n")
    _commit(repo, "prior")
    (repo / "main.py").write_text("count = 3\nprint(count)\n")
    tip = _commit(repo, "target")
    with pytest.raises(ValueError, match="bounded first-parent"):
        mine_cross_commit_candidates(_source(repo, base, tip), limits=MineLimits(max_commits=1))
    with pytest.raises(ValueError, match="license blob mismatch"):
        mine_cross_commit_candidates(_source(repo, base, tip, license_sha256="0" * 64))


def test_read_cap_stops_cleanly(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "main.py").write_text("count = 2\nprint(count)\n")
    _commit(repo, "prior")
    (repo / "main.py").write_text("count = 3\nprint(count)\n")
    tip = _commit(repo, "target")
    with pytest.raises(RuntimeError, match="read-byte cap"):
        mine_cross_commit_candidates(
            _source(repo, base, tip),
            limits=MineLimits(max_blob_bytes=100, max_total_read_bytes=150),
        )


def test_conflicting_file_spdx_is_not_mined(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    (repo / "main.py").write_text("# SPDX-License-Identifier: GPL-3.0\ncount = 1\n")
    base = _commit(repo, "file notice")
    (repo / "main.py").write_text("# SPDX-License-Identifier: GPL-3.0\ncount = 2\n")
    _commit(repo, "prior")
    (repo / "main.py").write_text("# SPDX-License-Identifier: GPL-3.0\ncount = 3\n")
    tip = _commit(repo, "target")
    assert mine_cross_commit_candidates(_source(repo, base, tip)) == []


def test_unsafe_paths_and_unpinned_revision_rejected(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    with pytest.raises(ValueError, match="unsafe repository path"):
        PinnedChronologySource(
            checkout=repo,
            repo_id="public/example",
            public_url="https://example.invalid/public/example",
            base_rev=base,
            tip_rev=base,
            file_path="../private.py",
            license_spdx="MIT",
            license_path="LICENSE",
            license_sha256=hashlib.sha256((repo / "LICENSE").read_bytes()).hexdigest(),
        )
