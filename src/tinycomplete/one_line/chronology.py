"""Mine review candidates from observed Git commit chronology in a local checkout.

This module never fetches a repository. Commit order is real; editor edit order
and intent are unknown. Every output remains unreviewed.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .contract import ActionKind, EditAction, EditState, Filetype, RecentEdit, apply_action

_HEX = re.compile(r"[0-9a-f]{40,64}\Z")
_LANGUAGES: dict[str, Filetype] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
}
_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"})
_SPDX_NOTICE = re.compile(r"SPDX-License-Identifier:\s*([^\r\n*]+)", re.I)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_notice(text: str, expected: str) -> str | None:
    notices = [value.strip() for value in _SPDX_NOTICE.findall(text[:2048])]
    if notices and any(value != expected for value in notices):
        raise ValueError("file SPDX notice conflicts with repository license")
    return notices[0] if notices else None


@dataclass(frozen=True)
class PinnedChronologySource:
    checkout: Path
    repo_id: str
    public_url: str
    base_rev: str  # Exclusive first-parent ancestor of tip_rev.
    tip_rev: str  # Inclusive last commit to inspect.
    file_path: str
    license_spdx: str
    license_path: str
    license_sha256: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.checkout.is_dir():
            raise ValueError("local checkout is required")
        if not self.repo_id or not self.public_url.startswith("https://"):
            raise ValueError("public source identity is required")
        if not _HEX.fullmatch(self.base_rev) or not _HEX.fullmatch(self.tip_rev):
            raise ValueError("immutable commit hashes are required")
        if not _HEX.fullmatch(self.license_sha256) or self.license_spdx not in _LICENSES:
            raise ValueError("pinned compatible license is required")
        for path in (self.file_path, self.license_path):
            if not path or Path(path).is_absolute() or ".." in Path(path).parts or "\0" in path:
                raise ValueError("unsafe repository path")
        if self.file_path.endswith((".d.ts", ".pb.go")):
            raise ValueError("generated declaration path is excluded")
        if Path(self.file_path).suffix.lower() not in _LANGUAGES:
            raise ValueError("unsupported source language")


@dataclass(frozen=True)
class MineLimits:
    max_commits: int = 32
    max_blob_bytes: int = 256_000
    max_total_read_bytes: int = 4_000_000
    max_seconds: float = 30.0
    max_candidates: int = 100

    def __post_init__(self) -> None:
        if (
            min(
                self.max_commits,
                self.max_blob_bytes,
                self.max_total_read_bytes,
                self.max_candidates,
            )
            < 1
        ):
            raise ValueError("all count and byte caps must be positive")
        if self.max_seconds <= 0 or self.max_blob_bytes > self.max_total_read_bytes:
            raise ValueError("invalid time or blob cap")


class _LocalGit:
    def __init__(self, checkout: Path, limits: MineLimits) -> None:
        self.checkout = checkout
        self.limits = limits
        self.deadline = time.monotonic() + limits.max_seconds
        self.read_bytes = 0
        self.blobs: dict[str, bytes] = {}

    def run(self, *args: str) -> bytes:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Git mining deadline exceeded")
        try:
            result = subprocess.run(
                ["git", "--no-pager", "-C", str(self.checkout), *args],
                capture_output=True,
                check=False,
                timeout=remaining,
                env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("Git mining deadline exceeded") from None
        if result.returncode:
            raise ValueError(f"local Git command failed: {args[0]}")
        self.read_bytes += len(result.stdout)
        if self.read_bytes > self.limits.max_total_read_bytes:
            raise RuntimeError("Git read-byte cap exceeded")
        return result.stdout

    def blob(self, commit: str, path: str) -> tuple[str, bytes] | None:
        try:
            oid = self.run("rev-parse", "--verify", f"{commit}:{path}").strip().decode("ascii")
        except ValueError:
            return None
        if not _HEX.fullmatch(oid):
            raise ValueError("invalid Git blob identity")
        if oid in self.blobs:
            return oid, self.blobs[oid]
        size = int(self.run("cat-file", "-s", oid).strip())
        if size > self.limits.max_blob_bytes:
            return None
        if self.read_bytes + size > self.limits.max_total_read_bytes:
            raise RuntimeError("Git read-byte cap exceeded")
        data = self.run("cat-file", "blob", oid)
        if len(data) != size:
            raise ValueError("Git blob size changed")
        self.blobs[oid] = data
        return oid, data


def _one_line_action(
    before: str, after: str, file_id: str, filetype: Filetype
) -> tuple[int, EditAction] | None:
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    changes = [
        op
        for op in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    if len(changes) != 1:
        return None
    tag, a0, a1, b0, b1 = changes[0]
    if tag == "replace" and a1 - a0 == b1 - b0 == 1:
        kind, text = "replace_line", new_lines[b0].removesuffix("\n").removesuffix("\r")
    elif tag == "delete" and a1 - a0 == 1:
        kind, text = "delete_line", None
    elif tag == "insert" and b1 - b0 == 1:
        kind, text = "insert_before", new_lines[b0].removesuffix("\n").removesuffix("\r")
    else:
        return None
    try:
        action = EditAction(cast(ActionKind, kind), text)
        state = EditState(file_id, filetype, before, a0, 0)
        if apply_action(state, action) != after:
            return None
    except (UnicodeError, ValueError):
        return None
    return a0, action


def mine_cross_commit_candidates(
    source: PinnedChronologySource, *, limits: MineLimits | None = None
) -> list[dict[str, Any]]:
    """Return exact one-line pairs with a prior replacement in an earlier commit.

    `base_rev` must occur within `max_commits` first-parent steps of `tip_rev`.
    Only adjacent *changes to the named file* are paired. Other commits may
    intervene, but the file blob must remain identical between the two edits.
    No network operation, clone, checkout mutation, or split allocation occurs.
    """
    limits = limits or MineLimits()
    git = _LocalGit(source.checkout, limits)
    for rev in (source.base_rev, source.tip_rev):
        resolved = git.run("rev-parse", "--verify", f"{rev}^{{commit}}").strip().decode()
        if resolved != rev:
            raise ValueError("revision does not resolve to its pinned commit")
    chain = (
        git.run(
            "rev-list", "--first-parent", f"--max-count={limits.max_commits + 1}", source.tip_rev
        )
        .decode()
        .splitlines()
    )
    if source.base_rev not in chain:
        raise ValueError("base is outside the bounded first-parent range")
    commits = list(reversed(chain[: chain.index(source.base_rev) + 1]))
    if len(commits) < 3:
        return []
    license_blob = git.blob(source.tip_rev, source.license_path)
    if license_blob is None or _sha(license_blob[1]) != source.license_sha256:
        raise ValueError("pinned license blob mismatch")
    filetype = _LANGUAGES[Path(source.file_path).suffix.lower()]
    file_id = f"{source.repo_id}/{source.file_path}"
    edits: list[tuple[str, str, str, str, str, str, int, EditAction] | None] = []
    for parent, child in zip(commits, commits[1:], strict=False):
        old_blob = git.blob(parent, source.file_path)
        new_blob = git.blob(child, source.file_path)
        if old_blob is None or new_blob is None:
            edits.append(None)
            continue
        if old_blob[0] == new_blob[0]:
            continue
        # Merge commits often import someone else's file edit; preserve a gap.
        if len(git.run("rev-list", "--parents", "-n", "1", child).decode().split()) != 2:
            edits.append(None)
            continue
        child_license = git.blob(child, source.license_path)
        if child_license is None or _sha(child_license[1]) != source.license_sha256:
            edits.append(None)
            continue
        try:
            before, after = old_blob[1].decode("utf-8"), new_blob[1].decode("utf-8")
        except UnicodeDecodeError:
            edits.append(None)
            continue
        if "\0" in before or "\0" in after:
            edits.append(None)
            continue
        try:
            _file_notice(before, source.license_spdx)
            _file_notice(after, source.license_spdx)
        except ValueError:
            edits.append(None)
            continue
        action = _one_line_action(before, after, file_id, filetype)
        if action is None:
            edits.append(None)
            continue
        edits.append((parent, child, old_blob[0], new_blob[0], before, after, *action))
    rows: list[dict[str, Any]] = []
    for prior, target in zip(edits, edits[1:], strict=False):
        if prior is None or target is None:
            continue
        (
            prior_parent,
            prior_child,
            prior_old_oid,
            prior_new_oid,
            prior_before,
            prior_after,
            prior_row,
            prior_action,
        ) = prior
        (
            target_parent,
            target_child,
            target_old_oid,
            target_new_oid,
            target_before,
            target_after,
            target_row,
            target_action,
        ) = target
        if prior_action.kind != "replace_line" or prior_after != target_before:
            continue
        # History replays from an observed earlier commit into the exact target pre-state.
        old_line = (
            prior_before.splitlines(keepends=True)[prior_row].removesuffix("\n").removesuffix("\r")
        )
        history = (RecentEdit(prior_row, old_line, cast(str, prior_action.text)),)
        try:
            check = EditState(file_id, filetype, prior_before, prior_row, 0)
            if apply_action(check, prior_action) != target_before:
                continue
            state = EditState(file_id, filetype, target_before, target_row, 0, history)
            if apply_action(state, target_action) != target_after:
                continue
        except ValueError:
            continue
        identity = _sha(
            json.dumps([source.repo_id, source.file_path, prior_child, target_child]).encode()
        )
        rows.append(
            {
                "id": f"git-chronology/{identity[:24]}",
                "state": asdict(state),
                "action": asdict(target_action),
                "after_source": target_after,
                "source_type": "git_cross_commit_chronology",
                "source_repo": source.repo_id,
                "source_aliases": list(source.aliases),
                "source_revision": target_child,
                "source_license": source.license_spdx,
                "session_or_commit": target_child,
                "mechanism": "unreviewed_git_change",
                "generator_family": f"public_git_chronology_v1/{source.repo_id}",
                "template_id": f"git/{source.repo_id}/{target_child}/{source.file_path}",
                "status": "unreviewed",
                "provenance": {
                    "public_url": source.public_url,
                    "path": source.file_path,
                    "first_parent_base": source.base_rev,
                    "first_parent_tip": source.tip_rev,
                    "prior_parent_commit": prior_parent,
                    "prior_commit": prior_child,
                    "target_parent_commit": target_parent,
                    "target_commit": target_child,
                    "prior_before_git_blob": prior_old_oid,
                    "prior_after_git_blob": prior_new_oid,
                    "target_before_git_blob": target_old_oid,
                    "target_after_git_blob": target_new_oid,
                    "license_path": source.license_path,
                    "license_sha256": source.license_sha256,
                    "file_spdx_notice": _file_notice(target_before, source.license_spdx),
                    "commit_order_observed": True,
                    "editor_edit_order_observed": False,
                    "intent_observed": False,
                },
                "validation": {
                    "history_replays_to_state": True,
                    "apply_reconstructs_committed_child": True,
                    "source_before_sha256": _sha(target_before.encode()),
                    "source_after_sha256": _sha(target_after.encode()),
                    "prior_before_sha256": _sha(prior_before.encode()),
                    "prior_after_sha256": _sha(prior_after.encode()),
                    "inferability_reviewed": False,
                    "accepted": False,
                },
            }
        )
        if len(rows) >= limits.max_candidates:
            break
    return rows
