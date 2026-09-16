"""Git history miner for synthetic edit trajectories.

Given parent/child commits: modified files -> unified diff -> changed
regions -> enclosing syntax nodes (Tree-sitter, optional) -> synthetic
intermediate states with future changes as labels.

WARNING: Git diff hunk order is NOT human temporal edit order. Provenance is
always ``git_synthetic`` and every record carries that disclaimer. Never
label this data as human behavior.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

from .schema import Provenance

__all__ = [
    "GitHunk",
    "GitEdit",
    "NOT_HUMAN_ORDER_NOTE",
    "extract_commit_pair",
    "mine_linear_history",
]

NOT_HUMAN_ORDER_NOTE = (
    "git_synthetic: diff hunk order is not human temporal edit order; "
    "intermediate states are synthetic stepping stones, not observed behavior."
)


@dataclass(frozen=True)
class GitHunk:
    old_start: int  # 1-based line number in parent
    old_count: int
    new_start: int  # 1-based line number in child
    new_count: int
    removed: tuple[str, ...]
    added: tuple[str, ...]
    enclosing: tuple[str, ...] = ()  # enclosing syntax node types (Tree-sitter)
    lines: tuple[tuple[str, str], ...] = ()  # ordered (" "/"-"/"+", text) incl. newline


@dataclass
class GitEdit:
    path: str
    parent_rev: str
    child_rev: str
    parent_text: str
    child_text: str
    hunks: list[GitHunk] = field(default_factory=list)
    intermediates: list[str] = field(default_factory=list)
    provenance: Provenance = Provenance.GIT_SYNTHETIC
    note: str = NOT_HUMAN_ORDER_NOTE


def _git(repo: str, *args: str) -> str:
    env = {**os.environ, "GIT_PAGER": "cat", "LC_ALL": "C"}
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _show_or_empty(repo: str, rev: str, name: str) -> str:
    """Blob content at rev, or "" when the file does not exist there."""
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{rev}:{name}"],
        cwd=repo,
        capture_output=True,
        env={**os.environ, "GIT_PAGER": "cat"},
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return _git(repo, "show", f"{rev}:{name}")


def _enclosing_nodes(source: str, language: str, start_line: int, end_line: int) -> tuple[str, ...]:
    """Innermost-to-outermost syntax node types covering 1-based line range."""
    if language != "python":
        return ()
    try:
        from tree_sitter_language_pack import get_parser
    except Exception:
        return ()
    try:
        raw = source.encode("utf-8")
        lines = raw.split(b"\n")
        if start_line < 1 or end_line > len(lines) + 1:
            return ()
        start_byte = sum(len(lines[i]) + 1 for i in range(start_line - 1))
        end_byte = sum(len(lines[i]) + 1 for i in range(min(end_line, len(lines))))
        parser = get_parser("python")
        root = parser.parse(raw).root_node
        start_byte = min(start_byte, len(raw))
        end_byte = min(end_byte, len(raw))
        while end_byte > start_byte and raw[end_byte - 1 : end_byte] == b"\n":
            end_byte -= 1
        node = root.descendant_for_byte_range(start_byte, max(end_byte, start_byte + 1))
        chain = []
        while node is not None and node.type != "module":
            chain.append(node.type)
            node = node.parent
        if not chain and start_byte < len(raw):
            # Hunk range straddles top-level items: anchor on its midpoint.
            mid = (start_byte + min(end_byte, len(raw))) // 2
            node = root.descendant_for_byte_range(mid, min(mid + 1, len(raw)))
            while node is not None and node.type != "module":
                chain.append(node.type)
                node = node.parent
        return tuple(chain)
    except Exception:
        return ()


def extract_commit_pair(repo: str, parent_rev: str, child_rev: str) -> list[GitEdit]:
    """Extract per-file synthetic edits between two commits (deterministic)."""
    from unidiff import PatchSet

    names = _git(repo, "diff", "--name-only", parent_rev, child_rev).split()
    edits: list[GitEdit] = []
    for name in sorted(names):
        if not name.endswith(".py"):
            continue
        parent_text = _show_or_empty(repo, parent_rev, name)
        child_text = _show_or_empty(repo, child_rev, name)
        diff = _git(repo, "diff", "-U3", parent_rev, child_rev, "--", name)
        patched = PatchSet(diff)
        hunks: list[GitHunk] = []
        for pf in patched:
            for h in pf:
                removed = tuple(line.value for line in h if line.line_type == "-")
                added = tuple(line.value for line in h if line.line_type == "+")
                ordered = tuple((line.line_type, line.value) for line in h)
                enclosing = _enclosing_nodes(
                    parent_text, "python", h.source_start, h.source_start + h.source_length
                )
                hunks.append(
                    GitHunk(
                        old_start=h.source_start,
                        old_count=h.source_length,
                        new_start=h.target_start,
                        new_count=h.target_length,
                        removed=removed,
                        added=added,
                        enclosing=enclosing,
                        lines=ordered,
                    )
                )
        intermediates = _intermediates(parent_text, hunks)
        edits.append(
            GitEdit(
                path=name,
                parent_rev=parent_rev,
                child_rev=child_rev,
                parent_text=parent_text,
                child_text=child_text,
                hunks=hunks,
                intermediates=intermediates,
            )
        )
    return edits


def _intermediates(parent_text: str, hunks: list[GitHunk]) -> list[str]:
    """Apply hunks cumulatively (in diff order) to parent lines.

    Walks each hunk's ordered lines so added lines land after their
    surrounding context instead of at the hunk start.
    """
    lines = parent_text.splitlines(keepends=True)
    states: list[str] = []
    offset = 0  # lines added-minus-removed by earlier hunks (all before this one)
    for h in hunks:
        at = h.old_start - 1 + offset
        for kind, text in h.lines:
            if kind == " ":
                if at >= len(lines) or lines[at] != text:
                    raise ValueError(f"context mismatch at line {at + 1}: {text!r}")
                at += 1
            elif kind == "-":
                if at >= len(lines) or lines[at] != text:
                    raise ValueError(f"removed-line mismatch at line {at + 1}: {text!r}")
                del lines[at]
            elif kind == "+":
                lines.insert(at, text)
                at += 1
            else:
                raise ValueError(f"unknown hunk line kind: {kind!r}")
        offset += sum(1 for kind, _ in h.lines if kind == "+") - sum(
            1 for kind, _ in h.lines if kind == "-"
        )
        states.append("".join(lines))
    return states


def mine_linear_history(repo: str, max_pairs: int = 20) -> list[GitEdit]:
    """Walk first-parent history (newest first) and extract consecutive pairs."""
    revs = _git(repo, "log", "--first-parent", "--format=%H", "--reverse").split()
    edits: list[GitEdit] = []
    for parent, child in zip(revs, revs[1:], strict=False):
        if len(edits) >= max_pairs:
            break
        edits.extend(extract_commit_pair(repo, parent, child))
    return edits
