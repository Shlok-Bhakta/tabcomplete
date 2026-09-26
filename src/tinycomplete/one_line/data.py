"""Public Git edit candidates and split isolation for single-line-edit-v1.

Git commits are durable source evidence, but their diffs are not editor timelines.
Rows from a commit with several changed regions therefore describe reconstructed
atomic states, not observed human keystrokes. This module creates candidates for
independent review; it does not certify that an edit was inferable from its prompt.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .contract import (
    ActionKind,
    EditAction,
    EditState,
    Filetype,
    RecentEdit,
    apply_action,
    physical_lines,
)

LANGUAGE_SUFFIXES: dict[str, Filetype] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
}
BLOCKED_PATH_PARTS = frozenset(
    {"vendor", "node_modules", "dist", "build", "generated", "third_party"}
)
SOURCE_KINDS = frozenset(
    {
        "git_observed_atomic",
        "git_reconstructed_atomic",
        "git_reconstructed_order",
        "synthetic_terminal_keep",
    }
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PublicGitSource:
    """An already downloaded public checkout, pinned to a parent and child commit."""

    checkout: Path
    repo_id: str
    public_url: str
    parent_rev: str
    child_rev: str
    license_spdx: str
    license_path: str
    license_sha256: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.repo_id or not self.public_url.startswith("https://"):
            raise ValueError("public repository identity and HTTPS URL are required")
        if not re.fullmatch(r"[a-f0-9]{40,64}", self.parent_rev):
            raise ValueError("parent revision must be an immutable commit hash")
        if not re.fullmatch(r"[a-f0-9]{40,64}", self.child_rev):
            raise ValueError("child revision must be an immutable commit hash")
        if not self.license_spdx or not re.fullmatch(r"[a-f0-9]{64}", self.license_sha256):
            raise ValueError("license identity and pinned license hash are required")


def _git(source: PublicGitSource, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-pager", "-C", str(source.checkout), *args],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        # Git stderr can contain local paths. Keep it out of reports and logs.
        raise ValueError(f"Git source command failed: {args[0]}")
    return result.stdout


def verify_source(source: PublicGitSource) -> dict[str, Any]:
    """Check commit ancestry and license blob without trusting the worktree."""

    parent = _git(source, "rev-parse", f"{source.parent_rev}^{{commit}}").strip().decode()
    child = _git(source, "rev-parse", f"{source.child_rev}^{{commit}}").strip().decode()
    if parent != source.parent_rev or child != source.child_rev:
        raise ValueError("source revision does not resolve to its pinned commit")
    if (
        source.parent_rev
        not in _git(source, "rev-list", "--parents", "-n", "1", source.child_rev)
        .decode()
        .split()[1:]
    ):
        raise ValueError("parent is not a direct parent of child")
    license_bytes = _git(source, "show", f"{source.child_rev}:{source.license_path}")
    if sha256_bytes(license_bytes) != source.license_sha256:
        raise ValueError("license blob hash mismatch")
    return {
        "repo_id": source.repo_id,
        "public_url": source.public_url,
        "parent_rev": parent,
        "child_rev": child,
        "license_spdx": source.license_spdx,
        "license_path": source.license_path,
        "license_sha256": source.license_sha256,
        "aliases": list(source.aliases),
    }


def _physical_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _line_content(line: str) -> str:
    return line.removesuffix("\r\n") if line.endswith("\r\n") else line.removesuffix("\n")


def _changed_paths(source: PublicGitSource) -> list[str]:
    raw = _git(
        source, "diff", "--name-only", "--diff-filter=M", "-z", source.parent_rev, source.child_rev
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            path = item.decode("utf-8")
        except UnicodeDecodeError:
            continue
        suffix = Path(path).suffix.lower()
        if suffix not in LANGUAGE_SUFFIXES or BLOCKED_PATH_PARTS.intersection(Path(path).parts):
            continue
        if path.endswith((".d.ts", ".pb.go")):
            continue
        paths.append(path)
    return sorted(paths)


def _candidate(
    source: PublicGitSource,
    path: str,
    before: str,
    after: str,
    opcode: tuple[str, int, int, int, int],
    change_count: int,
) -> dict[str, Any] | None:
    tag, a0, a1, b0, b1 = opcode
    new = _physical_lines(after)
    if tag == "replace" and a1 - a0 == b1 - b0 == 1:
        kind, target, text = "replace_line", a0, _line_content(new[b0])
    elif tag == "delete" and a1 - a0 == 1 and b0 == b1:
        kind, target, text = "delete_line", a0, None
    elif tag == "insert" and a0 == a1 and b1 - b0 == 1:
        kind, target, text = "insert_before", a0, _line_content(new[b0])
    else:
        return None
    if text is not None and ("\n" in text or "\r" in text):
        return None
    try:
        state = EditState(
            file_id=f"{source.repo_id}/{path}",
            filetype=LANGUAGE_SUFFIXES[Path(path).suffix.lower()],
            source=before,
            target_row=target,
            cursor_col=0,
            history=(),
            relevant=(),
        )
        action = EditAction(kind=cast(ActionKind, kind), text=text)
        reconstructed = apply_action(state, action)
    except ValueError:
        return None
    # A single-opcode commit must reproduce the actual child byte for byte.
    if change_count == 1 and reconstructed != after:
        return None
    if reconstructed == before:
        return None
    source_type = "git_observed_atomic" if change_count == 1 else "git_reconstructed_atomic"
    identifier = sha256_bytes(
        json.dumps(
            [source.repo_id, source.child_rev, path, opcode, kind, text], ensure_ascii=False
        ).encode()
    )
    return {
        "id": f"git/{identifier[:24]}",
        "state": asdict(state),
        "action": asdict(action),
        "after_source": reconstructed,
        "source_type": source_type,
        "source_repo": source.repo_id,
        "source_aliases": list(source.aliases),
        "source_revision": source.child_rev,
        "source_parent_revision": source.parent_rev,
        "source_license": source.license_spdx,
        "source_license_sha256": source.license_sha256,
        "session_or_commit": source.child_rev,
        "mechanism": "unreviewed_git_change",
        "generator_family": f"public_git_atomic_v1/{source.repo_id}",
        "template_id": f"git/{source.repo_id}/{source.child_rev}/{path}",
        "provenance": {
            "public_url": source.public_url,
            "path": path,
            "opcode": [tag, a0, a1, b0, b1],
            "commit_change_count": change_count,
            "human_edit_order_observed": False,
            "note": "Git diff order is not an editor timeline; this row needs review.",
        },
        "validation": {
            "apply_reconstructs_after": True,
            "equals_committed_child": reconstructed == after,
            "source_before_sha256": sha256_bytes(before.encode("utf-8")),
            "source_after_sha256": sha256_bytes(reconstructed.encode("utf-8")),
            "committed_child_sha256": sha256_bytes(after.encode("utf-8")),
            "inferability_reviewed": False,
        },
    }


def mine_public_git_pair(
    source: PublicGitSource, *, max_bytes: int = 512_000
) -> list[dict[str, Any]]:
    """Extract deterministic one-line review candidates from a pinned commit pair."""

    verify_source(source)
    rows: list[dict[str, Any]] = []
    for path in _changed_paths(source):
        parent_raw = _git(source, "show", f"{source.parent_rev}:{path}")
        child_raw = _git(source, "show", f"{source.child_rev}:{path}")
        if max(len(parent_raw), len(child_raw)) > max_bytes or b"\0" in parent_raw + child_raw:
            continue
        try:
            before, after = parent_raw.decode("utf-8"), child_raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        old, new = _physical_lines(before), _physical_lines(after)
        changes = [
            opcode
            for opcode in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
            if opcode[0] != "equal"
        ]
        for opcode in changes:
            row = _candidate(source, path, before, after, opcode, len(changes))
            if row is not None:
                rows.append(row)
    return rows


def replay_replacement_history(
    original_source: str,
    history: Iterable[RecentEdit],
    *,
    file_id: str,
    filetype: Filetype,
) -> str:
    """Replay exact one-line replacements in a declared synthetic order."""

    current = original_source
    for edit in history:
        lines = physical_lines(current.encode("utf-8"))
        if edit.row >= len(lines) or lines[edit.row].content.decode("utf-8") != edit.old_text:
            raise ValueError("history does not match its pre-edit line")
        state = EditState(file_id, filetype, current, edit.row, 0)
        current = apply_action(state, EditAction("replace_line", edit.new_text))
    return current


def mine_public_git_pair_enriched(
    source: PublicGitSource, *, max_bytes: int = 512_000, max_history: int = 5
) -> list[dict[str, Any]]:
    """Reconstruct prior replacements and terminal keep controls from a Git commit.

    All ordering is synthetic. A keep control means no additional change at a
    target after this commit's replay, not an observed human no-edit decision.
    """

    if max_history < 1 or max_history > 5:
        raise ValueError("history limit must be between one and five")
    base = mine_public_git_pair(source, max_bytes=max_bytes)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base:
        by_path[str(row["provenance"]["path"])].append(row)
    enriched: list[dict[str, Any]] = []
    for path, file_rows in sorted(by_path.items()):
        file_rows.sort(key=lambda row: (int(row["provenance"]["opcode"][1]), row["id"]))
        parent = str(file_rows[0]["state"]["source"])
        filetype = cast(Filetype, file_rows[0]["state"]["filetype"])
        file_id = str(file_rows[0]["state"]["file_id"])
        prior: list[dict[str, Any]] = []
        for target in file_rows:
            target_row = int(target["state"]["target_row"])
            selected = [
                item
                for item in prior[-max_history:]
                if item["action"]["kind"] == "replace_line"
                and int(item["state"]["target_row"]) < target_row
            ]
            if selected:
                parent_lines = physical_lines(parent.encode("utf-8"))
                history = tuple(
                    RecentEdit(
                        row=int(item["state"]["target_row"]),
                        old_text=parent_lines[int(item["state"]["target_row"])].content.decode(
                            "utf-8"
                        ),
                        new_text=str(item["action"]["text"]),
                    )
                    for item in selected
                )
                try:
                    current = replay_replacement_history(
                        parent, history, file_id=file_id, filetype=filetype
                    )
                    state = EditState(file_id, filetype, current, target_row, 0, history)
                    action = EditAction(**target["action"])
                    result = apply_action(state, action)
                except (TypeError, ValueError):
                    prior.append(target)
                    continue
                row = deepcopy(target)
                row["id"] = (
                    "git-history/"
                    + sha256_bytes(
                        json.dumps([target["id"], [item["id"] for item in selected]]).encode()
                    )[:24]
                )
                row["state"] = asdict(state)
                row["action"] = asdict(action)
                row["after_source"] = result
                row["source_type"] = "git_reconstructed_order"
                row["mechanism"] = "unreviewed_git_change_with_history"
                row["provenance"]["prior_candidate_ids"] = [item["id"] for item in selected]
                row["provenance"]["history_origin_sha256"] = sha256_bytes(parent.encode())
                row["provenance"]["human_edit_order_observed"] = False
                row["provenance"]["note"] = (
                    "Prior replacements were replayed in file order, not observed editor order."
                )
                committed = _git(source, "show", f"{source.child_rev}:{path}").decode("utf-8")
                row["validation"] = {
                    **row["validation"],
                    "history_replays_to_state": True,
                    "equals_committed_child": result == committed,
                    "source_before_sha256": sha256_bytes(current.encode()),
                    "source_after_sha256": sha256_bytes(result.encode()),
                    "inferability_reviewed": False,
                }
                enriched.append(row)
            prior.append(target)

        # The exact child is a terminal control only when every changed region
        # was one replacement and all those replacements replay exactly.
        if len(file_rows) < 2 or len(file_rows) > max_history:
            continue
        if any(item["action"]["kind"] != "replace_line" for item in file_rows):
            continue
        if len(file_rows) != int(file_rows[0]["provenance"]["commit_change_count"]):
            continue
        parent_lines = physical_lines(parent.encode("utf-8"))
        history = tuple(
            RecentEdit(
                row=int(item["state"]["target_row"]),
                old_text=parent_lines[int(item["state"]["target_row"])].content.decode("utf-8"),
                new_text=str(item["action"]["text"]),
            )
            for item in file_rows
        )
        try:
            current = replay_replacement_history(
                parent, history, file_id=file_id, filetype=filetype
            )
            committed = _git(source, "show", f"{source.child_rev}:{path}").decode("utf-8")
            if current != committed:
                continue
            target_row = history[-1].row
            state = EditState(file_id, filetype, current, target_row, 0, history)
        except (TypeError, ValueError, UnicodeDecodeError):
            continue
        action = EditAction("keep")
        row = deepcopy(file_rows[-1])
        row["id"] = (
            "git-terminal/"
            + sha256_bytes(
                json.dumps([source.repo_id, source.child_rev, path, target_row]).encode()
            )[:24]
        )
        row["state"] = asdict(state)
        row["action"] = asdict(action)
        row["after_source"] = current
        row["source_type"] = "synthetic_terminal_keep"
        row["mechanism"] = "terminal_no_additional_edit_control"
        row["provenance"]["prior_candidate_ids"] = [item["id"] for item in file_rows]
        row["provenance"]["history_origin_sha256"] = sha256_bytes(parent.encode())
        row["provenance"]["not_observed_no_edit"] = True
        row["provenance"]["human_edit_order_observed"] = False
        row["provenance"]["note"] = (
            "After full commit replay, no additional edit to this target is evidenced; "
            "this is a synthetic terminal control."
        )
        row["validation"] = {
            **row["validation"],
            "history_replays_to_state": True,
            "equals_committed_child": True,
            "source_before_sha256": sha256_bytes(current.encode()),
            "source_after_sha256": sha256_bytes(current.encode()),
            "inferability_reviewed": False,
        }
        enriched.append(row)
    return enriched


_NORMALIZE = re.compile(r"(?:\b\d+(?:\.\d+)?\b)|(?:\b[A-Za-z_][A-Za-z_0-9]*\b)")


def near_duplicate_key(row: Mapping[str, Any]) -> str:
    """Conservative normalized local-code fingerprint for split grouping."""

    state = row["state"]
    lines = str(state["source"]).splitlines()
    target = int(state["target_row"])
    excerpt = "\n".join(lines[max(0, target - 3) : target + 4])
    normalized = _NORMALIZE.sub("_", excerpt)
    normalized = " ".join(normalized.split())
    action = row["action"]
    payload = _NORMALIZE.sub("_", str(action.get("text") or ""))
    return sha256_bytes(
        json.dumps([state["filetype"], action["kind"], normalized, payload]).encode()
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(b)] = self.find(a)


def connected_groups(rows: Iterable[Mapping[str, Any]]) -> list[list[int]]:
    """Connect repo aliases, commits, template families, and normalized near duplicates."""

    records = list(rows)
    union = _UnionFind(len(records))
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(records):
        keys = [("repo", str(row["source_repo"]).casefold())]
        keys += [("repo", str(alias).casefold()) for alias in row.get("source_aliases", ())]
        keys += [("commit", str(row["session_or_commit"]))]
        keys += [("family", str(row["generator_family"]))]
        keys += [("template", str(row["template_id"]))]
        keys += [("near", near_duplicate_key(row))]
        for key in keys:
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def validate_splits(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject cross-split provenance links and duplicate pre-edit states."""

    records = list(rows)
    ids: set[str] = set()
    pre_states: set[str] = set()
    counts: Counter[str] = Counter()
    train_mechanisms: set[str] = set()
    heldout_mechanisms: set[str] = set()
    for row in records:
        identifier, split = str(row["id"]), str(row["split"])
        if identifier in ids:
            raise ValueError("duplicate row id")
        ids.add(identifier)
        if split not in {"train", "development", "test_new_repo", "test_new_mechanism"}:
            raise ValueError("unknown split")
        counts[split] += 1
        if split == "train":
            train_mechanisms.add(str(row["mechanism"]))
        elif split == "test_new_mechanism":
            heldout_mechanisms.add(str(row["mechanism"]))
        state_hash = sha256_bytes(
            json.dumps(row["state"], sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        if state_hash in pre_states:
            raise ValueError("duplicate model input state")
        pre_states.add(state_hash)
    if train_mechanisms & heldout_mechanisms:
        raise ValueError("reserved mechanism appears in training")
    for group in connected_groups(records):
        if len({records[index]["split"] for index in group}) != 1:
            raise ValueError("connected source group crosses splits")
    return {
        "rows": len(records),
        "splits": dict(sorted(counts.items())),
        "groups": len(connected_groups(records)),
    }
