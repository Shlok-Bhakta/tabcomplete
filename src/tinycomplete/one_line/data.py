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
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from .context import serialize_state_bounded
from .contract import (
    ActionKind,
    EditAction,
    EditState,
    Filetype,
    RecentEdit,
    apply_action,
    encode_action,
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
AUTHORING_FOCUS = (
    "local API call update",
    "argument propagation",
    "return value handling",
    "field access consistency",
    "literal or boundary correction",
    "import justified by use",
    "exception or result propagation",
    "method rename",
    "collection operation",
    "assertion tied to visible behavior",
    "cleanup after prior removal",
    "type annotation consistency",
    "null or option handling",
    "branch guard",
    "loop boundary",
    "string formatting",
    "asynchronous flow",
    "resource cleanup",
    "path handling",
    "serialization",
    "logging consistency",
    "error message",
    "configuration use",
    "call site consistency",
    "test expectation",
)
_PUBLIC_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_SENSITIVE_TEXT = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9_]{20,}\b"
    r"|\bsk-[A-Za-z0-9]{20,}\b|(?i:authorization\s*:\s*bearer\s+\S+)"
    r"|(?i:\b(?:password|passwd|api[_-]?key|secret|access[_-]?token)\b\s*[:=]\s*[\"'][^\"']{8,}[\"'])"
    r"|(?i:://[^\s/:]+:[^\s/@]+@)"
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


def discover_r2_public_sources(pool_path: Path, filetype: Filetype) -> list[dict[str, Any]]:
    """Use R2 Stack records only as leads for a fresh pinned public fetch."""

    suffixes = {"python": {".py"}, "typescript": {".ts"}, "rust": {".rs"}, "go": {".go"}}
    leads: list[dict[str, Any]] = []
    with pool_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            repo, path = row.get("repository"), row.get("path")
            if not isinstance(repo, str) or not _PUBLIC_REPO.fullmatch(repo):
                continue
            if not isinstance(path, str) or Path(path).suffix.lower() not in suffixes[filetype]:
                continue
            if row.get("licenses") != ["MIT"] or row.get("parse_status") != "pass":
                continue
            old_content = row.get("content")
            if not isinstance(old_content, str) or not 500 <= len(old_content) <= 20_000:
                continue
            parts = {part.lower() for part in Path(path).parts}
            if parts & (BLOCKED_PATH_PARTS | {"test", "tests", "examples", "docs", "fixtures"}):
                continue
            if Path(path).name.lower().startswith(("test_", "spec_")):
                continue
            lead = {
                "source_repo": repo,
                "source_aliases": sorted(set(row.get("repository_aliases") or [repo])),
                "source_path": path,
                "filetype": filetype,
                "r2_discovery_content_sha256": row["content_sha256"],
                "r2_discovery_revision": "17cad72c886a2858e08d4c349a00d6466f54df63",
            }
            leads.append(lead)
    return sorted(
        leads,
        key=lambda row: sha256_bytes((row["source_repo"] + "/" + row["source_path"]).encode()),
    )


def _github_head(repo: str) -> str | None:
    result = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD"],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode or not result.stdout.strip():
        return None
    revision = result.stdout.split()[0]
    return revision if re.fullmatch(r"[a-f0-9]{40}", revision) else None


def _github_blob(repo: str, revision: str, path: str, *, max_bytes: int = 65_536) -> bytes | None:
    if not _PUBLIC_REPO.fullmatch(repo) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("invalid immutable GitHub source identity")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("invalid public source path")
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/" + urllib.parse.quote(
        path, safe="/"
    )
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            content = response.read(max_bytes + 1)
    except urllib.error.HTTPError as error:
        if error.code in (403, 429):
            raise RuntimeError("public source host rate limited") from None
        return None
    except urllib.error.URLError:
        return None
    return content if len(content) <= max_bytes else None


def verify_live_public_source(
    lead: Mapping[str, Any],
    *,
    resolve_head: Callable[[str], str | None] = _github_head,
    fetch_blob: Callable[[str, str, str], bytes | None] | None = None,
) -> dict[str, Any] | None:
    """Pin a live source and MIT license to the same immutable commit.

    The R2 source bytes are never substituted when the historical blob is gone.
    A repository license does not prove every file's licensing; explicit
    conflicting file SPDX notices are rejected and missing notices are recorded.
    """

    repo, path = str(lead["source_repo"]), str(lead["source_path"])
    if not _PUBLIC_REPO.fullmatch(repo):
        raise ValueError("invalid public repository name")
    revision = resolve_head(repo)
    if revision is None or not re.fullmatch(r"[a-f0-9]{40}", revision):
        return None
    if fetch_blob is None:
        fetch_blob = _github_blob
    source_bytes = fetch_blob(repo, revision, path)
    if source_bytes is None or not 500 <= len(source_bytes) <= 20_000:
        return None
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if _SENSITIVE_TEXT.search(source) or "\x00" in source:
        return None
    notices = re.findall(r"SPDX-License-Identifier:\s*([^\r\n*]+)", source[:2048], re.I)
    if notices and any(not re.fullmatch(r"\s*MIT\s*", notice, re.I) for notice in notices):
        return None
    if any(len(part) > 20_000 for part in source.splitlines()):
        return None
    # A function-like source is more likely to support a meaningful one-line
    # editor task than a pure constant dump or generated table.
    signal = {
        "python": r"\b(?:def|class)\s+[A-Za-z_]",
        "typescript": r"\b(?:function|class|interface|type|const)\b|=>",
        "rust": r"\b(?:fn|impl|struct|enum)\b",
        "go": r"\b(?:func|type)\b",
    }[str(lead["filetype"])]
    if not re.search(signal, source):
        return None
    try:
        from tree_sitter_language_pack import get_parser

        if get_parser(str(lead["filetype"])).parse(source_bytes).root_node.has_error:
            return None
    except (ImportError, KeyError, ValueError):
        return None
    license_path = None
    license_bytes = None
    for candidate in ("LICENSE", "LICENSE.md", "LICENSE.txt", "license", "LICENSE-MIT"):
        value = fetch_blob(repo, revision, candidate)
        if value is None:
            continue
        text = value.decode("utf-8", errors="replace")
        if "GNU GENERAL PUBLIC LICENSE" in text.upper():
            continue
        if "MIT License" in text or "Permission is hereby granted, free of charge" in text:
            license_path, license_bytes = candidate, value
            break
    if license_path is None or license_bytes is None:
        return None
    source_hash = sha256_bytes(source_bytes)
    identifier = sha256_bytes(json.dumps([repo, revision, path]).encode())
    suffix = Path(path).suffix.lower()
    return {
        "id": f"public-source/{identifier[:24]}",
        "student_state_seed": {
            "file_id": f"file_{identifier[:12]}{suffix}",
            "filetype": lead["filetype"],
            "source": source,
        },
        "authoring_metadata": {
            "source_repo": repo,
            "source_aliases": list(lead.get("source_aliases", (repo,))),
            "source_revision": revision,
            "source_path": path,
            "source_url": f"https://github.com/{repo}/blob/{revision}/{urllib.parse.quote(path)}",
            "source_sha256": source_hash,
            "source_license": "MIT",
            "license_path": license_path,
            "license_sha256": sha256_bytes(license_bytes),
            "file_spdx_notice": notices[0].strip() if notices else None,
            "r2_discovery_content_sha256": lead["r2_discovery_content_sha256"],
            "r2_discovery_revision": lead["r2_discovery_revision"],
            "authoring_focus": None,
            "focus_is_author_only": True,
            "student_action_absent": True,
            "source_provenance_verified": True,
            "file_license_scope_unverified_without_notice": not bool(notices),
        },
    }


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def parse_author_response(
    response_text: str,
    source_row: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Validate one Muse author candidate without trusting its task judgment.

    The author supplies one prior replacement and one target action. The pinned
    public source supplies all file bytes; the response must not duplicate it.
    The caller provides the immutable q25 tokenizer, loaded once per batch.
    """

    if not isinstance(response_text, str) or len(response_text.encode("utf-8")) > 8192:
        raise ValueError("author response exceeds bounded JSON size")
    try:
        raw = json.loads(
            response_text,
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("author response is not strict JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "prior_edit",
        "target_row",
        "action",
        "intent_evidence",
        "objective",
    }:
        raise ValueError("author response has missing or unknown fields")
    prior, target_action, objective = raw["prior_edit"], raw["action"], raw["objective"]
    if not isinstance(prior, dict) or set(prior) != {"row", "old_text", "new_text"}:
        raise ValueError("prior edit must be one exact replacement")
    if type(prior["row"]) is not int or prior["row"] < 0:
        raise ValueError("prior edit row is invalid")
    if not isinstance(prior["old_text"], str) or not isinstance(prior["new_text"], str):
        raise ValueError("prior edit text must be UTF-8 strings")
    if prior["old_text"] == prior["new_text"] or any(
        char in text for text in (prior["old_text"], prior["new_text"]) for char in "\r\n"
    ):
        raise ValueError("prior replacement must change exactly one physical line")
    if type(raw["target_row"]) is not int or raw["target_row"] < 0:
        raise ValueError("target row is invalid")
    if not isinstance(target_action, dict) or set(target_action) != {"kind", "text"}:
        raise ValueError("target action has missing or unknown fields")
    if target_action["kind"] not in ("N", "D", "R", "I"):
        raise ValueError("target action kind is not canonical")
    if not isinstance(raw["intent_evidence"], str) or not 1 <= len(raw["intent_evidence"]) <= 1000:
        raise ValueError("intent evidence is missing or too long")
    if not isinstance(objective, dict) or not {"kind", "description"} <= set(objective):
        raise ValueError("objective needs kind and description")
    if set(objective) - {"kind", "description", "checks"}:
        raise ValueError("objective contains an unknown field")
    if not isinstance(objective["kind"], str) or not re.fullmatch(
        r"[a-z][a-z0-9_]{0,39}", objective["kind"]
    ):
        raise ValueError("objective kind is invalid")
    if (
        not isinstance(objective["description"], str)
        or not 1 <= len(objective["description"]) <= 1000
    ):
        raise ValueError("objective description is missing or too long")
    checks = objective.get("checks", [])
    if (
        not isinstance(checks, list)
        or len(checks) > 5
        or any(not isinstance(item, str) or not 1 <= len(item) <= 300 for item in checks)
    ):
        raise ValueError("objective checks are invalid")
    seed = source_row["student_state_seed"]
    source = str(seed["source"])
    metadata = source_row["authoring_metadata"]
    if sha256_bytes(source.encode("utf-8")) != metadata["source_sha256"]:
        raise ValueError("pinned public source hash mismatch")
    if metadata["source_license"] != "MIT" or not re.fullmatch(
        r"[a-f0-9]{40}", str(metadata["source_revision"])
    ):
        raise ValueError("public source identity is not pinned and licensed")
    if source in response_text:
        raise ValueError("author response duplicates the full source")
    file_id = str(seed["file_id"])
    filetype = cast(Filetype, seed["filetype"])
    history = (RecentEdit(prior["row"], prior["old_text"], prior["new_text"]),)
    current = replay_replacement_history(source, history, file_id=file_id, filetype=filetype)
    state = EditState(file_id, filetype, current, raw["target_row"], 0, history)
    kind = {"N": "keep", "D": "delete_line", "R": "replace_line", "I": "insert_before"}[
        target_action["kind"]
    ]
    action = EditAction(cast(ActionKind, kind), target_action["text"])
    if (
        action.kind == "replace_line"
        and state.target_row == history[0].row
        and action.text == history[0].old_text
    ):
        raise ValueError("target reverses the prior intended replacement")
    after = apply_action(state, action)
    if action.kind != "keep" and after == current:
        raise ValueError("edit-required action leaves source unchanged")
    wire = encode_action(action)
    response_tokens = len(tokenizer.encode(wire, add_special_tokens=False)) + 1
    if response_tokens > 64:
        raise ValueError("target action exceeds 64 q25 response tokens including EOS")
    context = serialize_state_bounded(state, tokenizer, max_input_tokens=1024)
    if context.included_history != 1 or context.input_tokens is None:
        raise ValueError("recent edit is absent from bounded student context")
    if context.input_tokens + response_tokens > 2048:
        raise ValueError("candidate exceeds total token budget")
    response_hash = sha256_bytes(response_text.encode("utf-8"))
    return {
        "id": "muse-author/"
        + sha256_bytes(json.dumps([source_row["id"], response_hash]).encode())[:24],
        "state": asdict(state),
        "action": asdict(action),
        "after_source": after,
        "source_type": "muse_author_public_candidate",
        "source_repo": metadata["source_repo"],
        "source_aliases": metadata["source_aliases"],
        "source_revision": metadata["source_revision"],
        "source_license": metadata["source_license"],
        "session_or_commit": metadata["source_revision"],
        "mechanism": "unreviewed_teacher_author",
        "generator_family": "muse_author_public_v1/" + str(metadata["source_repo"]),
        "template_id": "authoring-source/" + str(source_row["id"]),
        "provenance": {
            "source_id": source_row["id"],
            "source_sha256": metadata["source_sha256"],
            "source_license_sha256": metadata["license_sha256"],
            "author_response_sha256": response_hash,
            "requested_focus": metadata["authoring_focus"],
            "intent_evidence": raw["intent_evidence"],
            "objective": objective,
            "prior_edit_order": "teacher_constructed_synthetic",
            "human_edit_order_observed": False,
        },
        "validation": {
            "history_replays_to_state": True,
            "apply_reconstructs_after": True,
            "source_before_sha256": sha256_bytes(current.encode("utf-8")),
            "source_after_sha256": sha256_bytes(after.encode("utf-8")),
            "input_tokens": context.input_tokens,
            "response_tokens_including_eos": response_tokens,
            "history_visible_in_prompt": True,
            "blind_solver_verified": False,
            "reviewer_verified": False,
            "objective_verified": False,
            "accepted_training": False,
        },
    }


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
