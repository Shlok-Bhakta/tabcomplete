"""Bounded one-line mapping audit of Zed's pinned public Zeta train rows."""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from tinycomplete.one_line.contract import ActionKind, EditAction, EditState, Filetype, apply_action

START = "<|editable_region_start|>"
END = "<|editable_region_end|>"
CURSOR = "<|user_cursor_is_here|>"
SUFFIXES: dict[str, Filetype] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unpack(text: str) -> tuple[str, str, str, str]:
    if not text.startswith("```") or text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("missing or repeated control marker")
    header, separator, body = text.partition("\n")
    if not separator or not body.endswith("\n```"):
        raise ValueError("unsupported source fence")
    path = header[3:]
    body = body[:-4]
    prefix, _, tail = body.partition(START)
    region, marker, suffix = tail.partition(END)
    if not marker:
        raise ValueError("unclosed editable region")
    return path, prefix.replace(CURSOR, ""), region.replace(CURSOR, ""), suffix.replace(CURSOR, "")


def classify(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(row.get("input"), str) or row["input"].count(CURSOR) != 1:
        return "cursor_missing_or_repeated", None
    try:
        before_path, before_prefix, before_region, before_suffix = _unpack(row["input"])
        after_path, after_prefix, after_region, after_suffix = _unpack(row["output"])
    except (KeyError, TypeError, ValueError):
        return "unsupported_markers", None
    if (before_path, before_prefix, before_suffix) != (
        after_path,
        after_prefix,
        after_suffix,
    ):
        return "outside_region_changed", None
    filetype = next(
        (lang for suffix, lang in SUFFIXES.items() if before_path.endswith(suffix)), None
    )
    if filetype is None:
        return "unsupported_filetype", None
    if before_region == after_region:
        return "no_edit", {"filetype": filetype, "path": before_path}
    source = before_prefix + before_region + before_suffix
    target_source = after_prefix + after_region + after_suffix
    old = before_region.splitlines(keepends=True)
    new = after_region.splitlines(keepends=True)
    changed = [
        opcode
        for opcode in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
        if opcode[0] != "equal"
    ]
    if len(changed) != 1:
        return "multiple_region_changes", None
    tag, a0, a1, b0, b1 = changed[0]
    prior_lines = before_prefix.count("\n")
    target_row = prior_lines + a0
    cursor_row = row["input"].partition(CURSOR)[0].count("\n") - 1
    cursor_line_prefix = (
        row["input"].partition(CURSOR)[0].rsplit("\n", 1)[-1].replace(START, "").replace(END, "")
    )
    kind: ActionKind
    if tag == "replace" and a1 - a0 == 1 and b1 - b0 == 1:
        kind = "replace_line"
        text = new[b0].removesuffix("\n").removesuffix("\r")
    elif tag == "delete" and a1 - a0 == 1 and b0 == b1:
        kind = "delete_line"
        text = None
    elif tag == "insert" and a0 == a1 and b1 - b0 == 1:
        kind = "insert_before"
        text = new[b0].removesuffix("\n").removesuffix("\r")
    else:
        return "not_one_line_action", None
    try:
        state = EditState(before_path, filetype, source, target_row, 0)
        action = EditAction(kind, text)
        exact_replay = apply_action(state, action) == target_source
    except (TypeError, ValueError):
        return "invalid_action_state", None
    if not exact_replay:
        return "replay_mismatch", None
    target_old_line = old[a0].removesuffix("\n").removesuffix("\r") if a0 < len(old) else ""
    target_new_line = new[b0].removesuffix("\n").removesuffix("\r") if b0 < len(new) else ""
    suffix_applyable = (
        cursor_row == target_row
        and kind == "replace_line"
        and target_old_line.startswith(cursor_line_prefix)
        and target_new_line.startswith(cursor_line_prefix)
    )
    return "one_line_exact", {
        "filetype": filetype,
        "path": before_path,
        "action": kind,
        "cursor_row": cursor_row,
        "target_row": target_row,
        "cursor_on_target_row": cursor_row == target_row,
        "cursor_suffix_applyable": suffix_applyable,
        "row_distance": abs(cursor_row - target_row),
        "source_sha256": _sha(source.encode()),
        "after_sha256": _sha(target_source.encode()),
        "events_present": bool(row.get("events")),
        "repo_identity_available": False,
    }


def run(plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    request = urllib.request.Request(
        "https://huggingface.co/api/datasets/" + plan["dataset"],
        headers={"User-Agent": "tabcomplete-zeta-mapping-audit"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        metadata = json.load(response)
    if metadata["sha"] != plan["revision"]:
        raise RuntimeError("Zeta dataset revision changed")
    query = urllib.parse.urlencode(
        {
            "dataset": plan["dataset"],
            "config": "default",
            "split": "train",
            "offset": plan["offset"],
            "length": plan["rows"],
        }
    )
    request = urllib.request.Request(
        "https://datasets-server.huggingface.co/rows?" + query,
        headers={"User-Agent": "tabcomplete-zeta-mapping-audit"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(plan["maximum_response_bytes"] + 1)
    if len(raw) > plan["maximum_response_bytes"]:
        raise RuntimeError("Zeta response exceeds frozen byte cap")
    entries = json.loads(raw)["rows"]
    if len(entries) != plan["rows"]:
        raise RuntimeError("Zeta viewer returned unexpected row count")
    counts: collections.Counter[str] = collections.Counter()
    filetypes: collections.Counter[str] = collections.Counter()
    actions: collections.Counter[str] = collections.Counter()
    cursor_matches = 0
    cursor_suffix_applyable = 0
    examples: list[dict[str, Any]] = []
    for entry in entries:
        classification, detail = classify(entry["row"])
        counts[classification] += 1
        if detail and classification == "one_line_exact":
            filetypes[detail["filetype"]] += 1
            actions[detail["action"]] += 1
            cursor_matches += int(detail["cursor_on_target_row"])
            cursor_suffix_applyable += int(detail["cursor_suffix_applyable"])
            if len(examples) < plan["maximum_recorded_examples"]:
                examples.append({"row_index": entry["row_idx"], **detail})
    result = {
        "plan": str(plan_path),
        "dataset_revision": plan["revision"],
        "response_bytes": len(raw),
        "response_sha256": _sha(raw),
        "counts": dict(sorted(counts.items())),
        "filetypes": dict(sorted(filetypes.items())),
        "actions": dict(sorted(actions.items())),
        "one_line_cursor_on_target_row": cursor_matches,
        "one_line_cursor_suffix_applyable": cursor_suffix_applyable,
        "examples": examples,
        "accepted_training_labels": 0,
        "source_limit": "No verified repository, source commit, or file license.",
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.plan, args.output)["counts"])
