"""Bounded structural replayability screen for pinned public agent traces."""

from __future__ import annotations

import argparse
import collections
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    if not isinstance(function, dict):
        return {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except json.JSONDecodeError:
            return []
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def screen_messages(messages: list[dict[str, Any]]) -> collections.Counter[str]:
    """Count optimistic candidates; shell-free is necessary but not sufficient."""
    counts: collections.Counter[str] = collections.Counter()
    previous_editor_paths: set[str] = set()
    prior_bash = False
    for message in messages:
        for call in _calls(message):
            function = call.get("function") or {}
            name = function.get("name") if isinstance(function, dict) else None
            if name == "bash":
                prior_bash = True
                counts["bash_calls"] += 1
            if name != "str_replace_editor":
                continue
            arguments = _arguments(call)
            if arguments.get("command") != "str_replace":
                continue
            path = arguments.get("path")
            old = arguments.get("old_str")
            new = arguments.get("new_str")
            if not isinstance(path, str) or not isinstance(old, str) or not isinstance(new, str):
                continue
            if any(mark in old or mark in new for mark in ("\r", "\n")):
                previous_editor_paths.add(path)
                continue
            counts["one_line_calls"] += 1
            if path in previous_editor_paths:
                counts["one_line_with_prior_editor_same_file"] += 1
                if prior_bash:
                    counts["one_line_with_prior_editor_and_bash"] += 1
                else:
                    counts["optimistic_shell_free_one_line_sequences"] += 1
            previous_editor_paths.add(path)
    return counts


def run(plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    request = urllib.request.Request(
        "https://huggingface.co/api/datasets/" + plan["dataset"],
        headers={"User-Agent": "tabcomplete-public-trace-replayability"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        metadata = json.load(response)
    if metadata["sha"] != plan["revision"]:
        raise RuntimeError("trace dataset revision changed")
    total_bytes = 0
    totals: collections.Counter[str] = collections.Counter()
    eligible_rows: list[int] = []
    for offset in plan["offsets"]:
        query = urllib.parse.urlencode(
            {
                "dataset": plan["dataset"],
                "config": plan["config"],
                "split": plan["split"],
                "offset": offset,
                "length": plan["rows_per_offset"],
            }
        )
        request = urllib.request.Request(
            "https://datasets-server.huggingface.co/rows?" + query,
            headers={"User-Agent": "tabcomplete-public-trace-replayability"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(plan["maximum_one_response_bytes"] + 1)
        if len(raw) > plan["maximum_one_response_bytes"]:
            raise RuntimeError("response byte cap exceeded")
        total_bytes += len(raw)
        if total_bytes > plan["maximum_total_response_bytes"]:
            raise RuntimeError("total transfer byte cap exceeded")
        rows = json.loads(raw)["rows"]
        if len(rows) != plan["rows_per_offset"]:
            raise RuntimeError("unexpected row count")
        for entry in rows:
            totals["rows"] += 1
            counts = screen_messages(entry["row"].get("messages") or [])
            totals.update(counts)
            if counts["optimistic_shell_free_one_line_sequences"]:
                eligible_rows.append(entry["row_idx"])
    result = {
        "source_plan": str(plan_path),
        "revision": plan["revision"],
        "response_bytes": total_bytes,
        "counts": dict(sorted(totals.items())),
        "optimistic_shell_free_row_indices": eligible_rows,
        "reconstructed_states": 0,
        "accepted_training_labels": 0,
        "caveat": (
            "A shell-free call still needs pinned base state, tool-success verification, "
            "license, and inferability review."
        ),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.plan, args.output)["counts"])
