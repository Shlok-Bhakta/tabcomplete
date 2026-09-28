"""Bounded metadata-only structural probe of a pinned public trace dataset."""

from __future__ import annotations

import argparse
import collections
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _get_json(url: str, byte_cap: int) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(url, headers={"User-Agent": "tabcomplete-public-trace-probe"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(byte_cap + 1)
    if len(raw) > byte_cap:
        raise RuntimeError("public trace probe response exceeded frozen byte cap")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("public trace probe response was not an object")
    return data, len(raw)


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if isinstance(calls, str):
        calls = json.loads(calls)
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    if not isinstance(function, dict):
        return {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return arguments if isinstance(arguments, dict) else {}


def run(plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    metadata, _ = _get_json("https://huggingface.co/api/datasets/" + plan["dataset"], 2_000_000)
    if metadata.get("sha") != plan["dataset_revision"]:
        raise RuntimeError("public trace dataset revision changed")

    total_bytes = 0
    rows_seen = 0
    repositories: set[str] = set()
    languages: collections.Counter[str] = collections.Counter()
    licenses: collections.Counter[str] = collections.Counter()
    outcomes: collections.Counter[str] = collections.Counter()
    editor_commands: collections.Counter[str] = collections.Counter()
    one_line_calls = 0
    rows_with_one_line_call = 0
    malformed_call_rows = 0
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
        response, size = _get_json(
            "https://datasets-server.huggingface.co/rows?" + query,
            plan["maximum_one_response_bytes"],
        )
        if total_bytes + size > plan["maximum_total_response_bytes"]:
            raise RuntimeError("public trace probe exceeded total response-byte cap")
        total_bytes += size
        rows = response.get("rows") or []
        if len(rows) != plan["rows_per_offset"]:
            raise RuntimeError("public trace probe returned an unexpected row count")
        for entry in rows:
            row = entry["row"]
            rows_seen += 1
            repositories.add(str(row.get("repo")))
            languages[str(row.get("language"))] += 1
            licenses[str(row.get("license"))] += 1
            outcomes[str(row.get("resolved"))] += 1
            found_one_line = False
            try:
                for message in row.get("messages") or []:
                    for call in _tool_calls(message):
                        function = call.get("function") or {}
                        if function.get("name") != "str_replace_editor":
                            continue
                        arguments = _arguments(call)
                        command = str(arguments.get("command"))
                        editor_commands[command] += 1
                        if command != "str_replace":
                            continue
                        old = arguments.get("old_str")
                        new = arguments.get("new_str")
                        if not isinstance(old, str) or not isinstance(new, str):
                            continue
                        if any(mark in old or mark in new for mark in ("\r", "\n")):
                            continue
                        one_line_calls += 1
                        found_one_line = True
            except (TypeError, ValueError, AttributeError):
                malformed_call_rows += 1
            rows_with_one_line_call += int(found_one_line)

    result: dict[str, Any] = {
        "probe_plan": str(plan_path),
        "dataset_revision": plan["dataset_revision"],
        "config": plan["config"],
        "split": plan["split"],
        "offsets": plan["offsets"],
        "rows": rows_seen,
        "response_bytes": total_bytes,
        "unique_repositories": len(repositories),
        "languages": dict(sorted(languages.items())),
        "repository_licenses": dict(sorted(licenses.items())),
        "trajectory_resolved_flags": dict(sorted(outcomes.items())),
        "editor_commands": dict(sorted(editor_commands.items())),
        "one_line_str_replace_calls": one_line_calls,
        "trajectories_with_one_line_str_replace": rows_with_one_line_call,
        "malformed_call_rows": malformed_call_rows,
        "reconstructed_pre_states": 0,
        "independently_accepted_labels": 0,
        "sampling_note": (
            "Five fixed offset blocks; structural screen only, not a random quality sample"
        ),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(
        f"rows={result['rows']} one_line_calls={result['one_line_str_replace_calls']} "
        f"trajectories={result['trajectories_with_one_line_str_replace']}"
    )


if __name__ == "__main__":
    main()
