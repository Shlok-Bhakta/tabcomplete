"""Bounded local q25 diagnostic on synthetic-order public Git edit states.

The history order is constructed, not observed. This is a narrow research
diagnostic and cannot be used as a general editor quality estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from build_active_git_blind_review import _show
from probe_active_git_sequence_yield import _atoms


def _post(url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.load(response)
    return result


def _history(lines: list[str], prior: tuple[int, str, str, tuple[str, str]]) -> str:
    old, new = prior[1].encode(), prior[2].encode()
    common = 0
    while common < min(len(old), len(new)) and old[common] == new[common]:
        common += 1
    while common > 0 and ((old[common] & 0xC0) == 0x80 or (new[common] & 0xC0) == 0x80):
        common -= 1
    suffix = 0
    while (
        suffix < min(len(old) - common, len(new) - common) and old[-suffix - 1] == new[-suffix - 1]
    ):
        suffix += 1
    prefix = ("\n".join(lines[: prior[0]]) + ("\n" if prior[0] else "")).encode()
    start = len(prefix) + common
    end = len(prefix) + len(old) - suffix
    inserted = new[common : len(new) - suffix if suffix else len(new)].decode()
    return f"<actual-recent-edit start={start} end={end}>\n{inserted}\n</actual-recent-edit>\n"


def _prompt(
    before: str,
    prior: tuple[int, str, str, tuple[str, str]],
    target: tuple[int, str, str, tuple[str, str]],
    path: str,
    filetype: str,
    cursor_col: int,
) -> str:
    lines = before.splitlines()
    lines[prior[0]] = prior[2]
    row = target[0]
    target_bytes = lines[row].encode()
    if cursor_col < 0 or cursor_col > len(target_bytes):
        raise ValueError("cursor outside target line")
    line_prefix = target_bytes[:cursor_col].decode()
    region = target_bytes[cursor_col:].decode()
    history = _history(lines, prior)
    previous = "\n".join(lines[max(0, row - 80) : row])
    if previous:
        previous += "\n"
    later = lines[row + 1 : row + 41]
    after = "\n" + "\n".join(later) if later else ""
    region_start = len(("\n".join(lines[:row]) + ("\n" if row else "")).encode()) + cursor_col
    return (
        f"<repo {path}>\n<filetype {filetype}>\n{history}<file {path}>\n"
        f"{previous}{line_prefix}[[EDIT]]{region}[[/EDIT]]{after}\n</file>\n"
        f"<P {path} {region_start}>\n"
        "Return one compact next-edit action: N\\n for no edit or R\\n followed by exact "
        "replacement text. End with EOS.\n"
    )


def run(plan: dict[str, Any], scan: dict[str, Any], checkout_root: Path) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(plan["server_url"])
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("only the configured loopback inference server is allowed")
    model = Path(plan["model_file"])
    if hashlib.sha256(model.read_bytes()).hexdigest() != plan["model_sha256"]:
        raise ValueError("selected GGUF hash mismatch")
    by_id = {e["id"]: (repo, e) for repo in scan["results"] for e in repo["examples"]}
    results = []
    for identity in plan["ids"]:
        repo, item = by_id[identity]
        alias = plan["repo_aliases"][repo["repo"]]
        checkout = checkout_root / alias
        before = _show(checkout, item["parent"], item["path"]).decode()
        after = _show(checkout, item["child"], item["path"]).decode()
        atoms = _atoms(before, after)
        prior = next(a for a in atoms if a[0] == item["prior_row"])
        target = next(a for a in atoms if a[0] == item["target_row"])
        if prior[3] != target[3] or list(prior[3]) != item["substitution"]:
            raise ValueError("frozen structural candidate changed")
        filetype = plan["filetypes"][Path(item["path"]).suffix]
        path = "/tmp/public-edit-replay/" + alias + "/" + item["path"]
        cursor_policy = plan.get("cursor_policy", "line_start")
        if cursor_policy == "line_start":
            cursor_col = 0
        elif cursor_policy == "identifier_start":
            cursor_col = target[1].encode().find(prior[3][0].encode())
            if cursor_col < 0 or target[1].encode()[:cursor_col] != target[2].encode()[:cursor_col]:
                raise ValueError("identifier cursor lacks a preserved prefix")
        else:
            raise ValueError("unknown frozen cursor policy")
        prompt = _prompt(before, prior, target, path, filetype, cursor_col)
        tokens = _post(
            plan["server_url"], "/tokenize", {"content": prompt, "add_special": False}, 10
        )
        prompt_tokens = len(tokens["tokens"])
        if prompt_tokens > plan["maximum_prompt_tokens"]:
            results.append(
                {"id": identity, "skipped": "prompt_token_budget", "prompt_tokens": prompt_tokens}
            )
            continue
        started = time.monotonic()
        response = _post(
            plan["server_url"],
            "/completion",
            {
                "prompt": prompt,
                "n_predict": plan["decoding"]["maximum_output_tokens"],
                "temperature": 0,
                "stream": False,
                "cache_prompt": False,
                "id_slot": 0,
                "n_keep": 0,
            },
            plan["request_timeout_seconds"],
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        wire = response.get("content")
        stop_type = response.get("stop_type")
        valid = (
            stop_type == "eos"
            and isinstance(wire, str)
            and (wire == "N\n" or wire.startswith("R\n"))
        )
        predicted = wire[2:] if valid and isinstance(wire, str) and wire.startswith("R\n") else None
        expected = target[2].encode()[cursor_col:].decode()
        results.append(
            {
                "id": identity,
                "cursor_col_bytes": cursor_col,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": response.get("tokens_predicted"),
                "elapsed_ms": elapsed_ms,
                "stop_type": stop_type,
                "valid": valid,
                "action": "replace"
                if predicted is not None
                else "no_edit"
                if wire == "N\n" and valid
                else "invalid",
                "exact_target": predicted == expected,
                "wire": wire,
                "target_sha256": hashlib.sha256(expected.encode()).hexdigest(),
                "target_kind": "string_or_comment"
                if target[1].lstrip().startswith(("#", "//"))
                else "code",
                "synthetic_order": True,
            }
        )
    return {
        "plan_version": plan["version"],
        "model_sha256": plan["model_sha256"],
        "results": results,
        "exact_targets": sum(r.get("exact_target", False) for r in results),
        "accepted_training_labels": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic result already exists")
    plan = json.loads(args.plan.read_text())
    scan = json.loads(args.scan.read_text())
    result = run(plan, scan, args.checkout_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
