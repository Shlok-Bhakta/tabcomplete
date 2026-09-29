"""Screen recent public Git history for repeated one-line identifier edits.

This is source-yield evidence. Git commit order does not reveal edit order within
a commit, so no result is an accepted editor trajectory or training label.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tinycomplete.one_line.chronology import _identifier_replacement


def _git(checkout: Path, *args: str, timeout: int = 60) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"public Git command failed: {args[0]}")
    return result.stdout


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _clone(repo: dict[str, Any], root: Path, deadline: float) -> Path:
    checkout = root / repo["alias"]
    if not checkout.exists():
        result = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--depth=129",
                "--no-checkout",
                "--quiet",
                "https://github.com/" + repo["name"] + ".git",
                str(checkout),
            ],
            capture_output=True,
            timeout=max(1, int(deadline - time.monotonic())),
            check=False,
        )
        if result.returncode:
            raise RuntimeError("pinned public Git clone failed")
    if _git(checkout, "rev-parse", "HEAD").strip().decode() != repo["head"]:
        raise RuntimeError("public repository HEAD changed since plan freeze")
    return checkout


def _eligible_path(path: str, suffix: str) -> bool:
    parts = set(Path(path).parts)
    return (
        path.endswith(suffix)
        and not path.endswith((".d.ts", ".pb.go"))
        and not parts.intersection({"vendor", "node_modules", "dist", "build", "generated"})
    )


def _atoms(before: str, after: str) -> list[tuple[int, str, str, tuple[str, str]]]:
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    atoms: list[tuple[int, str, str, tuple[str, str]]] = []
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(
        None, old_lines, new_lines, autojunk=False
    ).get_opcodes():
        if tag != "replace" or a1 - a0 != b1 - b0 or a1 - a0 > 20:
            continue
        for offset in range(a1 - a0):
            old_line = old_lines[a0 + offset]
            new_line = new_lines[b0 + offset]
            if len(old_line.encode()) > 512 or len(new_line.encode()) > 512:
                continue
            old_text = old_line.removesuffix("\n").removesuffix("\r")
            new_text = new_line.removesuffix("\n").removesuffix("\r")
            if old_line[len(old_text) :] != new_line[len(new_text) :]:
                continue
            change = _identifier_replacement(old_text, new_text)
            if change:
                atoms.append((a0 + offset, old_text, new_text, change))
    return atoms


def _pairs(
    atoms: list[tuple[int, str, str, tuple[str, str]]],
) -> list[tuple[int, int, tuple[str, str]]]:
    matches: list[tuple[int, int, tuple[str, str]]] = []
    for i, first in enumerate(atoms):
        for second in atoms[i + 1 :]:
            if second[0] - first[0] > 80:
                break
            if second[3] == first[3]:
                matches.append((first[0], second[0], first[3]))
                break
    return matches


def _screen_repo(
    repo: dict[str, Any], root: Path, limits: dict[str, int], deadline: float
) -> dict[str, Any]:
    checkout = _clone(repo, root, deadline)
    license_bytes = _git(checkout, "show", repo["head"] + ":" + repo["license_path"])
    if hashlib.sha256(license_bytes).hexdigest() != repo["license_sha256"]:
        raise RuntimeError("pinned repository license bytes changed")
    if b"MIT License" not in license_bytes and b"MIT LICENSE" not in license_bytes:
        raise RuntimeError("pinned repository license text is not MIT")
    chain = (
        _git(
            checkout,
            "rev-list",
            "--first-parent",
            f"--max-count={limits['commits'] + 1}",
            repo["head"],
        )
        .decode()
        .splitlines()
    )
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for child, parent in zip(chain, chain[1:], strict=False):
        if time.monotonic() > deadline:
            counts["deadline_stop"] += 1
            break
        if _size(checkout) > limits["per_repo_bytes"]:
            counts["storage_stop"] += 1
            break
        counts["commit_pairs"] += 1
        paths = [
            raw.decode("utf-8")
            for raw in _git(
                checkout, "diff", "--name-only", "--diff-filter=M", "-z", parent, child
            ).split(b"\0")
            if raw
        ]
        eligible = sorted(path for path in paths if _eligible_path(path, repo["suffix"]))
        counts["eligible_changed_files"] += len(eligible)
        for path in eligible[: limits["files_per_commit"]]:
            counts["files_inspected"] += 1
            sizes = (
                int(_git(checkout, "cat-file", "-s", parent + ":" + path)),
                int(_git(checkout, "cat-file", "-s", child + ":" + path)),
            )
            if max(sizes) > limits["source_file_bytes"]:
                counts["oversize_files"] += 1
                continue
            try:
                before = _git(checkout, "show", parent + ":" + path).decode("utf-8")
                after = _git(checkout, "show", child + ":" + path).decode("utf-8")
            except UnicodeDecodeError:
                counts["non_utf8_files"] += 1
                continue
            if "\0" in before or "\0" in after:
                counts["binary_files"] += 1
                continue
            pairs = _pairs(_atoms(before, after))
            if pairs:
                counts["files_with_pair"] += 1
            for prior_row, target_row, substitution in pairs:
                counts["structural_pairs"] += 1
                if len(examples) < limits["recorded_examples_per_repo"]:
                    identity = hashlib.sha256(
                        json.dumps([repo["name"], child, path, prior_row, target_row]).encode()
                    ).hexdigest()
                    examples.append(
                        {
                            "id": "active-git/" + identity[:24],
                            "repo": repo["name"],
                            "child": child,
                            "parent": parent,
                            "path": path,
                            "prior_row": prior_row,
                            "target_row": target_row,
                            "substitution": list(substitution),
                            "source_type": "git_within_commit_synthetic_order",
                            "accepted_training_label": False,
                        }
                    )
    return {
        "repo": repo["name"],
        "head": repo["head"],
        "license_path": repo["license_path"],
        "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        "checkout_bytes": _size(checkout),
        "counts": dict(sorted(counts.items())),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkout-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("screen output already exists")
    plan = json.loads(args.plan.read_text())
    args.checkout_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + plan["limits"]["wall_seconds"]
    results = []
    for repo in plan["repositories"]:
        if time.monotonic() > deadline:
            break
        result = _screen_repo(repo, args.checkout_root, plan["limits"], deadline)
        results.append(result)
        if (
            sum(
                _size(args.checkout_root / item["alias"])
                for item in plan["repositories"]
                if (args.checkout_root / item["alias"]).exists()
            )
            > plan["limits"]["total_checkout_bytes"]
        ):
            raise RuntimeError("total checkout budget exceeded")
    output = {
        "plan": str(args.plan),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
        "attempted_repositories": len(results),
        "total_structural_pairs": sum(row["counts"].get("structural_pairs", 0) for row in results),
        "accepted_training_labels": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "attempted_repositories": len(results),
                "total_structural_pairs": output["total_structural_pairs"],
            }
        )
    )


if __name__ == "__main__":
    main()
