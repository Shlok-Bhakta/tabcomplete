"""Build a fixed blind audit from the pinned structural Git scan.

The intermediate editor state has synthetic within-commit order. This script
does not turn its gold lines into observed editor edits or training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from probe_active_git_sequence_yield import _atoms


def _show(checkout: Path, rev: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{rev}:{path}"],
        check=False,
        capture_output=True,
        timeout=45,
    )
    if result.returncode:
        raise ValueError("pinned Git blob unavailable")
    return result.stdout


def build(
    plan: dict[str, Any], result: dict[str, Any], checkout_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {item["id"]: (repo, item) for repo in result["results"] for item in repo["examples"]}
    blind: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for identity in plan["ids"]:
        repo, item = by_id[identity]
        checkout = (
            checkout_root
            / {
                "pydantic/pydantic": "pydantic",
                "vuejs/core": "vue_core",
                "tokio-rs/tokio": "tokio",
                "gin-gonic/gin": "gin",
            }[repo["repo"]]
        )
        license_bytes = _show(checkout, item["child"], repo["license_path"])
        historical = plan.get("historical_license_overrides", {}).get(identity)
        license_hash = historical["sha256"] if historical else repo["license_sha256"]
        if hashlib.sha256(license_bytes).hexdigest() != license_hash:
            raise ValueError("historical child license differs from source plan")
        if b"MIT License" not in license_bytes and b"MIT LICENSE" not in license_bytes:
            raise ValueError("historical child license is not MIT")
        before = _show(checkout, item["parent"], item["path"]).decode("utf-8")
        after = _show(checkout, item["child"], item["path"]).decode("utf-8")
        atoms = _atoms(before, after)
        prior = next(row for row in atoms if row[0] == item["prior_row"])
        target = next(row for row in atoms if row[0] == item["target_row"])
        if prior[3] != target[3] or list(prior[3]) != item["substitution"]:
            raise ValueError("recorded structural pair changed")
        source_lines = before.splitlines()
        if source_lines[prior[0]] != prior[1] or source_lines[target[0]] != target[1]:
            raise ValueError("recorded source rows changed")
        source_lines[prior[0]] = prior[2]
        low = max(0, target[0] - 15)
        high = min(len(source_lines), target[0] + 16)
        context = [{"row_1based": row + 1, "text": source_lines[row]} for row in range(low, high)]
        blind.append(
            {
                "id": identity,
                "filetype": Path(item["path"]).suffix,
                "target_row_1based": target[0] + 1,
                "prior_edit": {
                    "row_1based": prior[0] + 1,
                    "before": prior[1],
                    "after": prior[2],
                    "order_observed": False,
                },
                "context": context,
                "question": (
                    "Predict the exact replacement target line, or abstain. "
                    "Is this edit inferable and meaningful from this visible state?"
                ),
            }
        )
        gold.append(
            {
                "id": identity,
                "repo": repo["repo"],
                "path": item["path"],
                "parent": item["parent"],
                "child": item["child"],
                "target_before": target[1],
                "target_after": target[2],
                "substitution": list(target[3]),
                "source_type": "git_within_commit_synthetic_order",
                "accepted_training_label": False,
            }
        )
    return blind, gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--checkout-root", required=True, type=Path)
    parser.add_argument("--blind-output", required=True, type=Path)
    parser.add_argument("--gold-output", required=True, type=Path)
    args = parser.parse_args()
    if args.blind_output.exists() or args.gold_output.exists():
        raise FileExistsError("blind review artifact already exists")
    plan = json.loads(args.plan.read_text())
    result = json.loads(args.source_result.read_text())
    blind, gold = build(plan, result, args.checkout_root)
    args.blind_output.parent.mkdir(parents=True, exist_ok=True)
    args.gold_output.parent.mkdir(parents=True, exist_ok=True)
    args.blind_output.write_text(json.dumps(blind, indent=2, ensure_ascii=False) + "\n")
    args.gold_output.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
