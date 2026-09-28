"""Safety checks for the bounded public Git chronology pilot wrapper."""

from __future__ import annotations

import json
import runpy
from argparse import Namespace
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_one_line_chronology_pilot.py"


def test_transfer_budget_never_forwards_beyond_cap() -> None:
    budget_type = runpy.run_path(str(RUNNER))["TransferBudget"]
    budget = budget_type(5)
    assert budget.take(b"123") == b"123"
    assert budget.take(b"4567") == b"45"
    assert budget.take(b"8") == b""
    assert budget.used == 5
    assert budget.exhausted.is_set()


def test_generated_seed_is_recorded_without_network(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(RUNNER))
    run = namespace["run"]
    row = {
        "student_state_seed": {"filetype": "typescript"},
        "authoring_metadata": {
            "source_repo": "example/example",
            "source_revision": "a" * 40,
            "source_path": "generated/example.d.ts",
        },
    }
    run.__globals__["_ordered_seeds"] = lambda _: [row]
    args = Namespace(
        seeds=tmp_path / "unused",
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        max_repos=1,
        max_commits=32,
        transfer_cap_bytes=1024,
        per_repo_seconds=1,
        max_total_seconds=5,
    )
    report = run(args)
    progress = json.loads((args.report_dir / "progress.json").read_text())
    assert report["status_counts"] == {"excluded_generated_path": 1}
    assert report["proxied_transfer_bytes"] == 0
    assert progress["results"][0]["status"] == "excluded_generated_path"
    assert report["accepted_training_label_count"] == 0
