"""Target-only 2k/32k dependency diagnostic for parents and campaign checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/cpt-recipe-r1"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_long_context_r1")
SESSION_LIMIT_SECONDS = 8 * 60 * 60
FINALIZATION_RESERVE_SECONDS = 30 * 60


def run(command: list[str], *, name: str, env: dict[str, str] | None = None) -> float:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    started = time.time()
    process = subprocess.run(command, text=True, capture_output=True, env=merged)
    elapsed = time.time() - started
    log = OUTPUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit code {process.returncode}")
    return elapsed


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern}, found {matches}")
    return matches[0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0"],
        name="pip",
    )
    run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        name="git",
    )
    environment = {"PYTHONPATH": str(CHECKOUT / "src"), "TOKENIZERS_PARALLELISM": "false"}
    suite = OUTPUT / "long_context_v2.jsonl"
    run(
        [
            sys.executable,
            str(CHECKOUT / "scripts" / "materialize_long_context_diagnostic.py"),
            "--spec",
            str(CHECKOUT / "data" / "benchmarks" / "long_context_v2.spec.json"),
            "--output",
            str(suite),
            "--metadata",
            str(OUTPUT / "long_context_v2.metadata.json"),
        ],
        name="materialize",
        env=environment,
    )
    from huggingface_hub import snapshot_download

    base = Path(
        snapshot_download(
            "Qwen/Qwen3.5-0.8B-Base",
            revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        )
    )
    models = {
        "Base": base,
        "P5": find_one("**/P5/training_metadata.json").parent,
        "P12": find_one("**/P12/training_metadata.json").parent,
    }
    progress_path = find_one("**/campaign_progress.json")
    campaign_root = progress_path.parent
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress["status"] != "complete":
        raise RuntimeError("campaign source has no completion marker")
    selection = json.loads(find_one("**/selection.json").read_text(encoding="utf-8"))
    eligible = selection["eligible_improvements"]
    for label in eligible:
        models[label] = campaign_root / "arms" / label / "final"
    selected = selection["selected_candidate"]
    record = {
        "schema_version": 1,
        "status": "running",
        "models": list(models),
        "development_eligible_models": eligible,
        "provisional_development_selection": selected,
        "completed": [],
        "skipped": [],
    }
    (OUTPUT / "progress.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    estimate = 45 * 60
    for label, model_path in models.items():
        if time.time() - started + estimate + FINALIZATION_RESERVE_SECONDS > SESSION_LIMIT_SECONDS:
            record["skipped"].append({"model": label, "reason": "session reserve"})
            continue
        elapsed = run(
            [
                sys.executable,
                str(CHECKOUT / "scripts" / "evaluate_long_context_diagnostic.py"),
                "--suite",
                str(suite),
                "--model-path",
                str(model_path),
                "--model-label",
                label,
                "--output",
                str(OUTPUT / "results" / f"{label}-2k-32k.jsonl"),
                "--context-tokens",
                "2048",
                "32000",
                "--max-new-tokens",
                "96",
            ],
            name=f"evaluate-{label}",
            env={**environment, "CUDA_VISIBLE_DEVICES": "0"},
        )
        estimate = max(estimate, elapsed * 1.10)
        record["completed"].append(
            {"model": label, "lengths": [2048, 32000], "elapsed_seconds": elapsed}
        )
        (OUTPUT / "progress.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if selected and selected in models:
        if time.time() - started + estimate + FINALIZATION_RESERVE_SECONDS <= SESSION_LIMIT_SECONDS:
            elapsed = run(
                [
                    sys.executable,
                    str(CHECKOUT / "scripts" / "evaluate_long_context_diagnostic.py"),
                    "--suite",
                    str(suite),
                    "--model-path",
                    str(models[selected]),
                    "--model-label",
                    selected,
                    "--output",
                    str(OUTPUT / "results" / f"{selected}-4k-8k-16k.jsonl"),
                    "--context-tokens",
                    "4096",
                    "8192",
                    "16384",
                    "--max-new-tokens",
                    "96",
                ],
                name=f"evaluate-{selected}-middle-lengths",
                env={**environment, "CUDA_VISIBLE_DEVICES": "0"},
            )
            record["completed"].append(
                {"model": selected, "lengths": [4096, 8192, 16384], "elapsed_seconds": elapsed}
            )
    record["status"] = "complete"
    record["elapsed_seconds"] = time.time() - started
    (OUTPUT / "progress.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("long-context diagnostic complete")


if __name__ == "__main__":
    main()
