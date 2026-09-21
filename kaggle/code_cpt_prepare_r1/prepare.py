"""Freeze the fresh research-r1 CPT train, development, and test slices."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/cpt-recipe-r1"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_research_r1")


def run(command: list[str], *, log: Path) -> None:
    process = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log.name}")


def find_old_corpus() -> Path:
    candidates = []
    for path in Path("/kaggle/input").glob("**/train_blocks.npy"):
        metadata_path = path.parent / "corpus_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("actual_train_tokens") == 11_999_232 and metadata.get("seed") == 314159:
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one frozen Stage-1 corpus, found {len(candidates)}")
    return candidates[0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0", "datasets==4.8.5"],
        log=OUTPUT / "pip.log",
    )
    run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        log=OUTPUT / "git.log",
    )
    old_corpus = find_old_corpus()
    run(
        [
            sys.executable,
            "-m",
            "tinycomplete.code_cpt.prepare",
            "--research-r1",
            "--output-dir",
            str(OUTPUT / "corpus"),
            "--old-corpus-dir",
            str(old_corpus),
            "--exclusion-root",
            str(CHECKOUT / "data" / "benchmarks"),
            "--train-tokens",
            "5013504",
            "--validation-tokens",
            "114688",
            "--general-tokens",
            "16384",
            "--block-size",
            "2048",
            "--shuffle-buffer",
            "10000",
            "--seed",
            "424242",
        ],
        log=OUTPUT / "prepare.log",
    )
    print("research-r1 corpus: PASS")


if __name__ == "__main__":
    main()
