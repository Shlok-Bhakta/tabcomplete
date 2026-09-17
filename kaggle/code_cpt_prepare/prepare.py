"""CPU-only Kaggle job that freezes the Stage-1 packed corpus."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/code-cpt"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_corpus")


def run(
    command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "datasets>=4,<5", "transformers==5.5.0"])
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)])
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    run(
        [
            sys.executable,
            "-m",
            "tinycomplete.code_cpt.prepare",
            "--output-dir",
            str(OUTPUT),
            "--train-tokens",
            "12000000",
            "--validation-tokens",
            "16384",
            "--general-tokens",
            "16384",
            "--block-size",
            "2048",
            "--shuffle-buffer",
            "10000",
            "--seed",
            "314159",
        ],
        cwd=CHECKOUT,
        env={**os.environ, "PYTHONPATH": str(CHECKOUT / "src")},
    )
    metadata_path = OUTPUT / "corpus_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["git_sha"] = git_sha
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("Corpus preparation: PASS")
    print("Git SHA:", git_sha)
    print("Actual train tokens:", metadata["actual_train_tokens"])


if __name__ == "__main__":
    main()
