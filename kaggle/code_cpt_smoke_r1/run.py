"""Bounded production-runtime smoke from the verified P12 checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/cpt-recipe-r1"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_smoke_r1")
P12_SHA256 = "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43"


def run(command: list[str], *, log: Path, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    process = subprocess.run(command, text=True, capture_output=True, env=merged)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log.name}")


def locate_inputs() -> tuple[Path, Path]:
    corpora = []
    for metadata_path in Path("/kaggle/input").glob("**/corpus_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("actual_train_tokens") == 11_999_232:
            corpora.append(metadata_path.parent)
    archives = list(Path("/kaggle/input").glob("**/P12.zip"))
    if len(corpora) != 1 or len(archives) != 1:
        raise RuntimeError(f"expected one corpus and P12 archive, got {len(corpora)} and {len(archives)}")
    parent = Path("/kaggle/working/parents/P12")
    parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(parent)
    return corpora[0], parent


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==5.5.0",
            "accelerate==1.13.0",
            "bitsandbytes==0.50.2",
            "flash-linear-attention==0.5.2",
        ],
        log=OUTPUT / "pip.log",
    )
    run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        log=OUTPUT / "git.log",
    )
    corpus, parent = locate_inputs()
    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "tinycomplete.code_cpt.train",
            "train",
            "--corpus-dir",
            str(corpus),
            "--output-dir",
            str(OUTPUT / "run"),
            "--init-from",
            str(parent),
            "--expected-initial-sha256",
            P12_SHA256,
            "--parent-training-tokens",
            "11993088",
            "--learning-rate",
            "3e-6",
            "--lr-schedule",
            "constant",
            "--microbatch",
            "1",
            "--gradient-accumulation",
            "8",
            "--optimizer",
            "adamw_8bit",
            "--no-gradient-checkpointing",
            "--max-tokens",
            "98304",
            "--workers",
            "1",
            "--warmup-steps",
            "5",
            "--distributed-mode",
            "fsdp",
            "--save-final",
        ],
        log=OUTPUT / "run.log",
        env={
            "PYTHONPATH": str(CHECKOUT / "src"),
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    summary = json.loads((OUTPUT / "run" / "summary.json").read_text(encoding="utf-8"))
    (OUTPUT / "session.json").write_text(
        json.dumps(
            {
                "elapsed_seconds_including_setup": time.time() - started,
                "steady_state_tokens_per_second": summary["steady_state_tokens_per_second"],
                "tokens_per_second": summary["tokens_per_second"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("production smoke: PASS")


if __name__ == "__main__":
    main()
