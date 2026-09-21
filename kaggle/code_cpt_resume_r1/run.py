"""Actual two-T4 continuous-versus-resumed checkpoint test."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/cpt-recipe-r1"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_resume_r1")
P12_SHA256 = "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43"
TOKENS_PER_UPDATE = 32_768


def run(command: list[str], *, name: str, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    process = subprocess.run(command, text=True, capture_output=True, env=merged)
    (OUTPUT / f"{name}.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit code {process.returncode}")


def locate_inputs() -> tuple[Path, Path]:
    corpora = []
    for metadata_path in Path("/kaggle/input").glob("**/corpus_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("actual_train_tokens") == 5_013_504:
            corpora.append(metadata_path.parent)
    parents = []
    for metadata_path in Path("/kaggle/input").glob("**/P12/training_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("training_tokens") == 11_993_088:
            parents.append(metadata_path.parent)
    if len(corpora) != 1 or len(parents) != 1:
        raise RuntimeError(f"expected one fresh corpus and P12 parent, got {corpora} and {parents}")
    return corpora[0], parents[0]


def train_command(corpus: Path, output: Path, *, updates: int) -> list[str]:
    return [
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
        str(output),
        "--learning-rate",
        "3e-6",
        "--lr-schedule",
        "constant",
        "--lr-floor",
        "3e-7",
        "--decay-end-update",
        "153",
        "--microbatch",
        "1",
        "--gradient-accumulation",
        "8",
        "--optimizer",
        "adamw_8bit",
        "--no-gradient-checkpointing",
        "--max-tokens",
        str(updates * TOKENS_PER_UPDATE),
        "--workers",
        "1",
        "--warmup-steps",
        "5",
        "--parent-training-tokens",
        "11993088",
        "--distributed-mode",
        "fsdp",
        "--save-resume",
    ]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        name="pip",
    )
    run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        name="git",
    )
    corpus, parent = locate_inputs()
    environment = {"PYTHONPATH": str(CHECKOUT / "src"), "TOKENIZERS_PARALLELISM": "false"}

    continuous = train_command(corpus, OUTPUT / "continuous", updates=8)
    continuous.extend(
        ["--init-from", str(parent), "--expected-initial-sha256", P12_SHA256]
    )
    run(continuous, name="continuous", env=environment)

    split_first = train_command(corpus, OUTPUT / "split-first", updates=4)
    split_first.extend(
        ["--init-from", str(parent), "--expected-initial-sha256", P12_SHA256]
    )
    run(split_first, name="split-first", env=environment)

    split_second = train_command(corpus, OUTPUT / "split-second", updates=8)
    split_second.extend(
        ["--resume-from", str(OUTPUT / "split-first" / "resume-latest")]
    )
    run(split_second, name="split-second", env=environment)

    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "tinycomplete.code_cpt.resume_compare",
            "--checkpoint-a",
            str(OUTPUT / "continuous" / "resume-latest"),
            "--checkpoint-b",
            str(OUTPUT / "split-second" / "resume-latest"),
            "--output",
            str(OUTPUT / "state-comparison.json"),
        ],
        name="compare",
        env=environment,
    )

    continuous_summary = read_json(OUTPUT / "continuous" / "summary.json")
    first_summary = read_json(OUTPUT / "split-first" / "summary.json")
    second_summary = read_json(OUTPUT / "split-second" / "summary.json")
    state_comparison = read_json(OUTPUT / "state-comparison.json")
    continuous_losses = continuous_summary["loss_history"]
    split_losses = first_summary["loss_history"] + second_summary["loss_history"]
    loss_differences = [
        abs(left["loss"] - right["loss"])
        for left, right in zip(continuous_losses, split_losses, strict=True)
    ]
    result = {
        "schema_version": 1,
        "elapsed_seconds_including_setup": time.time() - started,
        "parent": "P12",
        "parent_sha256": P12_SHA256,
        "continuous_updates": continuous_summary["optimizer_steps"],
        "split_first_updates": first_summary["optimizer_steps"],
        "resumed_final_updates": second_summary["optimizer_steps"],
        "continuous_next_block": continuous_summary["corpus_position"]["next_block"],
        "resumed_next_block": second_summary["corpus_position"]["next_block"],
        "same_examples_consumed_once": (
            continuous_summary["corpus_position"]["next_block"]
            == second_summary["corpus_position"]["next_block"]
            == 128
        ),
        "max_loss_absolute_difference": max(loss_differences),
        "loss_absolute_differences": loss_differences,
        "state_exact_match": state_comparison["exact_match"],
        "state_comparison": state_comparison,
    }
    (OUTPUT / "resume_test.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(OUTPUT / "split-first" / "resume-latest")
    print("resume test complete")


if __name__ == "__main__":
    main()
