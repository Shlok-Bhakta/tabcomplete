"""Bounded real update probes for 2k through 32k coherent sequences."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/cpt-recipe-r1"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_context_probe_r1")
P12_SHA256 = "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43"
GEOMETRY = {
    2_048: {"accumulation": 8, "tokens_per_update": 32_768},
    4_096: {"accumulation": 4, "tokens_per_update": 32_768},
    8_192: {"accumulation": 2, "tokens_per_update": 32_768},
    16_384: {"accumulation": 1, "tokens_per_update": 32_768},
    32_768: {"accumulation": 1, "tokens_per_update": 65_536},
}


def command_run(
    command: list[str], *, name: str, env: dict[str, str] | None = None
) -> tuple[int, float, str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    started = time.time()
    process = subprocess.run(command, text=True, capture_output=True, env=merged)
    elapsed = time.time() - started
    output = process.stdout + process.stderr
    log = OUTPUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output, encoding="utf-8")
    return process.returncode, elapsed, output


def locate_parent() -> Path:
    matches = []
    for path in Path("/kaggle/input").glob("**/P12/training_metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("training_tokens") == 11_993_088:
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one P12 parent, found {matches}")
    return matches[0]


def prepare_probe_corpus(sequence_length: int, accumulation: int, tokenizer) -> tuple[Path, dict]:
    from tinycomplete.eval.long_context_diagnostic import build_diagnostic_family

    block_count = 2 * accumulation
    corpus = OUTPUT / "corpora" / f"seq-{sequence_length:05d}"
    corpus.mkdir(parents=True, exist_ok=True)
    blocks = np.lib.format.open_memmap(
        corpus / "train_blocks.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(block_count, sequence_length),
    )
    records = []
    families = ["constant", "enum", "signature", "field", "config"]
    for index in range(block_count):
        family = families[index % len(families)]
        cases = build_diagnostic_family(
            tokenizer=tokenizer,
            family=family,
            target_tokens=max(512, int(sequence_length * 0.94)),
            seed=910_000 + sequence_length + index,
            tolerance_fraction=0.04,
        )
        case = next(item for item in cases if item.condition == "long_far")
        token_ids = tokenizer.encode(case.prompt, add_special_tokens=False)
        source_tokens = len(token_ids)
        padding_source = (
            "\n# Additional coherent repository notes for the optimizer feasibility probe.\n"
        )
        padding_ids = tokenizer.encode(padding_source, add_special_tokens=False)
        while len(token_ids) < sequence_length:
            token_ids.extend(padding_ids)
        truncated_padding_tokens = len(token_ids) - sequence_length
        token_ids = token_ids[:sequence_length]
        blocks[index] = np.asarray(token_ids, dtype=np.uint32)
        records.append(
            {
                "block": index,
                "family": family,
                "prompt_sha256": case.prompt_sha256,
                "source_prompt_tokens": source_tokens,
                "added_padding_tokens": len(token_ids) - source_tokens,
                "truncated_padding_tokens": truncated_padding_tokens,
                "coherent_repository_prompt": True,
            }
        )
    blocks.flush()
    np.save(corpus / "train_languages.npy", np.zeros(block_count, dtype=np.uint8))
    metadata = {
        "schema_version": 1,
        "campaign": "long_context_training_feasibility_r1",
        "block_size": sequence_length,
        "training_blocks": block_count,
        "actual_train_tokens": block_count * sequence_length,
        "language_order": ["python"],
        "records": records,
    }
    (corpus / "corpus_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return corpus, metadata


def train_command(
    corpus: Path,
    parent: Path,
    destination: Path,
    sequence_length: int,
    accumulation: int,
    *,
    memory_retry: bool,
) -> list[str]:
    command = [
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
        str(destination),
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
        str(accumulation),
        "--optimizer",
        "adamw_8bit",
        "--max-tokens",
        str(GEOMETRY[sequence_length]["tokens_per_update"]),
        "--workers",
        "1",
        "--warmup-steps",
        "5",
        "--distributed-mode",
        "fsdp",
    ]
    if memory_retry:
        command.extend(["--gradient-checkpointing", "--no-fsdp-forward-prefetch"])
    else:
        command.extend(["--no-gradient-checkpointing", "--fsdp-forward-prefetch"])
    return command


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    code, _, _ = command_run(
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
    if code:
        raise RuntimeError("dependency installation failed")
    code, _, _ = command_run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        name="git",
    )
    if code:
        raise RuntimeError("repository clone failed")
    sys.path.insert(0, str(CHECKOUT / "src"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base",
        revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        trust_remote_code=False,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    parent = locate_parent()
    environment = {"PYTHONPATH": str(CHECKOUT / "src"), "TOKENIZERS_PARALLELISM": "false"}
    results = []
    for sequence_length, geometry in GEOMETRY.items():
        corpus, corpus_metadata = prepare_probe_corpus(
            sequence_length, geometry["accumulation"], tokenizer
        )
        attempts = []
        for memory_retry in (False, True):
            name = f"seq-{sequence_length:05d}-{'retry' if memory_retry else 'known'}"
            destination = OUTPUT / "runs" / name
            code, elapsed, output = command_run(
                train_command(
                    corpus,
                    parent,
                    destination,
                    sequence_length,
                    geometry["accumulation"],
                    memory_retry=memory_retry,
                ),
                name=name,
                env=environment,
            )
            summary_path = destination / "summary.json"
            attempt = {
                "memory_retry": memory_retry,
                "exit_code": code,
                "elapsed_seconds": elapsed,
                "oom": "out of memory" in output.lower(),
                "summary": (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.exists()
                    else None
                ),
            }
            attempts.append(attempt)
            if code == 0 or not attempt["oom"]:
                break
        results.append(
            {
                "sequence_length": sequence_length,
                **geometry,
                "microbatch_per_gpu": 1,
                "corpus": corpus_metadata,
                "attempts": attempts,
                "feasible": any(attempt["exit_code"] == 0 for attempt in attempts),
            }
        )
        (OUTPUT / "probe_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("long-context training probes complete")


if __name__ == "__main__":
    main()
