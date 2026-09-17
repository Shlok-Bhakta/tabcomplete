"""Kaggle T4x2 orchestrator for Stage-1 profiling, LR selection, and main CPT."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY = "https://github.com/Shlok-Bhakta/tabcomplete.git"
BRANCH = "stage1/code-cpt"
CHECKOUT = Path("/kaggle/working/tabcomplete")
OUTPUT = Path("/kaggle/working/code_cpt_run")


def run(command: list[str], *, env: dict[str, str] | None = None, log: Path | None = None) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    process = subprocess.run(command, env=merged, text=True, capture_output=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(process.stdout + process.stderr, encoding="utf-8")
    print("command exit:", process.returncode, "log:", log)
    return process.returncode


def find_corpus() -> Path:
    candidates = list(Path("/kaggle/input").glob("**/corpus_metadata.json"))
    if not candidates:
        raise FileNotFoundError("attached prepared corpus is missing")
    return candidates[0].parent


def launch_training(
    *,
    name: str,
    corpus: Path,
    gpus: int,
    microbatch: int,
    accumulation: int,
    checkpointing: bool,
    optimizer: str,
    learning_rate: float,
    max_tokens: int,
    workers: int = 1,
    eval_final: bool = False,
    main_run: bool = False,
) -> dict:
    destination = OUTPUT / name
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        "tinycomplete.code_cpt.train",
        "train",
        "--corpus-dir",
        str(corpus),
        "--output-dir",
        str(destination),
        "--learning-rate",
        str(learning_rate),
        "--microbatch",
        str(microbatch),
        "--gradient-accumulation",
        str(accumulation),
        "--optimizer",
        optimizer,
        "--max-tokens",
        str(max_tokens),
        "--workers",
        str(workers),
        "--prefetch-factor",
        "2",
        "--warmup-steps",
        "5",
    ]
    command.append("--gradient-checkpointing" if checkpointing else "--no-gradient-checkpointing")
    if eval_final:
        command.append("--eval-final")
    if main_run:
        command.extend(
            [
                "--deadline-seconds",
                "16200",
                "--save-final",
                "--save-resume",
                "--milestones",
                "1000000",
                "2500000",
                "5000000",
                "10000000",
            ]
        )
    started = time.time()
    code = run(
        command,
        env={
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(gpus)),
            "PYTHONPATH": str(CHECKOUT / "src"),
            "TOKENIZERS_PARALLELISM": "false",
        },
        log=destination / "process.log",
    )
    record = {
        "name": name,
        "exit_code": code,
        "elapsed_seconds_including_load": time.time() - started,
        "gpus": gpus,
        "microbatch_per_gpu": microbatch,
        "gradient_accumulation": accumulation,
        "checkpointing": checkpointing,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "requested_tokens": max_tokens,
        "workers": workers,
        "stable": False,
    }
    summary_path = destination / "summary.json"
    if code == 0 and summary_path.exists():
        record.update(json.loads(summary_path.read_text(encoding="utf-8")))
        record["stable"] = not record.get("nan_or_inf", True)
    return record


def pick_configuration(benchmarks: list[dict]) -> dict:
    stable = [row for row in benchmarks if row.get("stable")]
    if not stable:
        raise RuntimeError("no full-weight benchmark configuration was stable")
    return max(stable, key=lambda row: row["tokens_per_second"])


def pick_learning_rate(baseline: dict, pilots: list[dict]) -> dict:
    base = baseline["metrics"]
    eligible = []
    for pilot in pilots:
        evaluation_path = OUTPUT / pilot["name"] / "micro_eval.json"
        if not pilot.get("stable") or not evaluation_path.exists():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        regressions = sum(
            evaluation[language]["nll"] > base[language]["nll"] * 1.02
            for language in (
                "python",
                "typescript",
                "javascript",
                "java",
                "cpp",
                "rust",
                "go",
                "c",
                "csharp",
            )
        )
        general_ratio = evaluation["general"]["nll"] / base["general"]["nll"]
        pilot["micro_eval"] = evaluation
        pilot["language_regressions_over_2pct"] = regressions
        pilot["general_nll_ratio"] = general_ratio
        if regressions <= 2 and general_ratio <= 1.10:
            eligible.append(pilot)
    if not eligible:
        raise RuntimeError("all learning-rate pilots failed broad-improvement safety gates")
    return min(eligible, key=lambda row: row["micro_eval"]["overall_code"]["nll"])


def write_baseline_markdown(baseline: dict, path: Path) -> None:
    labels = {
        "python": "Python",
        "typescript": "TypeScript",
        "javascript": "JavaScript",
        "java": "Java",
        "cpp": "C++",
        "rust": "Rust",
        "go": "Go",
        "c": "C",
        "csharp": "C#",
        "overall_code": "Overall code",
        "general": "General text",
    }
    lines = [
        "# Untouched Qwen3.5-0.8B-Base MICRO baseline",
        "",
        f"Model revision: `{baseline['model_revision']}`",
        f"Tokenizer revision: `{baseline['tokenizer_revision']}`",
        f"Precision: {baseline['precision']}",
        f"Seed: {baseline['seed']}",
        f"GPU: {baseline['gpu']}",
        "",
        "| Corpus | NLL | Evaluated tokens |",
        "|---|---:|---:|",
    ]
    for key, label in labels.items():
        metric = baseline["metrics"][key]
        lines.append(f"| {label} | {metric['nll']:.6f} | {metric['tokens']} |")
    lines.extend(["", "Package versions:", ""])
    for package, version in sorted(baseline["packages"].items()):
        lines.append(f"- `{package}=={version}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    corpus = find_corpus()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==5.5.0",
            "accelerate",
            "bitsandbytes",
        ]
    )
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)])
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    import torch
    import transformers

    environment = {
        "git_sha": git_sha,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    (OUTPUT / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Stage-1 kernel requires exactly two Kaggle T4 GPUs")

    baseline_path = OUTPUT / "baseline.json"
    baseline_code = run(
        [
            sys.executable,
            "-m",
            "tinycomplete.code_cpt.train",
            "baseline",
            "--corpus-dir",
            str(corpus),
            "--output",
            str(baseline_path),
        ],
        env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(CHECKOUT / "src")},
        log=OUTPUT / "baseline.log",
    )
    if baseline_code:
        raise RuntimeError("untouched-base MICRO evaluation failed")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    write_baseline_markdown(baseline, OUTPUT / "baseline.md")

    benchmark_specs = [
        ("1gpu_mb1_ckpt_8bit", 1, 1, 16, True, "adamw_8bit", 1),
        ("2gpu_mb1_ckpt_8bit", 2, 1, 8, True, "adamw_8bit", 1),
        ("2gpu_mb1_no_ckpt_8bit", 2, 1, 8, False, "adamw_8bit", 1),
        ("2gpu_mb2_no_ckpt_8bit", 2, 2, 4, False, "adamw_8bit", 1),
        ("2gpu_mb4_no_ckpt_8bit", 2, 4, 2, False, "adamw_8bit", 1),
        ("2gpu_mb8_no_ckpt_8bit", 2, 8, 1, False, "adamw_8bit", 1),
        ("2gpu_mb1_ckpt_torch", 2, 1, 8, True, "adamw_torch", 1),
    ]
    benchmarks = []
    for name, gpus, microbatch, accumulation, checkpointing, optimizer, workers in benchmark_specs:
        benchmarks.append(
            launch_training(
                name=f"benchmark/{name}",
                corpus=corpus,
                gpus=gpus,
                microbatch=microbatch,
                accumulation=accumulation,
                checkpointing=checkpointing,
                optimizer=optimizer,
                learning_rate=1e-5,
                max_tokens=65_536,
                workers=workers,
            )
        )
    selected = pick_configuration(benchmarks)
    for workers in (2, 3):
        benchmarks.append(
            launch_training(
                name=f"benchmark/selected_workers{workers}",
                corpus=corpus,
                gpus=selected["gpus"],
                microbatch=selected["microbatch_per_gpu"],
                accumulation=selected["gradient_accumulation"],
                checkpointing=selected["checkpointing"],
                optimizer=selected["optimizer"],
                learning_rate=1e-5,
                max_tokens=65_536,
                workers=workers,
            )
        )
    selected = pick_configuration(benchmarks)
    (OUTPUT / "benchmarks.json").write_text(
        json.dumps(benchmarks, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    tokens_per_update = int(selected["tokens_per_update"])
    measured = float(selected["tokens_per_second"])
    pilot_tokens = max(131_072, min(524_288, int(measured * 900)))
    pilot_tokens = math.ceil(pilot_tokens / tokens_per_update) * tokens_per_update
    pilots = []
    for learning_rate in (3e-6, 1e-5, 3e-5):
        pilots.append(
            launch_training(
                name=f"pilots/lr_{learning_rate:g}",
                corpus=corpus,
                gpus=selected["gpus"],
                microbatch=selected["microbatch_per_gpu"],
                accumulation=selected["gradient_accumulation"],
                checkpointing=selected["checkpointing"],
                optimizer=selected["optimizer"],
                learning_rate=learning_rate,
                max_tokens=pilot_tokens,
                workers=selected["workers"],
                eval_final=True,
            )
        )
    winning_pilot = pick_learning_rate(baseline, pilots)
    (OUTPUT / "lr_sweep.json").write_text(
        json.dumps(pilots, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    corpus_metadata = json.loads((corpus / "corpus_metadata.json").read_text(encoding="utf-8"))
    main_result = launch_training(
        name="main",
        corpus=corpus,
        gpus=selected["gpus"],
        microbatch=selected["microbatch_per_gpu"],
        accumulation=selected["gradient_accumulation"],
        checkpointing=selected["checkpointing"],
        optimizer=selected["optimizer"],
        learning_rate=winning_pilot["learning_rate"],
        max_tokens=int(corpus_metadata["actual_train_tokens"]),
        workers=selected["workers"],
        main_run=True,
    )
    final = {
        "environment": environment,
        "selected_configuration": selected,
        "pilot_tokens": pilot_tokens,
        "selected_learning_rate": winning_pilot["learning_rate"],
        "main": main_result,
    }
    (OUTPUT / "orchestrator_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    if not main_result.get("stable"):
        raise RuntimeError("main training run did not complete stably")
    print("Stage-1 orchestration: PASS")


if __name__ == "__main__":
    main()
