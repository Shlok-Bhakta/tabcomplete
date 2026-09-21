"""Matched four-arm causal CPT pilot on the frozen research-r1 corpus."""

from __future__ import annotations

import hashlib
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
OUTPUT = Path("/kaggle/working/code_cpt_campaign_r1")
TOKENS_PER_UPDATE = 32_768
PILOT_UPDATES = 153
PILOT_TOKENS = PILOT_UPDATES * TOKENS_PER_UPDATE
SESSION_LIMIT_SECONDS = 9 * 60 * 60
FINALIZATION_RESERVE_SECONDS = 30 * 60
INITIAL_ARM_ESTIMATE_SECONDS = 75 * 60
PARENTS = {
    "P5": {
        "tokens": 5_013_504,
        "sha256": "d499c3fe2d720246c710c5c9e20653a4599aa25d77ebef871cb28431ccc888e8",
    },
    "P12": {
        "tokens": 11_993_088,
        "sha256": "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43",
    },
}
ARMS = [
    ("C12", "P12", "constant"),
    ("C5", "P5", "constant"),
    ("D12", "P12", "cosine"),
    ("D5", "P5", "cosine"),
]


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
        raise RuntimeError(f"{name} failed with exit code {process.returncode}; see {log}")
    return elapsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_inputs() -> tuple[Path, Path, dict[str, Path]]:
    fresh = []
    historical = []
    for metadata_path in Path("/kaggle/input").glob("**/corpus_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tokens = metadata.get("actual_train_tokens")
        if tokens == 5_013_504 and metadata.get("campaign") == "code_cpt_research_r1":
            fresh.append(metadata_path.parent)
        elif tokens == 11_999_232 and metadata.get("seed") == 314159:
            historical.append(metadata_path.parent)
    parents = {}
    for name, specification in PARENTS.items():
        matches = []
        for metadata_path in Path("/kaggle/input").glob(f"**/{name}/training_metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("training_tokens") == specification["tokens"]:
                matches.append(metadata_path.parent)
        if len(matches) != 1:
            raise RuntimeError(f"expected one {name} parent, found {matches}")
        parents[name] = matches[0]
    if len(fresh) != 1 or len(historical) != 1:
        raise RuntimeError(f"expected one fresh and historical corpus, got {fresh} and {historical}")
    return fresh[0], historical[0], parents


def evaluate_parent(
    name: str,
    checkpoint: Path,
    corpus: Path,
    output: Path,
    expected_sha256: str,
    environment: dict[str, str],
) -> float:
    return run(
        [
            sys.executable,
            "-m",
            "tinycomplete.code_cpt.train",
            "checkpoint-eval",
            "--checkpoint",
            str(checkpoint),
            "--corpus-dir",
            str(corpus),
            "--output",
            str(output),
            "--expected-sha256",
            expected_sha256,
        ],
        name=name,
        env={**environment, "CUDA_VISIBLE_DEVICES": "0"},
    )


def train_arm(
    name: str,
    parent_name: str,
    schedule: str,
    parent: Path,
    fresh_corpus: Path,
    baseline: Path,
    environment: dict[str, str],
) -> float:
    specification = PARENTS[parent_name]
    destination = OUTPUT / "arms" / name
    return run(
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
            str(fresh_corpus),
            "--output-dir",
            str(destination),
            "--init-from",
            str(parent),
            "--expected-initial-sha256",
            specification["sha256"],
            "--parent-training-tokens",
            str(specification["tokens"]),
            "--learning-rate",
            "3e-6",
            "--lr-schedule",
            schedule,
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
            str(PILOT_TOKENS),
            "--workers",
            "1",
            "--warmup-steps",
            "5",
            "--distributed-mode",
            "fsdp",
            "--evaluation-milestones",
            str(31 * TOKENS_PER_UPDATE),
            str(77 * TOKENS_PER_UPDATE),
            "--baseline-path",
            str(baseline),
            "--deadline-seconds",
            str(105 * 60),
            "--save-final",
        ],
        name=f"arm-{name}",
        env=environment,
    )


def write_progress(value: dict) -> None:
    (OUTPUT / "campaign_progress.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    initial_free = shutil.disk_usage(OUTPUT).free
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
    fresh, historical, parents = locate_inputs()
    environment = {"PYTHONPATH": str(CHECKOUT / "src"), "TOKENIZERS_PARALLELISM": "false"}
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True
    ).strip()
    fresh_metadata = json.loads((fresh / "corpus_metadata.json").read_text(encoding="utf-8"))
    progress = {
        "schema_version": 1,
        "status": "running",
        "git_sha": git_sha,
        "started_unix": started,
        "initial_free_bytes": initial_free,
        "corpus_fingerprint": fresh_metadata["corpus_fingerprint"],
        "pilot_tokens_per_arm": PILOT_TOKENS,
        "completed_arms": [],
        "skipped_arms": [],
    }
    write_progress(progress)

    for parent_name, parent in parents.items():
        destination = OUTPUT / "parents" / parent_name
        evaluate_parent(
            f"parent-{parent_name}-fresh",
            parent,
            fresh,
            destination / "fresh_development.json",
            PARENTS[parent_name]["sha256"],
            environment,
        )
        evaluate_parent(
            f"parent-{parent_name}-historical",
            parent,
            historical,
            destination / "historical_micro.json",
            PARENTS[parent_name]["sha256"],
            environment,
        )

    arm_estimate = INITIAL_ARM_ESTIMATE_SECONDS
    for name, parent_name, schedule in ARMS:
        elapsed = time.time() - started
        if elapsed + arm_estimate + FINALIZATION_RESERVE_SECONDS > SESSION_LIMIT_SECONDS:
            progress["skipped_arms"].append(
                {"name": name, "reason": "session wall-time reserve"}
            )
            write_progress(progress)
            continue
        if shutil.disk_usage(OUTPUT).free < 4 * 2**30:
            progress["skipped_arms"].append({"name": name, "reason": "save-space reserve"})
            write_progress(progress)
            continue
        baseline = OUTPUT / "parents" / parent_name / "fresh_development.json"
        arm_elapsed = train_arm(
            name,
            parent_name,
            schedule,
            parents[parent_name],
            fresh,
            baseline,
            environment,
        )
        arm_estimate = max(arm_estimate, arm_elapsed * 1.10)
        summary = json.loads(
            (OUTPUT / "arms" / name / "summary.json").read_text(encoding="utf-8")
        )
        final_model = OUTPUT / "arms" / name / "final" / "model.safetensors"
        record = {
            "name": name,
            "parent": parent_name,
            "lr_schedule": schedule,
            "elapsed_seconds": arm_elapsed,
            "additional_input_tokens": summary["additional_input_tokens"],
            "successful_optimizer_updates": summary["successful_optimizer_updates"],
            "steady_state_tokens_per_second": summary["steady_state_tokens_per_second"],
            "stop_reason": summary["stop_reason"],
            "model_sha256": sha256_file(final_model),
            "model_bytes": final_model.stat().st_size,
        }
        progress["completed_arms"].append(record)
        write_progress(progress)
        evaluate_parent(
            f"arm-{name}-historical",
            OUTPUT / "arms" / name / "final",
            historical,
            OUTPUT / "arms" / name / "historical_micro.json",
            record["model_sha256"],
            environment,
        )

    progress.update(
        status="complete",
        elapsed_seconds=time.time() - started,
        final_free_bytes=shutil.disk_usage(OUTPUT).free,
    )
    write_progress(progress)
    print("matched CPT campaign complete")


if __name__ == "__main__":
    main()
