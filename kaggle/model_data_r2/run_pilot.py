"""One bounded R2 arm, using the existing FSDP1 trainer and restart format."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "__CHECKOUT_COMMIT__"
ARM = "__ARM__"
SESSION_SECONDS = 9900
START = time.time()
DEADLINE = START + SESSION_SECONDS - 900
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/model_data_r2_pilot")
P12_SHA = "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43"
TOKENIZER_SHA = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
TOKENS = 32768


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command, name, environment=None):
    remaining = DEADLINE - time.time()
    if remaining < 60:
        raise TimeoutError("finalization reserve reached")
    started = time.time()
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=remaining,
        env={**os.environ, **(environment or {})},
        cwd=ROOT if ROOT.exists() else None,
    )
    (OUT / (name + ".log")).write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(name + " failed; inspect private job artifacts")
    return time.time() - started


def verify_state(directory, updates):
    summary = read(directory / "summary.json")
    assert summary["optimizer_steps"] == updates
    assert summary["training_tokens"] == updates * TOKENS
    assert summary["corpus_position"]["next_block"] == updates * 16
    assert summary["resume_state_saved"] and not summary["nan_or_inf"]
    assert summary["skipped_optimizer_updates_this_run"] == 0
    state = directory / "resume-latest"
    assert read(state / "COMPLETE.json")["complete"]
    manifest = read(state / "checkpoint_manifest.json")
    for entry in manifest["files"]:
        path = state / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert sha(path) == entry["sha256"]
    save(directory / "verified-checkpoint-manifest.json", manifest)
    return summary


def train(corpus, parent, destination, updates, environment, baseline, resume=None):
    destination.mkdir(parents=True, exist_ok=True)
    save(
        destination / "observability-run.json",
        {
            "campaign_id": "tabcomplete-model-data-r2",
            "run_id": "r2-" + ARM.lower() + "-" + destination.name,
        },
    )
    # The scientific token cap is absolute; resumed work does not replay the prefix.
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(ROOT / "scripts/r2_training_worker.py"),
        "train",
        "--corpus-dir",
        str(corpus),
        "--output-dir",
        str(destination),
        "--learning-rate",
        "3e-6",
        "--lr-schedule",
        "cosine",
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
        str(updates * TOKENS),
        "--workers",
        "1",
        "--warmup-steps",
        "5",
        "--seed",
        "928173",
        "--parent-training-tokens",
        "11993088",
        "--distributed-mode",
        "fsdp",
        "--save-resume",
        "--baseline-path",
        str(baseline),
        "--deadline-seconds",
        str(max(1, DEADLINE - time.time() - 1200)),
    ]
    if resume:
        command += ["--resume-from", str(resume)]
    else:
        command += ["--init-from", str(parent), "--expected-initial-sha256", P12_SHA]
    if updates in (3, 153):
        command += ["--save-final"]
    if updates == 153:
        command += ["--evaluation-milestones", str(31 * TOKENS), str(77 * TOKENS)]
    run(
        command,
        destination.name,
        {
            **environment,
            "TABCOMPLETE_RUN_ID": "r2-" + ARM.lower() + "-" + destination.name,
            "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(destination / "telemetry.jsonl"),
        },
    )
    return verify_state(destination, updates)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert ARM in ("R2_STANDARD", "R2_FILTERED")
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
            "opentelemetry-api==1.44.0",
            "opentelemetry-sdk==1.44.0",
            "opentelemetry-exporter-otlp-proto-http==1.44.0",
        ],
        "setup",
    )
    run(
        [
            "git",
            "clone",
            "--branch",
            "research/model-data-r2",
            "https://github.com/Shlok-Bhakta/tabcomplete.git",
            str(ROOT),
        ],
        "clone",
    )
    run(["git", "checkout", COMMIT], "checkout")
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from tinycomplete.code_cpt.prepare import corpus_fingerprint

    parents = list(Path("/kaggle/input").glob("**/P12/model.safetensors"))
    assert len(parents) == 1
    parent = parents[0].parent
    assert sha(parent / "model.safetensors") == P12_SHA
    assert sha(parent / "tokenizer.json") == TOKENIZER_SHA
    corpora = list(Path("/kaggle/input").glob("**/" + ARM + "/corpus_metadata.json"))
    assert len(corpora) == 1
    corpus = corpora[0].parent
    metadata = read(corpora[0])
    assert metadata["actual_train_tokens"] == 5013504
    assert metadata["packing_seed"] == 928173
    assert (
        corpus_fingerprint(
            [
                corpus / "train_blocks.npy",
                corpus / "train_languages.npy",
                *list((corpus / "manifests").glob("*.jsonl")),
            ]
        )
        == (metadata["corpus_fingerprint"])
    )
    if shutil.disk_usage(OUT).free < 18 * 2**30:
        raise RuntimeError("18 GiB free disk required for verified pilot finalization")
    environment = {
        "PYTHONPATH": str(ROOT / "src"),
        "TOKENIZERS_PARALLELISM": "false",
        "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
        "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
        "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "1",
        "TABCOMPLETE_OBSERVABILITY_ARTIFACT_ROOT": str(OUT / "captured"),
        "TABCOMPLETE_CAMPAIGN_ID": "tabcomplete-model-data-r2",
    }
    progress = {
        "schema_version": 1,
        "arm": ARM,
        "git_sha": COMMIT,
        "started_unix": START,
        "session_deadline_seconds": SESSION_SECONDS,
        "plan_sha256": sha(ROOT / "reports/research/model_data_r2/preregistered_plan.json"),
        "corpus": metadata,
        "parent_sha256": P12_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "hardware": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "training_input_tokens": 0,
        "status": "running",
    }
    save(OUT / "progress.json", progress)
    parent_eval = OUT / "parent-fresh.json"
    run(
        [
            sys.executable,
            "-m",
            "tinycomplete.code_cpt.train",
            "checkpoint-eval",
            "--checkpoint",
            str(parent),
            "--corpus-dir",
            str(corpus),
            "--output",
            str(parent_eval),
            "--expected-sha256",
            P12_SHA,
        ],
        "parent-eval",
        {
            **environment,
            "CUDA_VISIBLE_DEVICES": "0",
            "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(OUT / "parent-telemetry.jsonl"),
        },
    )
    baseline = OUT / "baseline.json"
    save(baseline, read(parent_eval)["metrics"])
    smoke = train(corpus, parent, OUT / "smoke", 3, environment, baseline)
    progress["training_input_tokens"] += smoke["additional_input_tokens"]
    assert (OUT / "smoke/final/model.safetensors").stat().st_size > 1_000_000_000
    if ARM == "R2_STANDARD":
        continuous = train(corpus, parent, OUT / "continuous", 8, environment, baseline)
        resumed = train(
            corpus, parent, OUT / "resumed", 8, environment, baseline, OUT / "smoke/resume-latest"
        )
        progress["training_input_tokens"] += (
            continuous["additional_input_tokens"] + resumed["additional_input_tokens"]
        )
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
                str(OUT / "continuous/resume-latest"),
                "--checkpoint-b",
                str(OUT / "resumed/resume-latest"),
                "--output",
                str(OUT / "restart-comparison.json"),
                "--lr-schedule",
                "cosine",
            ],
            "restart-compare",
            environment,
        )
        comparison = read(OUT / "restart-comparison.json")
        differences = [
            abs(a["loss"] - b["loss"])
            for a, b in zip(
                continuous["loss_history"],
                smoke["loss_history"] + resumed["loss_history"],
                strict=True,
            )
        ]
        assert max(differences) <= 0.0001
        for rank in comparison["ranks"]:
            assert all(
                rank[key]
                for key in (
                    "metadata_equal",
                    "optimizer_structure_equal",
                    "scheduler_equal",
                    "scaler_equal",
                )
            )
            assert rank["model"]["max_abs_difference"] <= 0.0005
            assert rank["model"]["mean_abs_difference"] <= 0.0000001
            assert rank["optimizer"]["max_abs_difference"] <= 0.0005
        save(
            OUT / "restart-verification.json",
            {
                "continuous_updates": 8,
                "split": [3, 5],
                "next_consumed_block": resumed["corpus_position"]["next_block"],
                "additional_test_tokens": progress["training_input_tokens"],
                "comparison": comparison,
                "loss_absolute_differences": differences,
                "numerical_gate_passed": True,
            },
        )
    # Only campaign-owned smoke/restart weights, after their complete manifests and
    # semantic comparison are preserved. No research checkpoint is removed.
    for name in ("smoke", "continuous", "resumed"):
        for component in ("resume-latest", "final"):
            path = OUT / name / component
            if path.exists():
                if component == "final":
                    save(
                        OUT / name / "inference-verification.json",
                        {
                            "model_sha256": sha(path / "model.safetensors"),
                            "tokenizer_sha256": sha(path / "tokenizer.json"),
                        },
                    )
                shutil.rmtree(path)
    progress["smoke_verified"] = True
    save(OUT / "progress.json", progress)
    if DEADLINE - time.time() < 90 * 60:
        raise TimeoutError("insufficient remaining time to start full pilot safely")
    final = train(corpus, parent, OUT / ARM, 153, environment, baseline)
    progress["training_input_tokens"] += final["additional_input_tokens"]
    progress.update(
        status="complete",
        elapsed_seconds=time.time() - START,
        final_model_sha256=sha(OUT / ARM / "final/model.safetensors"),
        final_resume_verified=True,
        scientific_tokens=final["additional_input_tokens"],
    )
    assert final["successful_optimizer_updates"] == 153
    assert final["additional_input_tokens"] == 5013504
    save(OUT / "progress.json", progress)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save(
            OUT / "failure.json",
            {"type": type(error).__name__, "elapsed_seconds": time.time() - START, "arm": ARM},
        )
        raise
