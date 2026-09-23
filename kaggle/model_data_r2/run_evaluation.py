"""Two isolated inference lanes in one bounded T4x2 allocation; no training."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMIT = "__CHECKOUT_COMMIT__"
START = time.time()
SESSION_SECONDS = 9000
DEADLINE = START + SESSION_SECONDS - 900
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/model_data_r2_evaluation")
LOCK = threading.Lock()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def run(command, name, environment=None, limit=2700):
    remaining = min(limit, DEADLINE - time.time())
    if remaining < 60:
        raise TimeoutError("finalization reserve reached")
    print(json.dumps({"stage": name, "state": "started"}), flush=True)
    started = time.time()
    result = subprocess.run(
        command,
        cwd=ROOT if ROOT.exists() else None,
        capture_output=True,
        text=True,
        timeout=remaining,
        env={**os.environ, **(environment or {})},
    )
    (OUT / (name + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(name + " failed; see private log")
    print(
        json.dumps({"stage": name, "state": "completed", "seconds": time.time() - started}),
        flush=True,
    )
    return time.time() - started


def one(pattern):
    matches = list(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise ValueError("ambiguous input pattern: " + pattern)
    return matches[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==5.5.0",
            "accelerate==1.13.0",
            "flash-linear-attention==0.5.2",
            "tree-sitter==0.25.2",
            "tree-sitter-language-pack",
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
    from huggingface_hub import snapshot_download

    models = {
        "q35-p12": one("**/P12/model.safetensors").parent,
        "q35-d12": one("**/arms/D12/final/model.safetensors").parent,
    }
    expected = {
        "q35-p12": "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43",
        "q35-d12": "ce0705a6ca265ee40eb7c65b4831b938d6af37c1b508d68cdace2b02ec6bdd8e",
    }
    for alias in ("R2_STANDARD", "R2_FILTERED"):
        models[alias] = one("**/" + alias + "/final/model.safetensors").parent
        summary = json.loads((models[alias].parent / "summary.json").read_text())
        assert summary["additional_input_tokens"] == 5013504 and summary["optimizer_steps"] == 153
        assert summary["resume_state_saved"]
    for alias, model, revision in [
        ("q35-base", "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
        ("q25-coder", "Qwen/Qwen2.5-Coder-0.5B", "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"),
        (
            "granite-h350",
            "ibm-granite/granite-4.0-h-350m-base",
            "dc555b6939c863bb96034d1eae7601da36bd42e4",
        ),
    ]:
        models[alias] = Path(
            snapshot_download(
                model,
                revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
            )
        )
    inventory = {}
    for alias, path in models.items():
        hashed = sha(path / "model.safetensors")
        if alias in expected:
            assert hashed == expected[alias]
        inventory[alias] = {
            "weights_sha256": hashed,
            "tokenizer_sha256": sha(path / "tokenizer.json"),
            "config_sha256": sha(path / "config.json"),
        }
    save(OUT / "model-inventory.json", inventory)
    line_suite = one("**/tabcomplete-model-data-r2-inputs/causal_line_v1.jsonl")
    assert sha(line_suite) == "d614a7354d3341f0cc3667828c970d47f5646ec046029241d6c91a34411f17dd"
    corpus = one("**/tabcomplete-model-data-r2-inputs/R2_STANDARD/corpus_metadata.json").parent
    historical = one("**/tabcomplete-model-data-r2-inputs/historical/corpus_metadata.json").parent
    progress = {
        "schema_version": 1,
        "git_sha": COMMIT,
        "started_unix": START,
        "training_input_tokens": 0,
        "status": "running",
        "completed": [],
        "failures": [],
        "planned_phases": 17,
    }

    def phase(alias, label, command, gpu):
        environment = {
            "PYTHONPATH": str(ROOT / "src"),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
            "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
            "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "1",
            "TABCOMPLETE_OBSERVABILITY_ARTIFACT_ROOT": str(OUT / "captured"),
            "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(
                OUT / f"{alias}-{label}-telemetry.jsonl"
            ),
            "TABCOMPLETE_CAMPAIGN_ID": "tabcomplete-model-data-r2",
            "TABCOMPLETE_RUN_ID": f"r2-{alias}-{label}",
        }
        name = alias + "-" + label
        try:
            elapsed = run(command, name, environment)
            event = {"model": alias, "phase": label, "gpu_lane": gpu, "seconds": elapsed}
            key = "completed"
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            event = {
                "model": alias,
                "phase": label,
                "gpu_lane": gpu,
                "type": type(error).__name__,
                "quality_score": None,
                "log": name + ".log",
            }
            key = "failures"
        with LOCK:
            progress[key].append(event)
            save(OUT / "progress.json", progress)

    def lane(gpu, aliases):
        for alias in aliases:
            if time.time() > DEADLINE - 300:
                break
            model = models[alias]
            if alias not in ("q35-p12", "q25-coder", "q35-d12"):
                phase(
                    alias,
                    "causal",
                    [
                        sys.executable,
                        str(ROOT / "scripts/generate_code_predictions.py"),
                        "--suite",
                        str(ROOT / "data/benchmarks/code_completion_v2.jsonl"),
                        "--model-path",
                        str(model),
                        "--model-label",
                        alias,
                        "--max-new-tokens",
                        "96",
                        "--device",
                        "cuda:0",
                        "--workers",
                        "1",
                        "--output",
                        str(OUT / alias / "causal.jsonl"),
                    ],
                    gpu,
                )
            phase(
                alias,
                "line",
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_causal_line.py"),
                    "--suite",
                    str(line_suite),
                    "--model",
                    str(model),
                    "--alias",
                    alias,
                    "--output",
                    str(OUT / alias / "line"),
                ],
                gpu,
            )
            if alias in ("R2_STANDARD", "R2_FILTERED", "q35-base"):
                for name, data, repository_only in (
                    ("fresh", corpus, alias != "q35-base"),
                    ("historical", historical, False),
                ):
                    command = [
                        sys.executable,
                        "-m",
                        "tinycomplete.code_cpt.train",
                        "checkpoint-eval",
                        "--checkpoint",
                        str(model),
                        "--corpus-dir",
                        str(data),
                        "--output",
                        str(OUT / alias / (name + ".json")),
                        "--expected-sha256",
                        inventory[alias]["weights_sha256"],
                    ]
                    if repository_only:
                        command.append("--repository-only")
                    phase(alias, name, command, gpu)

    # Exactly one sequential queue owns each visible GPU; task completion cannot
    # accidentally place a second worker on the other lane's occupied device.
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(lane, 0, ["q35-p12", "R2_STANDARD", "q35-base"]),
            pool.submit(lane, 1, ["q25-coder", "R2_FILTERED", "q35-d12", "granite-h350"]),
        ]
        for job in jobs:
            job.result()
    progress.update(
        status="complete"
        if not progress["failures"] and len(progress["completed"]) == progress["planned_phases"]
        else "partial",
        unexecuted_phases=progress["planned_phases"]
        - len(progress["completed"])
        - len(progress["failures"]),
        elapsed_seconds=time.time() - START,
    )
    save(OUT / "progress.json", progress)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save(
            OUT / "failure.json",
            {"type": type(error).__name__, "elapsed_seconds": time.time() - START},
        )
        raise
