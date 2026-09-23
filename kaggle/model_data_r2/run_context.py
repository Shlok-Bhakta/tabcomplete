"""One-hour inference-only context diagnostic; existing fixture and exact models."""

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
DEADLINE = START + 3600 - 900
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/model_data_r2_context")
LOCK = threading.Lock()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def run(command, name, env=None):
    remaining = DEADLINE - time.time()
    if remaining < 60:
        raise TimeoutError("context finalization reserve reached")
    result = subprocess.run(
        command,
        cwd=ROOT if ROOT.exists() else None,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=remaining,
    )
    (OUT / (name + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(name + " failed; see private log")


def one(pattern):
    paths = list(Path("/kaggle/input").glob(pattern))
    assert len(paths) == 1, "ambiguous context input"
    return paths[0]


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
    for alias, model, revision in [
        ("q35-base", "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
        ("q25-coder", "Qwen/Qwen2.5-Coder-0.5B", "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"),
    ]:
        models[alias] = Path(
            snapshot_download(
                model,
                revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
            )
        )
    assert sha(models["q35-p12"] / "model.safetensors") == (
        "d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43"
    )
    assert sha(models["q35-d12"] / "model.safetensors") == (
        "ce0705a6ca265ee40eb7c65b4831b938d6af37c1b508d68cdace2b02ec6bdd8e"
    )
    suite = one("**/code_cpt_long_context_r1/long_context_v2.jsonl")
    assert sha(suite) == "76825fa15d5b7c914739c81e14dd4ece1cfc9442d1278bb2861db7fd9f07cbbd"
    progress = {
        "git_sha": COMMIT,
        "suite_sha256": sha(suite),
        "status": "running",
        "training_input_tokens": 0,
        "results": [],
        "planned_models": list(models),
    }

    def lane(gpu, aliases):
        for alias in aliases:
            remaining = DEADLINE - time.time()
            if remaining < 300:
                break
            environment = {
                "PYTHONPATH": str(ROOT / "src"),
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
                "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
                "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "1",
                "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(OUT / (alias + "-telemetry.jsonl")),
                "TABCOMPLETE_OBSERVABILITY_ARTIFACT_ROOT": str(OUT / "captured"),
                "TABCOMPLETE_CAMPAIGN_ID": "tabcomplete-model-data-r2",
                "TABCOMPLETE_RUN_ID": "r2-context-" + alias,
            }
            started = time.time()
            try:
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/evaluate_r2_context.py"),
                        "--model",
                        str(models[alias]),
                        "--suite",
                        str(suite),
                        "--alias",
                        alias,
                        "--output",
                        str(OUT / alias),
                        "--seconds",
                        str(int(min(1500, remaining - 60))),
                    ],
                    alias,
                    environment,
                )
                result = {"model": alias, "status": "finished", "gpu_lane": gpu}
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
                result = {
                    "model": alias,
                    "status": "runtime_blocked",
                    "type": type(error).__name__,
                    "quality_score": None,
                    "gpu_lane": gpu,
                    "log": alias + ".log",
                }
            result["seconds"] = time.time() - started
            with LOCK:
                progress["results"].append(result)
                (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(lane, 0, ["q35-base", "q35-d12"]),
            pool.submit(lane, 1, ["q35-p12", "q25-coder"]),
        ]
        for job in jobs:
            job.result()
    progress.update(status="terminal", elapsed_seconds=time.time() - START)
    (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")


if __name__ == "__main__":
    main()
