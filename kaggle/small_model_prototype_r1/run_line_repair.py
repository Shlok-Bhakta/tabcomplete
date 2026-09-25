"""Versioned retry of only the missing corrected line phases."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
DEADLINE = START + 3600 - 900
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/small_model_prototype_r1_line_repair")
COMMIT = "__CHECKOUT_COMMIT__"
PLAN_SHA = "__PLAN_SHA__"
MODELS = [
    ("q3-base", "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
     "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
    ("lfm350-base", "LiquidAI/LFM2.5-350M-Base", "9960764e30892e01f29a6dc23df2533fcd8bd5ae",
     "af70818c41a5cdb3f9587f91de12ff5f7847b8b0a2ba734534205ccea1d98aba"),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args, label, env=None):
    remaining = DEADLINE - time.monotonic()
    if remaining < 120:
        raise TimeoutError("finalization reserve reached")
    result = subprocess.run(args, cwd=ROOT if ROOT.exists() else None,
                            env={**os.environ, **(env or {})}, capture_output=True, text=True,
                            timeout=remaining)
    (OUT / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private log")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(OUT / "hf-cache")
    state = {"status": "setup", "models": {}, "plan_sha256": PLAN_SHA,
             "training_input_tokens": 0, "session_seconds_limit": 3600,
             "finalization_reserve_seconds": 900}
    (OUT / "progress.json").write_text(json.dumps(state, indent=2))
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0",
         "accelerate==1.13.0", "tree-sitter==0.25.2", "tree-sitter-language-pack",
         "opentelemetry-api==1.44.0", "opentelemetry-sdk==1.44.0",
         "opentelemetry-exporter-otlp-proto-http==1.44.0", "pydantic", "httpx", "pyyaml"],
        "setup")
    run(["git", "clone", "--branch", "research/small-model-prototype-r1",
         "https://github.com/Shlok-Bhakta/tabcomplete.git", str(ROOT)], "clone")
    run(["git", "checkout", COMMIT], "checkout")
    if sha(ROOT / "reports/research/small_model_prototype_r1/plan.json") != PLAN_SHA:
        raise ValueError("plan changed")
    fixture = list(Path("/kaggle/input").glob("**/tabcomplete-model-data-r2-inputs/causal_line_v1-r3.jsonl"))
    if len(fixture) != 1 or sha(fixture[0]) != "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead":
        raise ValueError("corrected fixture changed")
    from huggingface_hub import snapshot_download
    for alias, model_id, revision, expected_weight in MODELS:
        model = Path(snapshot_download(model_id, revision=revision, allow_patterns=[
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "model.safetensors"
        ]))
        if sha(model / "model.safetensors") != expected_weight:
            raise ValueError("downloaded weight changed")
        folder = OUT / alias
        folder.mkdir()
        run([sys.executable, str(ROOT / "scripts/evaluate_causal_line.py"), "--suite", str(fixture[0]),
             "--model", str(model), "--alias", alias, "--output", str(folder),
             "--plan-sha", PLAN_SHA], alias + "-line", env={
                 "PYTHONPATH": str(ROOT / "src"),
                 "TABCOMPLETE_CAMPAIGN_ID": "small-model-prototype-r1",
                 "TABCOMPLETE_OBSERVABILITY_ENABLED": "1",
                 "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
                 "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(folder / "telemetry.jsonl"),
             })
        summary = json.loads((folder / "summary.json").read_text())
        state["models"][alias] = {"status": "complete", "cases": summary["total"],
                                  "exact": summary["exact"], "syntax_pass": summary["syntax_pass"],
                                  "model_weight_sha256": expected_weight}
        (OUT / "progress.json").write_text(json.dumps(state, indent=2))
    state["status"] = "complete"
    state["elapsed_seconds"] = time.monotonic() - START
    (OUT / "progress.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
