"""One bounded private allocation; resume state comes from previous kernel output."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
SESSION = __SESSION_JSON__
DEADLINE = STARTED + SESSION["session_seconds"]
OUT = Path("/kaggle/working/q25_code_cpt_r2")
REPO = Path("/tmp/tabcomplete-q25-code-cpt-r2")
OUT.mkdir(parents=True, exist_ok=True)
STATUS = {"session": SESSION, "state": "setup", "stages": []}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save():
    STATUS["elapsed_seconds"] = time.monotonic() - STARTED
    temporary = OUT / "worker-status.json.tmp"
    temporary.write_text(json.dumps(STATUS, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUT / "worker-status.json")


def run(command, label, *, limit=None, env=None):
    started = time.monotonic()
    remaining = DEADLINE - started
    if remaining < 30:
        raise TimeoutError("session deadline reached")
    timeout = min(remaining - 15, limit or remaining - 30)
    with (OUT / (label + ".log")).open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, env=env, cwd=REPO if REPO.exists() else None)
    STATUS["stages"].append({"name": label, "exit_code": result.returncode,
                             "elapsed_seconds": time.monotonic() - started})
    save()
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private stage log")


def evaluate(model, alias, inputs, env):
    root = OUT / alias
    run([sys.executable, str(REPO / "scripts/evaluate_q25_code_cpt.py"), "--model", str(model),
         "--suite", str(REPO / "data/benchmarks/code_completion_v2.jsonl"),
         "--output", str(root / "causal"), "--alias", alias,
         "--plan-sha", SESSION["plan_sha256"]], alias + "-causal", limit=1200, env=env)
    (root / "line").mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(REPO / "scripts/evaluate_causal_line.py"),
         "--model", str(model), "--suite", str(inputs / "causal_line_v1-r3.jsonl"),
         "--output", str(root / "line"), "--alias", alias,
         "--plan-sha", SESSION["plan_sha256"]], alias + "-line", limit=900, env=env)


def main():
    save()
    try:
        manifests = [p for p in Path("/kaggle/input").rglob("input-manifest.json")
                     if sha(p) == SESSION["input_manifest_sha256"]]
        if len(manifests) != 1:
            raise ValueError("frozen input bundle missing or ambiguous")
        inputs = manifests[0].parent
        manifest = json.loads(manifests[0].read_text())
        for name, record in manifest["files"].items():
            if Path(name).name != name or sha(inputs / name) != record["sha256"]:
                raise ValueError("input identity differs")
        if sha(inputs / "plan.json") != SESSION["plan_sha256"]:
            raise ValueError("frozen plan differs")
        plan = json.loads((inputs / "plan.json").read_text())
        models = [p.parent for p in Path("/kaggle/input").rglob("model.safetensors")
                  if p.stat().st_size == 988097824 and sha(p) == plan["configuration"]["model"]["weight_sha256"]]
        if len(models) != 1:
            raise ValueError("approved pretrained artifact missing or ambiguous")
        model = models[0]
        for name, record in plan["existing_model_files"].items():
            if sha(model / name) != record["sha256"]:
                raise ValueError("model or tokenizer identity differs")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        run([sys.executable, "-m", "pip", "install", "--no-input", "-q",
             "transformers==5.17.0", "bitsandbytes==0.50.2", "PyYAML==6.0.2",
             "opentelemetry-api==1.44.0", "opentelemetry-sdk==1.44.0",
             "opentelemetry-exporter-otlp-proto-http==1.44.0", "pydantic>=2",
             "tree-sitter==0.25.2", "tree-sitter-language-pack==1.20.0", "httpx"], "setup", limit=600)
        run(["git", "clone", "--branch", "research/q25-code-cpt-r2",
             "https://github.com/Shlok-Bhakta/tabcomplete.git", str(REPO)], "checkout", limit=180)
        run(["git", "checkout", SESSION["commit"]], "pin", limit=60)
        env = {**os.environ, "PYTHONPATH": str(REPO / "src") + ":" + str(REPO / "scripts"),
               "TABCOMPLETE_OBSERVABILITY_ENABLED": "1", "TABCOMPLETE_OBSERVABILITY_MODE": "offline",
               "TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE": str(OUT / "observability.jsonl"),
               "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT": "0",
               "CUDA_VISIBLE_DEVICES": "0",
               "TOKENIZERS_PARALLELISM": "false"}
        import torch
        import transformers
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("intended T4 backend unavailable")
        STATUS["hardware"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "cuda_devices": torch.cuda.device_count(),
                              "training_device": torch.cuda.get_device_name(0),
                              "world_size": 1, "other_devices_unused": True}
        resume = None
        measured_evaluation_seconds = 1200
        if SESSION["resume_source"]:
            pointers = list(Path("/kaggle/input").glob("**/training/latest.json"))
            if len(pointers) != 1:
                raise ValueError("resume checkpoint pointer missing or ambiguous")
            pointer = json.loads(pointers[0].read_text())
            name = pointer.get("path") or pointer.get("checkpoint")
            if not isinstance(name, str):
                raise ValueError("resume pointer invalid")
            resume = pointers[0].parent / Path(name).name
            if not resume.exists():
                raise ValueError("resume checkpoint is unavailable")
        else:
            STATUS["state"] = "baseline_evaluation"
            save()
            evaluation_started = time.monotonic()
            evaluate(model, "untouched-q25", inputs, env)
            measured_evaluation_seconds = time.monotonic() - evaluation_started
        finalization_reserve = max(
            plan["configuration"]["budget"]["finalization_reserve_seconds"],
            int(1.5 * measured_evaluation_seconds + 300),
        )
        STATUS["measured_evaluation_seconds"] = measured_evaluation_seconds
        STATUS["finalization_reserve_seconds"] = finalization_reserve
        STATUS["state"] = "training"
        save()
        command = [sys.executable, "-m", "tinycomplete.code_cpt.q25", "--model", str(model),
                   "--train", str(inputs / "train_blocks.npy"),
                   "--development", str(inputs / "development_blocks.npy"),
                   "--plan", str(inputs / "plan.json"), "--output", str(OUT / "training"),
                   "--session-seconds", str(int(DEADLINE - time.monotonic())),
                   "--reserve-seconds", str(finalization_reserve),
                   "--external-campaign-tokens", str(SESSION.get("external_campaign_tokens", 0)),
                   "--execute"]
        if resume:
            command.extend(["--resume", str(resume)])
        run(command, "training", env=env)
        result = json.loads((OUT / "training/run_result.json").read_text())
        STATUS["training"] = result
        export = OUT / "training/inference-f16"
        if export.exists() and DEADLINE - time.monotonic() > measured_evaluation_seconds * 1.5 + 60:
            STATUS["state"] = "candidate_evaluation"
            save()
            evaluate(export, "cpt-q25", inputs, env)
            STATUS["candidate_evaluation_status"] = "complete"
        else:
            STATUS["candidate_evaluation_status"] = "deferred_with_checkpoint_preserved"
        STATUS["state"] = "complete" if result["status"] == "complete" else "partial_checkpoint_preserved"
        save()
        return 0
    except Exception as exc:
        STATUS.update(state="failed", error_class=type(exc).__name__)
        save()
        print(json.dumps({"state": "failed", "error_class": type(exc).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
