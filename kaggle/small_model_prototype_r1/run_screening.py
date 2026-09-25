"""Bounded unquantized screening of the two new approved foundations."""

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
SESSION_SECONDS = 7200
RESERVE_SECONDS = 900
DEADLINE = START + SESSION_SECONDS - RESERVE_SECONDS
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/small_model_prototype_r1_screening")
COMMIT = "__CHECKOUT_COMMIT__"
PLAN_SHA = "__PLAN_SHA__"
ALLOWED = [
    ("q3-base", "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", 596049920),
    ("lfm350-base", "LiquidAI/LFM2.5-350M-Base", "9960764e30892e01f29a6dc23df2533fcd8bd5ae", 354483968),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(value):
    (OUT / "progress.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(args, label):
    remaining = DEADLINE - time.monotonic()
    if remaining < 120:
        raise TimeoutError("finalization reserve reached")
    result = subprocess.run(args, cwd=ROOT if ROOT.exists() else None, capture_output=True, text=True, timeout=remaining)
    (OUT / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private job log")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "setup", "started_monotonic": START, "session_seconds_limit": SESSION_SECONDS,
             "finalization_reserve_seconds": RESERVE_SECONDS, "training_input_tokens": 0,
             "models": {}, "plan_sha256": PLAN_SHA}
    save(state)
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0", "accelerate==1.13.0",
         "opentelemetry-api==1.44.0", "opentelemetry-sdk==1.44.0",
         "opentelemetry-exporter-otlp-proto-http==1.44.0", "tree-sitter==0.25.2",
         "tree-sitter-language-pack", "pydantic", "httpx", "pyyaml"], "setup")
    run(["git", "clone", "--branch", "research/small-model-prototype-r1",
         "https://github.com/Shlok-Bhakta/tabcomplete.git", str(ROOT)], "clone")
    run(["git", "checkout", COMMIT], "checkout")
    if sha(ROOT / "reports/research/small_model_prototype_r1/plan.json") != PLAN_SHA:
        raise ValueError("frozen plan hash mismatch")
    suite = ROOT / "data/benchmarks/code_completion_v2.jsonl"
    if sha(suite) != "ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4":
        raise ValueError("causal fixture changed")
    lines = list(Path("/kaggle/input").glob("**/tabcomplete-model-data-r2-inputs/causal_line_v1-r3.jsonl"))
    if len(lines) != 1 or sha(lines[0]) != "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead":
        raise ValueError("corrected line fixture missing or changed")
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from tinycomplete.eval.code_benchmark import load_suite
    from tinycomplete.eval.code_generation import (TransformersGenerationProvider, build_causal_prompt,
                                                   build_prediction_run_metadata, generate_predictions)
    from tinycomplete.observability.bootstrap import current_runtime
    state.update(status="screening", torch=torch.__version__, transformers=transformers.__version__,
                 gpu=torch.cuda.get_device_name(0))
    save(state)
    cases = load_suite(suite)

    class BoundedProvider(TransformersGenerationProvider):
        def generate_detailed(self, prompt, max_new_tokens):
            if time.monotonic() >= DEADLINE:
                raise TimeoutError("finalization reserve reached")
            return super().generate_detailed(prompt, max_new_tokens)

    for alias, model_id, revision, expected_parameters in ALLOWED:
        record = {"status": "starting", "source": model_id, "revision": revision}
        state["models"][alias] = record
        save(state)
        try:
            model_path = Path(snapshot_download(model_id, revision=revision, allow_patterns=[
                "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "model.safetensors"
            ]))
            record["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                               for p in model_path.iterdir() if p.is_file()}
            record["status"] = "loading"
            save(state)
            provider = BoundedProvider(str(model_path), device="cuda:0")
            actual = sum(p.numel() for p in provider.model.parameters())
            if actual != expected_parameters:
                raise ValueError("loaded parameter count differs from frozen Hub total")
            smoke = provider.generate_detailed("# synthetic Python\ndef add(a, b):\n    return", 16)
            record.update(model_class=type(provider.model).__name__, total_parameters=actual,
                          tokenizer_class=type(provider.tokenizer).__name__,
                          smoke={"text": smoke.text, "finish_reason": smoke.finish_reason,
                                 "tokens": smoke.tokens},
                          peak_gpu_bytes=torch.cuda.max_memory_allocated())
            if not smoke.text or not all(torch.isfinite(p).all().item() for p in list(provider.model.parameters())[:1]):
                raise ValueError("smoke produced empty or nonfinite output")
            record["status"] = "causal"
            save(state)
            metadata = build_prediction_run_metadata(suite_path=suite, case_count=len(cases),
                provider="transformers", model_source=alias, model_revision=sha(model_path / "model.safetensors"),
                max_new_tokens=96, workers=1)
            metadata.update(plan_sha256=PLAN_SHA, tokenizer_sha256=sha(model_path / "tokenizer.json"),
                            precision="fp16", protocol_version="strict-causal-v2")
            predictions = generate_predictions(cases, provider, OUT / (alias + "-causal.jsonl"),
                run_metadata=metadata, max_new_tokens=96, workers=1)
            record["causal_cases"] = len(predictions)
            record["causal_lengths"] = [{"case_id": case.id,
                "input_tokens": len(provider.tokenizer.encode(build_causal_prompt(case))),
                "output_tokens": pred.generated_tokens, "output_bytes": len(pred.completion.encode())}
                for case, pred in zip(cases, predictions, strict=True)]
            del provider
            gc.collect()
            torch.cuda.empty_cache()
            record["status"] = "line"
            save(state)
            (OUT / (alias + "-line")).mkdir(exist_ok=True)
            run([sys.executable, str(ROOT / "scripts/evaluate_causal_line.py"), "--suite", str(lines[0]),
                 "--model", str(model_path), "--alias", alias, "--output", str(OUT / (alias + "-line")),
                 "--plan-sha", PLAN_SHA],
                 alias + "-line")
            record["status"] = "complete"
            record["elapsed_seconds"] = time.monotonic() - START
            save(state)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            record["status"] = "blocked"
            record["error_type"] = type(exc).__name__
            save(state)
            gc.collect()
            torch.cuda.empty_cache()
    state["status"] = "complete" if all(v["status"] == "complete" for v in state["models"].values()) else "partial"
    state["elapsed_seconds"] = time.monotonic() - START
    save(state)
    current_runtime().shutdown()


if __name__ == "__main__":
    main()
