"""One-allocation matched supervised pilot on preregistered finalists."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
SESSION_SECONDS = 14_400
FINALIZATION_RESERVE = 1_800
DEADLINE = START + SESSION_SECONDS - FINALIZATION_RESERVE
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/small_model_prototype_r1_adaptation")
COMMIT = "__CHECKOUT_COMMIT__"
PLAN_SHA = "__PLAN_SHA__"
SELECTION_SHA = "__SELECTION_SHA__"
FINALISTS = __FINALISTS__
MODEL_SOURCES = {
    "q25-coder": ("Qwen/Qwen2.5-Coder-0.5B", "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301",
                  "d58d103feb3d4c4bbfb86cfbe5620053a78c437a5469cefa44d8bf7dccb4e363"),
    "q3-base": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
    "lfm350-base": ("LiquidAI/LFM2.5-350M-Base", "9960764e30892e01f29a6dc23df2533fcd8bd5ae",
                    "af70818c41a5cdb3f9587f91de12ff5f7847b8b0a2ba734534205ccea1d98aba"),
}
DATA_HASHES = {
    "train": "b7be3cdb79379ed98618da9ee959bce5e049d4974b90db20377d1de92d554402",
    "development": "3e0c6046dd9726b174c263d823b726f8dee778281845dd09bfe9dd7062dcd465",
    "heldout": "0b66a598e08b28b8dcfa4a0517ca695abc1b55527f7bb7bc718b213cf769a9d5",
}
MAX_NEW_BYTES = 12 * 1024**3
MAX_TRAIN_INPUT_TOKENS = 32_000_000


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(value):
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(OUT / "hf-cache")
    (OUT / "progress.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def bytes_under(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(args, label, *, environment=None):
    if bytes_under(OUT) > MAX_NEW_BYTES - 3 * 1024**3:
        raise RuntimeError("artifact budget has insufficient room for a checkpoint")
    remaining = DEADLINE - time.monotonic()
    if remaining < 900:
        raise TimeoutError("finalization reserve reached")
    started = time.monotonic()
    result = subprocess.run(args, cwd=ROOT if ROOT.exists() else None,
                            env={**os.environ, **(environment or {})},
                            capture_output=True, text=True, timeout=remaining)
    (OUT / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private job log")
    return time.monotonic() - started


def training_call(alias, model_path, phase, lr, *, max_examples=0, destination=None):
    result_path = destination or (OUT / alias / (phase + "-" + str(lr)))
    result_path.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(ROOT / "scripts/train_small_next_edit.py"),
            "--model", str(model_path), "--expected-weight-sha256", MODEL_SOURCES[alias][2],
            "--data", str(OUT / "data/train.jsonl"), "--data-sha256", DATA_HASHES["train"],
            "--development", str(OUT / "data/development.jsonl"),
            "--development-sha256", DATA_HASHES["development"],
            "--output", str(result_path), "--phase", phase,
            "--learning-rate", str(lr), "--deadline-monotonic", str(DEADLINE)]
    if max_examples:
        args.extend(["--max-examples", str(max_examples)])
    elapsed = run(args, alias + "-" + phase + "-" + str(lr))
    return json.loads((result_path / (phase + "-result.json")).read_text()), elapsed


def score(result):
    summary = result["evaluation"]["summary"]
    return (summary["edit_required_exact_after_state"],
            summary["by_action"].get("no_edit", {}).get("exact_after_state", 0),
            summary["valid"], summary["terminated"])


def main():
    if FINALISTS[0] != "q25-coder" or len(FINALISTS) > 2 or len(set(FINALISTS)) != len(FINALISTS):
        raise ValueError("invalid finalist selection")
    if any(alias not in MODEL_SOURCES for alias in FINALISTS):
        raise ValueError("non-allowlisted finalist")
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "setup", "finalists": FINALISTS, "plan_sha256": PLAN_SHA,
             "selection_sha256": SELECTION_SHA, "session_seconds_limit": SESSION_SECONDS,
             "finalization_reserve_seconds": FINALIZATION_RESERVE,
             "reserved_training_input_tokens": 0, "models": {}}
    save(state)
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0",
         "accelerate==1.13.0", "bitsandbytes==0.50.2", "pydantic", "httpx", "pyyaml",
         "opentelemetry-api==1.44.0", "opentelemetry-sdk==1.44.0",
         "opentelemetry-exporter-otlp-proto-http==1.44.0"], "setup")
    run(["git", "clone", "--branch", "research/small-model-prototype-r1",
         "https://github.com/Shlok-Bhakta/tabcomplete.git", str(ROOT)], "clone")
    run(["git", "checkout", COMMIT], "checkout")
    if sha(ROOT / "reports/research/small_model_prototype_r1/plan.json") != PLAN_SHA:
        raise ValueError("plan changed")
    if sha(ROOT / "reports/research/small_model_prototype_r1/selection.json") != SELECTION_SHA:
        raise ValueError("finalist selection changed")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from train_small_next_edit import encode_rows, read_rows

    data_dir = OUT / "data"
    run([sys.executable, str(ROOT / "scripts/build_small_edit_data.py"),
         "--output", str(data_dir)], "data-verify")
    for split, expected in DATA_HASHES.items():
        if sha(data_dir / (split + ".jsonl")) != expected:
            raise ValueError("synthetic adaptation data changed")
    state["status"] = "pilots"
    save(state)
    for alias in FINALISTS:
        source, revision, expected_weight = MODEL_SOURCES[alias]
        record = state["models"].setdefault(alias, {"source": source, "revision": revision})
        model_path = Path(snapshot_download(source, revision=revision, allow_patterns=[
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "model.safetensors"
        ]))
        if sha(model_path / "model.safetensors") != expected_weight:
            raise ValueError("downloaded checkpoint changed")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        train_rows = read_rows(data_dir / "train.jsonl")
        inventory, _ = encode_rows(tokenizer, train_rows)
        per_pass = sum(row["input_tokens"] for row in inventory)
        fixture_rows = []
        from build_small_edit_data import ACTIONS, LANGUAGES, example
        for language in LANGUAGES:
            for action in ACTIONS:
                for variant in range(4):
                    fixture_rows.append(example(language, action, 99, variant))
        fixture_inventory, _ = encode_rows(tokenizer, fixture_rows)
        fixture_tokens = sum(row["input_tokens"] for row in fixture_inventory) * 2
        probe_tokens = sum(row["input_tokens"] for row in inventory[:1024])
        planned = fixture_tokens + probe_tokens * 2 + per_pass
        if state["reserved_training_input_tokens"] + planned > MAX_TRAIN_INPUT_TOKENS:
            raise RuntimeError("campaign-wide token budget exceeded")
        state["reserved_training_input_tokens"] += planned
        record["reserved_tokens"] = planned
        record["token_inventory"] = {"main_pass": per_pass, "fixture": fixture_tokens,
                                     "each_lr_probe": probe_tokens}
        save(state)
        fixture, elapsed = training_call(alias, model_path, "fixture", 3e-4)
        record["fixture"] = {"seconds": elapsed, "summary": fixture["evaluation"]["summary"]}
        save(state)
        fixture_actions = fixture["evaluation"]["summary"]["by_action"]
        if any(fixture_actions[action]["valid"] == 0 or fixture_actions[action]["terminated"] == 0
               for action in ACTIONS):
            record["status"] = "fixture_failed"
            save(state)
            continue
        probes = {}
        for lr in (1e-5, 3e-5):
            result, elapsed = training_call(alias, model_path, "probe", lr, max_examples=1024)
            probes[str(lr)] = {"seconds": elapsed, "score": score(result),
                               "summary": result["evaluation"]["summary"]}
            record["probes"] = probes
            save(state)
        choice = max((1e-5, 3e-5), key=lambda lr: probes[str(lr)]["score"])
        record["chosen_learning_rate"] = choice
        record["selection_before_main"] = True
        save(state)
        main_result, elapsed = training_call(alias, model_path, "main", choice,
                                              destination=OUT / alias / "main")
        record["main"] = {"seconds": elapsed, "training": main_result["training"],
                          "token_inventory": main_result["token_inventory"],
                          "development": main_result["evaluation"]["summary"]}
        save(state)
        verified, elapsed = training_call(alias, model_path, "verify", choice, max_examples=16,
                                           destination=OUT / alias / "main")
        record["reload"] = {"seconds": elapsed, **verified["reload"]}
        record["status"] = "complete"
        save(state)
        if bytes_under(OUT) > MAX_NEW_BYTES:
            raise RuntimeError("new artifact budget exceeded")
        del tokenizer
    state["status"] = "complete" if all(state["models"].get(alias, {}).get("status") == "complete"
                                         for alias in FINALISTS) else "partial"
    state["elapsed_seconds"] = time.monotonic() - START
    save(state)


if __name__ == "__main__":
    main()
