"""One sealed evaluation of the development-selected adapted checkpoint."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
DEADLINE = START + 7200 - 900
ROOT = Path("/kaggle/working/tabcomplete")
OUT = Path("/kaggle/working/small_model_prototype_r1_heldout")
COMMIT = "__CHECKOUT_COMMIT__"
PLAN_SHA = "__PLAN_SHA__"
SELECTION_SHA = "__SELECTION_SHA__"
ALIAS = "__SELECTED_ALIAS__"
WEIGHT_SHA = "__ADAPTED_WEIGHT_SHA__"
DATA_SHA = "0b66a598e08b28b8dcfa4a0517ca695abc1b55527f7bb7bc718b213cf769a9d5"
DEV_SHA = "3e0c6046dd9726b174c263d823b726f8dee778281845dd09bfe9dd7062dcd465"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args, label, *, env=None):
    remaining = DEADLINE - time.monotonic()
    if remaining < 900:
        raise TimeoutError("held-out session finalization reserve reached")
    result = subprocess.run(
        args, cwd=ROOT if ROOT.exists() else None,
        env={**os.environ, **(env or {})}, capture_output=True, text=True,
        timeout=remaining,
    )
    (OUT / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private log")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "setup", "plan_sha256": PLAN_SHA, "selection_sha256": SELECTION_SHA,
             "selected_alias": ALIAS, "adapted_weight_sha256": WEIGHT_SHA,
             "sealed_split_sha256": DATA_SHA, "gpu_session_seconds_limit": 7200,
             "finalization_reserve_seconds": 900}
    marker = OUT / "result.json"
    marker.write_text(json.dumps(state, indent=2) + "\n")
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.0",
         "accelerate==1.13.0", "pydantic", "httpx", "pyyaml"], "setup")
    run(["git", "clone", "--branch", "research/small-model-prototype-r1",
         "https://github.com/Shlok-Bhakta/tabcomplete.git", str(ROOT)], "clone")
    run(["git", "checkout", COMMIT], "checkout")
    if sha(ROOT / "reports/research/small_model_prototype_r1/plan.json") != PLAN_SHA:
        raise ValueError("plan identity changed")
    decision = ROOT / "reports/research/small_model_prototype_r1/deployment/selection.json"
    if sha(decision) != SELECTION_SHA:
        raise ValueError("deployment decision changed")
    pattern = "**/small_model_prototype_r1_adaptation/progress.json"
    markers = list(Path("/kaggle/input").glob(pattern))
    if len(markers) != 1:
        raise RuntimeError("expected exactly one adapted source output")
    progress = json.loads(markers[0].read_text())
    if progress["status"] != "complete" or progress["plan_sha256"] != PLAN_SHA:
        raise ValueError("adaptation source was not complete under this plan")
    if progress["models"][ALIAS]["main"]["training"]["inference_weight_sha256"] != WEIGHT_SHA:
        raise ValueError("selected adapted weight identity mismatch")
    model = markers[0].parent / ALIAS / "main/inference"
    if sha(model / "model.safetensors") != WEIGHT_SHA:
        raise ValueError("attached checkpoint hash mismatch")
    data = OUT / "data"
    run([sys.executable, str(ROOT / "scripts/build_small_edit_data.py"),
         "--output", str(data)], "data")
    if sha(data / "heldout.jsonl") != DATA_SHA or sha(data / "development.jsonl") != DEV_SHA:
        raise ValueError("held-out source changed")
    result_path = OUT / "evaluation"
    run([sys.executable, str(ROOT / "scripts/train_small_next_edit.py"),
         "--model", str(model), "--expected-weight-sha256", WEIGHT_SHA,
         "--data", str(data / "heldout.jsonl"), "--data-sha256", DATA_SHA,
         "--development", str(data / "development.jsonl"), "--development-sha256", DEV_SHA,
         "--output", str(result_path), "--phase", "evaluate",
         "--deadline-monotonic", str(DEADLINE)], "sealed-evaluation",
        env={"PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")})
    evaluation = json.loads((result_path / "evaluate-result.json").read_text())
    state.update(status="complete", elapsed_seconds=time.monotonic() - START,
                 evaluation_summary=evaluation["evaluation"]["summary"])
    marker.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
