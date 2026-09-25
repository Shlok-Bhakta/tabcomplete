"""CPU-only Q4 export of the one development-selected adapted checkpoint."""

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
RUNTIME = Path("/kaggle/working/llama.cpp")
OUT = Path("/kaggle/working/small_model_prototype_r1_selected_conversion")
COMMIT = "__CHECKOUT_COMMIT__"
PLAN_SHA = "__PLAN_SHA__"
SELECTION_SHA = "__SELECTION_SHA__"
ALIAS = "__SELECTED_ALIAS__"
WEIGHT_SHA = "__ADAPTED_WEIGHT_SHA__"
RUNTIME_SHA = "f072b103714dfa1eee531f80b24512faf38e3dd2"
MAX_NEW_BYTES = 12 * 1024**3


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def used_bytes():
    return sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())


def run(args, label, *, cwd=None):
    remaining = DEADLINE - time.monotonic()
    if remaining < 900:
        raise TimeoutError("conversion session finalization reserve reached")
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=remaining)
    (OUT / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private log")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(OUT / "hf-cache")
    state = {"status": "setup", "plan_sha256": PLAN_SHA, "selection_sha256": SELECTION_SHA,
             "selected_alias": ALIAS, "adapted_weight_sha256": WEIGHT_SHA,
             "runtime_revision": RUNTIME_SHA, "precision": "Q4_K_M",
             "artifact_budget_bytes": MAX_NEW_BYTES, "gpu_enabled": False}
    marker = OUT / "manifest.json"
    marker.write_text(json.dumps(state, indent=2) + "\n")
    run([sys.executable, "-m", "pip", "install", "-q", "numpy~=2.2.6",
         "sentencepiece>=0.1.98,<0.3.0", "transformers==4.57.6",
         "safetensors", "protobuf>=4.21,<5", "huggingface_hub"], "setup")
    run(["git", "clone", "--branch", "research/small-model-prototype-r1",
         "https://github.com/Shlok-Bhakta/tabcomplete.git", str(ROOT)], "repo-clone")
    run(["git", "checkout", COMMIT], "repo-checkout", cwd=ROOT)
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
    run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git",
         str(RUNTIME)], "runtime-clone")
    run(["git", "checkout", RUNTIME_SHA], "runtime-checkout", cwd=RUNTIME)
    run(["cmake", "-S", str(RUNTIME), "-B", str(RUNTIME / "build"),
         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF",
         "-DLLAMA_BUILD_TESTS=OFF"], "configure")
    run(["cmake", "--build", str(RUNTIME / "build"), "--target", "llama-quantize",
         "-j", "4"], "build")
    if used_bytes() + 3 * 1024**3 > MAX_NEW_BYTES:
        raise RuntimeError("insufficient conversion budget")
    f16 = OUT / "selected-f16.gguf"
    q4 = OUT / (ALIAS + "-adapted-Q4_K_M.gguf")
    run([sys.executable, str(RUNTIME / "convert_hf_to_gguf.py"), str(model),
         "--outtype", "f16", "--outfile", str(f16)], "convert")
    if used_bytes() + 1 * 1024**3 > MAX_NEW_BYTES:
        raise RuntimeError("insufficient quantization budget")
    run([str(RUNTIME / "build/bin/llama-quantize"), str(f16), str(q4), "Q4_K_M"],
        "quantize")
    state.update(status="complete", q4_sha256=sha(q4), q4_bytes=q4.stat().st_size,
                 converter_sha256=sha(RUNTIME / "convert_hf_to_gguf.py"),
                 quantizer_sha256=sha(RUNTIME / "build/bin/llama-quantize"),
                 elapsed_seconds=time.monotonic() - START)
    f16.unlink()
    state["new_artifact_bytes"] = used_bytes()
    marker.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
