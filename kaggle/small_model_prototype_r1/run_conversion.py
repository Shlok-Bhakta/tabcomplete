"""CPU-only private Q4 conversion of the two new allowlisted foundations."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
DEADLINE = START + 7200 - 900
OUT = Path("/kaggle/working/small_model_prototype_r1_conversion")
RUNTIME = Path("/kaggle/working/llama.cpp")
RUNTIME_SHA = "f072b103714dfa1eee531f80b24512faf38e3dd2"
MODELS = [
    ("q3-base", "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
     "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
     "apache-2.0", 596049920),
    ("lfm350-base", "LiquidAI/LFM2.5-350M-Base", "9960764e30892e01f29a6dc23df2533fcd8bd5ae",
     "af70818c41a5cdb3f9587f91de12ff5f7847b8b0a2ba734534205ccea1d98aba",
     "lfm1.0", 354483968),
]
MAX_BYTES = 12 * 1024**3


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run(args, label, cwd=None):
    remaining = DEADLINE - time.monotonic()
    if remaining < 120:
        raise TimeoutError("finalization reserve reached")
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=remaining)
    (OUT / (label + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect private log")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(OUT / "hf-cache")
    manifest = {"status": "setup", "runtime_revision": RUNTIME_SHA,
                "artifact_budget_bytes": MAX_BYTES, "session_seconds_limit": 7200,
                "finalization_reserve_seconds": 900, "gpu_enabled": False, "models": {}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, "-m", "pip", "install", "-q", "numpy~=2.2.6",
         "sentencepiece>=0.1.98,<0.3.0", "transformers==4.57.6", "safetensors",
         "protobuf>=4.21,<5", "huggingface_hub"], "setup")
    run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git",
         str(RUNTIME)], "clone")
    run(["git", "checkout", RUNTIME_SHA], "runtime-checkout", cwd=RUNTIME)
    run(["cmake", "-S", str(RUNTIME), "-B", str(RUNTIME / "build"),
         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF",
         "-DLLAMA_BUILD_TESTS=OFF"], "configure")
    run(["cmake", "--build", str(RUNTIME / "build"), "--target", "llama-quantize",
         "-j", "4"], "build-quantizer")
    from huggingface_hub import snapshot_download
    for alias, repo, revision, expected_weight, license_id, total_parameters in MODELS:
        if size(OUT) + 3 * 1024**3 > MAX_BYTES:
            raise RuntimeError("insufficient model conversion budget")
        model = Path(snapshot_download(repo, revision=revision, allow_patterns=[
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "model.safetensors"
        ]))
        if sha(model / "model.safetensors") != expected_weight:
            raise ValueError("source weight identity mismatch")
        f16 = OUT / (alias + "-f16.gguf")
        q4 = OUT / (alias + "-Q4_K_M.gguf")
        run([sys.executable, str(RUNTIME / "convert_hf_to_gguf.py"), str(model),
             "--outtype", "f16", "--outfile", str(f16)], alias + "-convert")
        if size(OUT) + 1 * 1024**3 > MAX_BYTES:
            raise RuntimeError("insufficient quantization budget")
        run([str(RUNTIME / "build/bin/llama-quantize"), str(f16), str(q4), "Q4_K_M"],
            alias + "-quantize")
        manifest["models"][alias] = {"source": repo, "revision": revision,
            "source_weight_sha256": expected_weight, "total_parameters": total_parameters,
            "license": license_id, "q4_sha256": sha(q4), "q4_bytes": q4.stat().st_size,
            "converter_sha256": sha(RUNTIME / "convert_hf_to_gguf.py"),
            "quantizer_sha256": sha(RUNTIME / "build/bin/llama-quantize")}
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        f16.unlink()
        cache_root = model.parents[1]
        if not cache_root.is_relative_to(OUT / "hf-cache"):
            raise RuntimeError("refusing to remove a preexisting cache")
        shutil.rmtree(cache_root)
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.monotonic() - START
    manifest["final_artifact_bytes"] = size(OUT)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
