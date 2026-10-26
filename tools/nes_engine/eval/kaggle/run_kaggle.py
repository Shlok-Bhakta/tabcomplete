"""Kaggle GPU runner for next-edit engine evaluations.

Builds the engine with CUDA for the attached GPU, downloads the pinned Sweep
model, and runs every configuration in variants.json over the attached case
sets. Results land in /kaggle/working/results.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
# Build tree and model stay out of the output directory.
SCRATCH = Path("/tmp/nes")
DATA = Path("/kaggle/input")
MODEL_URL = ("https://huggingface.co/sweepai/sweep-next-edit-0.5B/resolve/"
             "8175624004cec8bf0b550fba35d6ad92126e48ac/sweep-next-edit-0.5b.q8_0.gguf")


def sh(cmd, env=None, **kw):
    print("+", cmd, flush=True)
    merged = {**os.environ, **(env or {})}
    subprocess.run(cmd, shell=True, check=True, env=merged, **kw)


def find(name):
    hits = list(DATA.rglob(name))
    if not hits:
        raise SystemExit(f"missing input {name}")
    return hits[0]


def main():
    started = time.time()
    log = {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    src = SCRATCH / "nes_engine"
    if src.exists():
        shutil.rmtree(src)
    # Kaggle unpacks archives in datasets; accept either form.
    unpacked = [p.parent for p in DATA.rglob("Cargo.toml") if p.parent.name == "nes_engine"]
    if unpacked:
        shutil.copytree(unpacked[0], src)
    else:
        with tarfile.open(find("nes_engine.tar.gz")) as t:
            t.extractall(SCRATCH)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh("command -v cargo || (curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal)")
    sh("(command -v clang && ls /usr/lib/llvm-*/lib/libclang*.so*) >/dev/null 2>&1 || "
       "(apt-get update -qq && apt-get install -y -qq clang libclang-dev >/dev/null)")
    cargo = os.path.expanduser("~/.cargo/bin/cargo")
    llvm = sorted(Path("/usr/lib").glob("llvm-*/lib"))
    env = {
        "CMAKE_CUDA_ARCHITECTURES": "75",
        "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 4),
        "LIBCLANG_PATH": str(llvm[-1]) if llvm else "",
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs:" + os.environ.get("LIBRARY_PATH", ""),
        "PATH": "/usr/local/cuda/bin:" + os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"],
    }
    t0 = time.time()
    out = WORK / "results"
    out.mkdir(exist_ok=True)
    sh("cmake --version | head -1; nvcc --version | tail -1; gcc --version | head -1", env=env)
    build = subprocess.run(f"cd {src} && {cargo} build --release --features cuda -vv",
                           shell=True, env={**os.environ, **env}, capture_output=True, text=True)
    (out / "build.log").write_text(build.stdout[-200000:] + "\n--- stderr ---\n" + build.stderr[-400000:])
    log["build_s"] = round(time.time() - t0)
    if build.returncode != 0:
        tail = build.stderr[-6000:]
        print(tail, flush=True)
        (out / "timing.json").write_text(json.dumps(log))
        raise SystemExit("engine build failed")
    model = SCRATCH / "model.gguf"
    if not model.exists():
        urllib.request.urlretrieve(MODEL_URL, model)
    engine = src / "target/release/tabcomplete-nes"
    variants = json.loads(find("variants.json").read_text())
    runner = src / "eval/run_eval.py"
    for v in variants:
        cases = find(v["cases"])
        name = v["name"]
        t0 = time.time()
        cmd = [sys.executable, "-I", str(runner), "--engine", str(engine), "--model", str(model),
               "--cases", str(cases), "--name", name, "--limit", str(v.get("limit", 0)),
               "--out", str(out / f"{name}.rows.jsonl")]
        if v.get("no_jump", True):
            cmd.append("--no-jump")
        cmd += ["--", "--gpu-layers", "99", *v["flags"]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
        if res.returncode != 0 or not line:
            line = json.dumps({"name": name, "error": res.stderr[-1500:]})
        print(line, flush=True)
        with open(out / "summary.jsonl", "a") as f:
            f.write(line + "\n")
        log[name] = round(time.time() - t0)
    log["total_s"] = round(time.time() - started)
    (out / "timing.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log))


if __name__ == "__main__":
    main()
