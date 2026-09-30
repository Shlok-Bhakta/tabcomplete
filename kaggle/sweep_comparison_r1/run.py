"""Bounded, private Kaggle launcher for the authorized Sweep GGUF comparison.

The input spec and runner are prepared on CPU before allocating a GPU. No
training or additional model families are used. Setup, compilation and failures
consume the same session deadline as inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

START = float(os.environ.get("TABCOMPLETE_SWEEP_SESSION_START", time.monotonic()))
SESSION_SECONDS = 7200
RESERVE_SECONDS = 1200
DEADLINE = START + SESSION_SECONDS - RESERVE_SECONDS
RUNTIME_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"
MODEL_REVISION = "409016591c6c1a94f545f22328a85a3516118f34"
MODEL_SHA256 = "1321ea5e5d7529e60f9770c6a0b3a965f89542d16cf4ae51bab267f6a88150da"
OUT = Path("/kaggle/working/sweep_comparison_r1")
REPO = Path("/kaggle/temp/tabcomplete-sweep")
RUNTIME = Path("/kaggle/temp/llama-sweep")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def remaining_seconds(now: float | None = None) -> float:
    remaining = DEADLINE - (time.monotonic() if now is None else now)
    if remaining <= 0:
        raise TimeoutError("Sweep session finalization reserve reached")
    return remaining


def run(argv: list[str], label: str, cwd: Path | None = None) -> None:
    # Never include raw process exception messages or credentials in status.
    with (OUT / (label + ".log")).open("w") as handle:
        timeout = remaining_seconds()
        process = subprocess.Popen(
            argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Include owned compiler/server children when enforcing the budget.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise TimeoutError("Sweep stage exceeded session deadline") from None
    if process.returncode:
        raise RuntimeError(label + " failed; inspect private stage log")


def validate_spec(spec: dict) -> None:
    if (
        spec.get("schema") != "sweep-comparison-kaggle-input-v1"
        or spec.get("model_revision") != MODEL_REVISION
        or spec.get("model_sha256") != MODEL_SHA256
        or spec.get("runtime_revision") != RUNTIME_REVISION
        or spec.get("session_seconds") != SESSION_SECONDS
        or spec.get("reserve_seconds") != RESERVE_SECONDS
        or spec.get("training_enabled") is not False
        or not isinstance(spec.get("files"), dict)
        or not isinstance(spec.get("runner_arguments"), list)
        or not spec["runner_arguments"]
        or any(not isinstance(item, str) or len(item) > 4096
               for item in spec["runner_arguments"])
        or "--mode" in spec["runner_arguments"]
        or spec.get("runner_modes") != ["download", "quantize", "quality", "next-edit"]
        or not isinstance(spec.get("runner_sha256"), str)
        or len(spec["runner_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in spec["runner_sha256"])
        or not isinstance(spec.get("commit"), str)
        or len(spec["commit"]) != 40
        or any(c not in "0123456789abcdef" for c in spec["commit"])
    ):
        raise ValueError("Sweep worker spec identity mismatch")
    for name, sha in spec["files"].items():
        if (
            not isinstance(name, str) or Path(name).name != name
            or not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("Sweep input file identity is invalid")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = {
        "schema": "sweep-comparison-worker-status-v1", "state": "preflight",
        "session_seconds": SESSION_SECONDS, "reserve_seconds": RESERVE_SECONDS,
        "training_input_tokens": 0,
    }
    stage = "input_validation"
    try:
        specs = list(Path("/kaggle/input").glob("**/sweep-worker-spec.json"))
        if len(specs) != 1:
            raise ValueError("exactly one Sweep input spec is required")
        input_dir = specs[0].parent
        spec = json.loads(specs[0].read_text())
        validate_spec(spec)
        for name, sha in spec["files"].items():
            if digest(input_dir / name) != sha:
                raise ValueError("Sweep input artifact hash mismatch")
        status["input_spec_sha256"] = digest(specs[0])
        stage = "python_setup"
        if os.environ.get("TABCOMPLETE_SWEEP_PY311") != "1":
            run([sys.executable, "-m", "pip", "install", "uv==0.12.3"], "install-uv")
            run([sys.executable, "-m", "uv", "venv", "--python", "3.11",
                 "/kaggle/temp/sweep-python311"], "python311")
            env = os.environ.copy()
            env.update(TABCOMPLETE_SWEEP_PY311="1", TABCOMPLETE_SWEEP_SESSION_START=str(START))
            os.execve("/kaggle/temp/sweep-python311/bin/python", [
                "/kaggle/temp/sweep-python311/bin/python", str(Path(__file__).resolve())
            ], env)
        if sys.version_info[:2] != (3, 11):
            raise ValueError("Sweep worker requires Python 3.11")
        run([sys.executable, "-m", "ensurepip"], "ensurepip")
        run([sys.executable, "-m", "pip", "install", "httpx==0.28.1", "pydantic==2.12.5",
             "PyYAML==6.0.2", "tree-sitter==0.25.2", "tree-sitter-language-pack==1.20.0",
             "opentelemetry-api==1.44.0", "opentelemetry-sdk==1.44.0",
             "opentelemetry-exporter-otlp-proto-http==1.44.0"], "dependencies")
        stage = "checkout"
        run(["git", "clone", "--filter=blob:none", "https://github.com/Shlok-Bhakta/tabcomplete.git",
             str(REPO)], "repo-clone")
        run(["git", "checkout", spec["commit"]], "repo-checkout", REPO)
        if digest(REPO / "scripts/run_sweep_comparison.py") != spec["runner_sha256"]:
            raise ValueError("Sweep runner revision mismatch")
        stage = "runtime_build"
        run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git",
             str(RUNTIME)], "runtime-clone")
        run(["git", "checkout", RUNTIME_REVISION], "runtime-checkout", RUNTIME)
        run(["cmake", "-S", str(RUNTIME), "-B", str(RUNTIME / "build"),
             "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF",
             "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DCMAKE_CUDA_ARCHITECTURES=75"],
            "runtime-configure")
        run(["cmake", "--build", str(RUNTIME / "build"), "--target", "llama-server",
             "llama-quantize", "-j", "2"], "runtime-build")
        stage = "comparison"
        arguments = [part.replace("{input}", str(input_dir)).replace("{output}", str(OUT))
                     .replace("{runtime}", str(RUNTIME))
                     .replace("{scratch}", "/kaggle/temp/sweep-artifacts")
                     for part in spec["runner_arguments"]]
        os.environ.update(PYTHONPATH=str(REPO / "src"), TABCOMPLETE_OBSERVABILITY_MODE="offline",
                          TABCOMPLETE_SWEEP_DEADLINE=str(DEADLINE))
        for mode in spec["runner_modes"]:
            stage = "comparison_" + mode
            status.update(state="running", stage=stage,
                          elapsed_session_seconds=time.monotonic() - START)
            (OUT / "worker-status.json").write_text(json.dumps(status, indent=2) + "\n")
            run([sys.executable, str(REPO / "scripts/run_sweep_comparison.py"),
                 *arguments, "--mode", mode], "comparison-" + mode, REPO)
        status["state"] = "complete"
    except Exception as exc:
        status.update(state="failed", failure_stage=stage, error_type=type(exc).__name__)
    finally:
        status["elapsed_session_seconds"] = time.monotonic() - START
        (OUT / "worker-status.json").write_text(json.dumps(status, indent=2) + "\n")
    if status["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
