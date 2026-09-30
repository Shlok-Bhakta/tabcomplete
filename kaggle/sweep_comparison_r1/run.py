"""Bounded, private Kaggle launcher for the authorized Sweep GGUF comparison.

The input spec and runner are prepared on CPU before allocating a GPU. No
training or additional model families are used. Setup, compilation and failures
consume the same session deadline as inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

START = float(os.environ.get("TABCOMPLETE_SWEEP_SESSION_START", time.monotonic()))
DEFAULT_SESSION_SECONDS = 7200
SESSION_SECONDS = DEFAULT_SESSION_SECONDS
FIFTH_SESSION_SECONDS = 4800
SIXTH_SESSION_SECONDS = 7200
RESERVE_SECONDS = 1200
DEADLINE = START + SESSION_SECONDS - RESERVE_SECONDS
RUNTIME_REVISION = "f072b103714dfa1eee531f80b24512faf38e3dd2"
MODEL_REVISION = "409016591c6c1a94f545f22328a85a3516118f34"
MODEL_SHA256 = "1321ea5e5d7529e60f9770c6a0b3a965f89542d16cf4ae51bab267f6a88150da"
Q4_FILE = "sweep-next-edit-1.5b.q4_k_m.gguf"
Q4_SHA256 = "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4"
Q4_BYTES = 883_289_056
OUT = Path("/kaggle/working/sweep_comparison_r1")
REPO = Path("/kaggle/temp/tabcomplete-sweep")
RUNTIME = Path("/kaggle/temp/llama-sweep")
STORAGE_ROOT = Path("/kaggle/temp")
STORAGE_CAP = 12 * 1024**3
DISK_RESERVE = 2 * 1024**3


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


def check_storage(additional_bytes: int = 0) -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p in STORAGE_ROOT.rglob("*")
                if p.is_file() and not p.is_symlink())
    if total + additional_bytes > STORAGE_CAP:
        raise RuntimeError("Sweep temporary artifact cap exceeded")
    if shutil.disk_usage(STORAGE_ROOT).free - additional_bytes < DISK_RESERVE:
        raise RuntimeError("Sweep filesystem reserve would be violated")


def run(argv: list[str], label: str, cwd: Path | None = None) -> None:
    if STORAGE_ROOT.exists():
        check_storage()
    # Never include raw process exception messages or credentials in status.
    with (OUT / (label + ".log")).open("w") as handle:
        timeout_deadline = time.monotonic() + remaining_seconds()
        process = subprocess.Popen(
            argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                try:
                    left = min(remaining_seconds(), timeout_deadline - time.monotonic())
                    if left <= 0:
                        raise TimeoutError("Sweep stage exceeded session deadline")
                    process.wait(timeout=min(15, left))
                except subprocess.TimeoutExpired:
                    if STORAGE_ROOT.exists():
                        check_storage()
        except (TimeoutError, RuntimeError):
            # Include owned compiler/server children when enforcing the budget.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise
    if process.returncode:
        raise RuntimeError(label + " failed; inspect private stage log")


def validate_spec(spec: dict) -> None:
    attempt = spec.get("campaign_attempt_number")
    expected_session = FIFTH_SESSION_SECONDS if attempt == 5 else DEFAULT_SESSION_SECONDS
    expected_max_attempts = 6 if attempt == 6 else 5 if attempt == 5 else 4 if attempt == 4 else 3
    if (
        spec.get("schema") != "sweep-comparison-kaggle-input-v1"
        or spec.get("model_revision") != MODEL_REVISION
        or spec.get("model_sha256") != MODEL_SHA256
        or spec.get("runtime_revision") != RUNTIME_REVISION
        or isinstance(attempt, bool)
        or not isinstance(attempt, int) or not 1 <= attempt <= 6
        or not isinstance(spec.get("files"), dict)
        or spec.get("session_seconds") != expected_session
        or spec.get("reserve_seconds") != RESERVE_SECONDS
        or spec.get("campaign_max_attempts") != expected_max_attempts
        or (attempt == 5 and (
            not isinstance(spec.get("fifth_attempt_plan_sha256"), str)
            or len(spec["fifth_attempt_plan_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in spec["fifth_attempt_plan_sha256"])
            or spec["files"].get("plan.json") != spec.get("fifth_attempt_plan_sha256")
        ))
        or (attempt != 5 and spec.get("fifth_attempt_plan_sha256") is not None)
        or (attempt == 6 and (
            not isinstance(spec.get("sixth_attempt_plan_sha256"), str)
            or len(spec["sixth_attempt_plan_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in spec["sixth_attempt_plan_sha256"])
            or spec["files"].get("plan.json") != spec["sixth_attempt_plan_sha256"]
        ))
        or (attempt != 6 and spec.get("sixth_attempt_plan_sha256") is not None)
        or spec.get("training_enabled") is not False
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


def configure_session(spec: dict) -> None:
    """Apply only the session size authorized by the validated campaign attempt."""
    validate_spec(spec)
    global SESSION_SECONDS, DEADLINE
    SESSION_SECONDS = spec["session_seconds"]
    DEADLINE = START + SESSION_SECONDS - RESERVE_SECONDS


def validate_fifth_attempt_plan(plan_path: Path, spec: dict) -> None:
    """Bind the reduced fifth-attempt worker budget to frozen plan-v9."""
    if spec.get("campaign_attempt_number") != 5:
        raise ValueError("reduced Sweep worker session is only for attempt 5")
    plan_hash = digest(plan_path)
    if (
        plan_hash != spec.get("fifth_attempt_plan_sha256")
        or spec.get("files", {}).get("plan.json") != plan_hash
    ):
        raise ValueError("Sweep fifth-attempt plan digest mismatch")
    plan = json.loads(plan_path.read_text())
    plan_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    canonical = (json.dumps(
        plan_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != plan.get("plan_sha256"):
        raise ValueError("Sweep fifth-attempt plan self-hash mismatch")
    amendment = plan.get("allocation_amendment_v9")
    if (
        plan.get("schema") != "sweep-comparison-plan-v1"
        or plan.get("revision", {}).get("number") != 2
        or not isinstance(plan.get("code"), dict)
        or not isinstance(amendment, dict)
        or amendment.get("schema") != "sweep-allocation-amendment-v9"
        or amendment.get("number") != 9
        or plan.get("code", {}).get("runner_sha256") != spec.get("runner_sha256")
        or spec.get("session_seconds") != FIFTH_SESSION_SECONDS
        or spec.get("reserve_seconds") != RESERVE_SECONDS
    ):
        raise ValueError("Sweep fifth-attempt plan and worker configuration disagree")
    limits = amendment.get("attempt_limits")
    expected_limits = {
        "maximum_total_attempts": 5,
        "attempts_already_used": 4,
        "additional_attempts_remaining": 1,
        "per_attempt_session_wall_seconds_max": FIFTH_SESSION_SECONDS,
        "finalization_reserve_seconds": RESERVE_SECONDS,
        "maximum_work_seconds": FIFTH_SESSION_SECONDS - RESERVE_SECONDS,
        "campaign_aggregate_gpu_session_wall_seconds_max": 14400,
        "fifth_attempt_allowed": True,
    }
    if limits != expected_limits or spec.get("campaign_max_attempts") != 5:
        raise ValueError("Sweep fifth-attempt allocation limits are not authorized")
    prior = amendment.get("verified_prior_attempts")
    expected_ids = [
        "shlokbhakta/tabcomplete-sweep-comparison-r1",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-retry",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-verified",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-final",
    ]
    if (
        not isinstance(prior, list) or len(prior) != 4
        or any(not isinstance(row, dict) for row in prior)
        or [row.get("attempt_number") for row in prior] != [1, 2, 3, 4]
        or [row.get("kernel_id") for row in prior] != expected_ids
        or any(row.get("state") != "ERROR" for row in prior)
    ):
        raise ValueError("Sweep fifth-attempt plan lacks four terminal-error predecessors")
    prior_wall = amendment.get("prior_wall_upper_bound_seconds")
    aggregate = amendment.get("aggregate_wall_upper_bound_with_fifth_attempt_seconds")
    margin = amendment.get("remaining_aggregate_wall_margin_seconds")
    try:
        prior_values = [float(row["conservative_wall_upper_bound_seconds"]) for row in prior]
        if any(not math.isfinite(value) or value < 0 for value in prior_values):
            raise ValueError("negative or non-finite prior duration")
        measured_prior_wall = sum(prior_values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Sweep fifth-attempt predecessor time evidence is invalid") from exc
    if (
        isinstance(prior_wall, bool) or not isinstance(prior_wall, (int, float))
        or not math.isfinite(float(prior_wall)) or prior_wall < 0
        or round(measured_prior_wall, 6) != round(float(prior_wall), 6)
        or round(float(prior_wall) + FIFTH_SESSION_SECONDS, 6) != aggregate
        or not isinstance(aggregate, (int, float)) or aggregate > 14400
        or round(14400 - float(aggregate), 6) != margin
        or round(float(spec.get("prior_wall_upper_bound_seconds", -1)), 6)
        != round(float(prior_wall), 6)
        or spec.get("campaign_wall_cap_seconds") != 14400
    ):
        raise ValueError("Sweep fifth-attempt aggregate wall-time budget is inconsistent")


def validate_sixth_attempt_plan(plan_path: Path, spec: dict) -> None:
    """Bind the single sixth allocation to plan-v10 and its six-hour cap."""
    if spec.get("campaign_attempt_number") != 6:
        raise ValueError("sixth Sweep session is only for attempt 6")
    plan_hash = digest(plan_path)
    if (
        plan_hash != spec.get("sixth_attempt_plan_sha256")
        or spec.get("files", {}).get("plan.json") != plan_hash
    ):
        raise ValueError("Sweep sixth-attempt plan digest mismatch")
    plan = json.loads(plan_path.read_text())
    plan_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    canonical = (json.dumps(
        plan_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != plan.get("plan_sha256"):
        raise ValueError("Sweep sixth-attempt plan self-hash mismatch")
    amendment = plan.get("allocation_amendment_v10")
    code = plan.get("code")
    if (
        plan.get("schema") != "sweep-comparison-plan-v1"
        or plan.get("revision", {}).get("number") != 2
        or not isinstance(code, dict)
        or not isinstance(amendment, dict)
        or amendment.get("schema") != "sweep-allocation-amendment-v10"
        or amendment.get("number") != 10
        or code.get("runner_sha256") != spec.get("runner_sha256")
        or spec.get("session_seconds") != SIXTH_SESSION_SECONDS
        or spec.get("reserve_seconds") != RESERVE_SECONDS
        or spec.get("campaign_max_attempts") != 6
        or spec.get("campaign_wall_cap_seconds") != 21600
    ):
        raise ValueError("Sweep sixth-attempt plan and worker configuration disagree")
    native_provider_sha = code.get("native_provider_sha256")
    if (
        not isinstance(native_provider_sha, str)
        or len(native_provider_sha) != 64
        or any(c not in "0123456789abcdef" for c in native_provider_sha)
    ):
        raise ValueError("Sweep plan-v10 lacks the pinned native provider source")

    limits = amendment.get("attempt_limits")
    expected_limits = {
        "maximum_total_attempts": 6,
        "attempts_already_used": 5,
        "additional_attempts_remaining": 1,
        "per_attempt_session_wall_seconds_max": SIXTH_SESSION_SECONDS,
        "finalization_reserve_seconds": RESERVE_SECONDS,
        "maximum_work_seconds": SIXTH_SESSION_SECONDS - RESERVE_SECONDS,
        "campaign_aggregate_gpu_session_wall_seconds_max": 21600,
        "sixth_attempt_allowed": True,
    }
    if limits != expected_limits:
        raise ValueError("Sweep plan-v10 does not authorize exactly one sixth attempt")
    if amendment.get("authorization") != {
        "basis": "User explicitly authorized one additional free Kaggle allocation.",
        "additional_model_downloads": False,
        "automatic_renewal_consumption": False,
        "maximum_additional_allocations": 1,
        "paid_compute": False,
        "training": False,
    }:
        raise ValueError("Sweep plan-v10 allocation is not within the authorized scope")
    if amendment.get("quota_gate") != {
        "all_relevant_job_statuses_verified": True,
        "automatic_renewal_consumption": False,
        "minimum_remaining_gpu_hours": 2.0,
        "refresh_immediately_before_submission": True,
        "require_no_active_gpu_jobs": True,
        "require_same_renewal_as_campaign": True,
    }:
        raise ValueError("Sweep plan-v10 quota gate is invalid")
    observed_quota = amendment.get("fresh_quota_observation_at_freeze")
    if (
        not isinstance(observed_quota, dict)
        or observed_quota.get("source")
        != "authenticated kaggle quota --format json and kernel statuses"
        or observed_quota.get("renewal")
        != plan.get("quota_observation", {}).get("renewal")
        or observed_quota.get("active_jobs") != []
        or observed_quota.get("all_relevant_job_statuses_verified") is not True
        or isinstance(observed_quota.get("remaining"), bool)
        or not isinstance(observed_quota.get("remaining"), (int, float))
        or not math.isfinite(float(observed_quota["remaining"]))
        or float(observed_quota["remaining"]) < 2.0
    ):
        raise ValueError("Sweep plan-v10 frozen quota observation is missing or invalid")
    failure_evidence = amendment.get("attempt_005_failure_evidence")
    if (
        not isinstance(failure_evidence, dict)
        or failure_evidence.get("kernel_id")
        != "shlokbhakta/tabcomplete-sweep-comparison-r1-attempt-5"
        or failure_evidence.get("state") != "ERROR"
        or failure_evidence.get("predictions") != 0
        or failure_evidence.get("model_quality_assessed") is not False
        or failure_evidence.get("conservative_wall_upper_bound_seconds") != 2013.138070
        or not isinstance(failure_evidence.get("manifest_sha256"), str)
        or len(failure_evidence["manifest_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in failure_evidence["manifest_sha256"])
        or not isinstance(failure_evidence.get("terminal_observation_sha256"), str)
        or len(failure_evidence["terminal_observation_sha256"]) != 64
        or any(
            c not in "0123456789abcdef"
            for c in failure_evidence["terminal_observation_sha256"]
        )
    ):
        raise ValueError("Sweep plan-v10 fifth-attempt failure metadata is invalid")
    if amendment.get("session_enforcement") != {
        "finalization_reserve_seconds": RESERVE_SECONDS,
        "kaggle_push_timeout_seconds": SIXTH_SESSION_SECONDS,
        "monotonic_worker_work_deadline_seconds": SIXTH_SESSION_SECONDS - RESERVE_SECONDS,
        "native_per_request_timeout_seconds_max": 120,
        "setup_compile_evaluation_and_saving_included": True,
        "no_automatic_seventh_allocation": True,
    }:
        raise ValueError("Sweep plan-v10 session enforcement is invalid")

    prior = amendment.get("verified_prior_attempts")
    expected_ids = [
        "shlokbhakta/tabcomplete-sweep-comparison-r1",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-retry",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-verified",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-final",
        "shlokbhakta/tabcomplete-sweep-comparison-r1-attempt-5",
    ]
    if (
        not isinstance(prior, list) or len(prior) != 5
        or any(not isinstance(row, dict) for row in prior)
        or [row.get("attempt_number") for row in prior] != [1, 2, 3, 4, 5]
        or [row.get("kernel_id") for row in prior] != expected_ids
        or any(row.get("state") != "ERROR" for row in prior)
    ):
        raise ValueError("Sweep plan-v10 lacks five terminal-error predecessors")
    prior_wall = amendment.get("prior_wall_upper_bound_seconds")
    aggregate = amendment.get("aggregate_wall_upper_bound_with_sixth_attempt_seconds")
    margin = amendment.get("remaining_aggregate_wall_margin_seconds")
    try:
        values = [float(row["conservative_wall_upper_bound_seconds"]) for row in prior]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid duration")
        measured_prior = sum(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Sweep plan-v10 predecessor time evidence is invalid") from exc
    expected_total = round(measured_prior + SIXTH_SESSION_SECONDS, 6)
    if (
        isinstance(prior_wall, bool) or not isinstance(prior_wall, (int, float))
        or not math.isfinite(float(prior_wall))
        or round(measured_prior, 6) != round(float(prior_wall), 6)
        or round(expected_total, 6) != aggregate
        or not isinstance(aggregate, (int, float)) or aggregate > 21600
        or round(21600 - float(aggregate), 6) != margin
        or round(float(spec.get("prior_wall_upper_bound_seconds", -1)), 6)
        != round(float(prior_wall), 6)
        or round(float(prior_wall), 6) != 11394.350495
        or round(float(aggregate), 6) != 18594.350495
        or round(float(margin), 6) != 3005.649505
    ):
        raise ValueError("Sweep sixth-attempt aggregate wall-time budget is inconsistent")


def stage_canonical_q4(input_dir: Path, scratch: Path) -> None:
    """Reuse the exact privately staged derivative, without GPU requantization."""
    source = input_dir / Q4_FILE
    if source.stat().st_size != Q4_BYTES or digest(source) != Q4_SHA256:
        raise ValueError("canonical Q4 artifact identity mismatch")
    check_storage(Q4_BYTES)
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / Q4_FILE
    if target.exists():
        if target.stat().st_size != Q4_BYTES or digest(target) != Q4_SHA256:
            raise ValueError("existing worker Q4 artifact identity mismatch")
        return
    shutil.copyfile(source, target)
    if digest(target) != Q4_SHA256:
        raise ValueError("worker Q4 staging hash mismatch")


def cuda_driver_configuration() -> tuple[list[str], dict]:
    """Locate the mounted driver; use supported non-VMM CUDA if it is absent."""
    try:
        value = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10)
        for line in value.stdout.splitlines():
            if line.strip().startswith("libcuda.so") and " => " in line:
                path = Path(line.split(" => ", 1)[1].strip())
                if path.is_file():
                    return ["-DCUDA_cuda_driver_LIBRARY=" + str(path)], {
                        "driver_library": str(path), "virtual_memory_management": True,
                    }
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ["-DGGML_CUDA_NO_VMM=ON"], {
        "driver_library": "not resolved through ldconfig",
        "virtual_memory_management": False,
        "reason": "Use the pinned runtime's supported cudaMalloc allocation path",
    }


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
        if spec["campaign_attempt_number"] == 5:
            validate_fifth_attempt_plan(input_dir / "plan.json", spec)
        if spec["campaign_attempt_number"] == 6:
            validate_sixth_attempt_plan(input_dir / "plan.json", spec)
        configure_session(spec)
        status.update(session_seconds=SESSION_SECONDS, reserve_seconds=RESERVE_SECONDS)
        for name, sha in spec["files"].items():
            if digest(input_dir / name) != sha:
                raise ValueError("Sweep input artifact hash mismatch")
        status["input_spec_sha256"] = digest(specs[0])
        check_storage(1024**3)
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
        check_storage(1024**3)
        run(["git", "clone", "--filter=blob:none", "https://github.com/Shlok-Bhakta/tabcomplete.git",
             str(REPO)], "repo-clone")
        run(["git", "checkout", spec["commit"]], "repo-checkout", REPO)
        if digest(REPO / "scripts/run_sweep_comparison.py") != spec["runner_sha256"]:
            raise ValueError("Sweep runner revision mismatch")
        stage = "runtime_build"
        check_storage(4 * 1024**3)
        run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git",
             str(RUNTIME)], "runtime-clone")
        run(["git", "checkout", RUNTIME_REVISION], "runtime-checkout", RUNTIME)
        driver_flags, driver_configuration = cuda_driver_configuration()
        driver_configuration["cmake_driver_flags"] = driver_flags
        (OUT / "cuda-driver-configuration.json").write_text(
            json.dumps(driver_configuration, indent=2) + "\n"
        )
        run(["cmake", "-S", str(RUNTIME), "-B", str(RUNTIME / "build"),
             "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF",
             "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DCMAKE_CUDA_ARCHITECTURES=75",
             *driver_flags],
            "runtime-configure")
        run(["cmake", "--build", str(RUNTIME / "build"), "--target", "llama-server",
             "llama-quantize", "-j", "2"], "runtime-build")
        versions = {"python": sys.version, "platform": platform.platform()}
        for label, command in {
            "compiler": ["c++", "--version"], "cuda": ["nvcc", "--version"],
            "cmake": ["cmake", "--version"], "packages": [sys.executable, "-m", "pip", "freeze"],
        }.items():
            try:
                value = subprocess.run(command, capture_output=True, text=True, timeout=30)
                versions[label] = value.stdout.strip() if value.returncode == 0 else "unavailable"
            except (OSError, subprocess.TimeoutExpired):
                versions[label] = "unavailable"
        (OUT / "runtime-environment.json").write_text(json.dumps(versions, indent=2) + "\n")
        stage = "canonical_q4_staging"
        stage_canonical_q4(input_dir, Path("/kaggle/temp/sweep-artifacts"))
        stage = "comparison"
        arguments = [part.replace("{input}", str(input_dir)).replace("{output}", str(OUT))
                     .replace("{runtime}", str(RUNTIME))
                     .replace("{scratch}", "/kaggle/temp/sweep-artifacts")
                     for part in spec["runner_arguments"]]
        os.environ.update(
            PYTHONPATH=str(REPO / "src"), TABCOMPLETE_OBSERVABILITY_MODE="offline",
            TABCOMPLETE_OBSERVABILITY_ENABLED="1",
            TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT="1",
            TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE=str(OUT / "telemetry/offline.jsonl"),
            TABCOMPLETE_OBSERVABILITY_ARTIFACT_ROOT=str(OUT / "telemetry/artifacts"),
            TABCOMPLETE_OBSERVABILITY_OFFLINE_MAX_BYTES=str(128 * 1024**2),
            TABCOMPLETE_SWEEP_DEADLINE=str(DEADLINE),
        )
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
