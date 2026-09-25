"""Frozen, storage-bounded small-model campaign entry point.

The entry point records an immutable plan before any new model output. GPU
allocation is deliberately a separate phase; this CPU process never allocates
one while preparing data or documentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/small_model_prototype_r1"
R2 = ROOT.parent / "tabcomplete-model-data-r2"
R2_ARTIFACTS = R2 / "artifacts/research/model_data_r2"
ALLOWED = {
    "q25-coder": "Qwen/Qwen2.5-Coder-0.5B",
    "q3-base": "Qwen/Qwen3-0.6B-Base",
    "lfm350-base": "LiquidAI/LFM2.5-350M-Base",
    "p12-control": "local-r2-validated",
}
EXISTING = {
    "q25-coder-q4": R2_ARTIFACTS / "q25-Q4_K_M.gguf",
    "p12-control-q4": R2_ARTIFACTS / "p12-text-Q4_K_M.gguf",
    "p12-control-fp16": ROOT.parent
    / "tabcomplete/outputs/kaggle/code_cpt_train_v3/code_cpt_run/main/final/model.safetensors",
}
FIXTURES = {
    "causal_200": ROOT / "data/benchmarks/code_completion_v2.jsonl",
    "line_180": R2_ARTIFACTS / "frozen-corpora/causal_line_v1-r3.jsonl",
    "next_edit_v2_reserved": ROOT / "data/benchmarks/next_edit_v2.jsonl",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command(*args: str, timeout: int = 60) -> str:
    done = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode:
        raise RuntimeError(f"{args[0]} exited {done.returncode}")
    return done.stdout


def quota() -> dict:
    observed_at = datetime.now(UTC).isoformat()
    try:
        data = json.loads(command("kaggle", "quota", "--format", "json"))
        gpu = next(row for row in data if row["resource"] == "GPU")
        result = {
            "remaining": float(gpu["remaining"].rstrip("h")),
            "used": float(gpu["used"].rstrip("h")),
            "total": float(gpu["total"].rstrip("h")),
            "renewal": gpu["refreshAt"],
            "units": "Kaggle account-hours",
        }
    except (RuntimeError, ValueError, KeyError, StopIteration):
        result = {"remaining": None, "renewal": None, "units": "Kaggle account-hours"}
    try:
        listed = csv.DictReader(
            io.StringIO(
                command("kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv")
            )
        )
        jobs = []
        for row in listed:
            try:
                status = command("kaggle", "kernels", "status", row["ref"]).strip()
            except RuntimeError:
                status = "unknown"
            jobs.append({"reference": row["ref"], "status": status})
    except RuntimeError:
        jobs = [{"reference": "unknown", "status": "unknown"}]
    result.update(
        observed_at=observed_at,
        source="authenticated Kaggle CLI quota and kernel status",
        jobs=jobs,
        active_jobs=[
            job for job in jobs if any(s in job["status"].upper() for s in ("RUNNING", "QUEUED"))
        ],
    )
    return result


def hub_metadata(alias: str, model_id: str) -> dict:
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"https://huggingface.co/api/models/{model_id}", params={"blobs": "true"}
        )
        response.raise_for_status()
        data = response.json()
    files = [
        {"name": row["rfilename"], "bytes": row.get("size")}
        for row in data["siblings"]
        if row["rfilename"].endswith((".safetensors", ".json", ".model"))
    ]
    if not files or not data.get("safetensors", {}).get("total"):
        raise ValueError(f"{alias}: missing total parameters or remote file sizes")
    return {
        "source": model_id,
        "revision": data["sha"],
        "tokenizer_revision": data["sha"],
        "total_parameters": data["safetensors"]["total"],
        "license": data.get("cardData", {}).get("license"),
        "remote_files": files,
    }


def freeze(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if config["models"] != ALLOWED:
        raise ValueError("model allowlist changed")
    path = REPORT / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["config_sha256"] != digest(config_path):
            raise ValueError("configuration changed after freeze; register a new plan revision")
        for inventory in ("existing_artifacts", "fixtures"):
            for item in plan[inventory].values():
                file = Path(item["path"])
                if not file.is_file() or digest(file) != item["sha256"]:
                    raise ValueError(f"{inventory} changed after freeze: {file}")
        runtime = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp"
        environment = plan["environment"]
        if (
            platform.node() != environment["host"]
            or platform.python_version() != environment["python"]
            or digest(ROOT / "uv.lock") != environment["uv_lock_sha256"]
            or command("git", "-C", str(runtime), "rev-parse", "HEAD").strip()
            != environment["runtime_revision"]
            or digest(runtime / "build/bin/llama-server")
            != environment["runtime_binary_sha256"]
        ):
            raise ValueError("frozen environment changed; register a new plan revision")
        return plan
    existing = {
        alias: {"path": str(file), "bytes": file.stat().st_size, "sha256": digest(file)}
        for alias, file in EXISTING.items()
        if file.exists()
    }
    fixtures = {
        alias: {"path": str(file), "bytes": file.stat().st_size, "sha256": digest(file)}
        for alias, file in FIXTURES.items()
    }
    models = {
        alias: hub_metadata(alias, model_id)
        for alias, model_id in ALLOWED.items()
        if alias != "p12-control"
    }
    models["p12-control"] = {"source": "existing validated R2 P12 checkpoint", "files": existing}
    observed_quota = quota()
    runtime = ROOT.parent / "tabcomplete/outputs/tools/llama.cpp"
    plan = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_commit": config["source_commit"],
        "config_sha256": digest(config_path),
        "models": models,
        "existing_artifacts": existing,
        "fixtures": fixtures,
        "quota": observed_quota,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runtime_revision": command("git", "-C", str(runtime), "rev-parse", "HEAD").strip(),
            "runtime_binary_sha256": digest(runtime / "build/bin/llama-server"),
            "uv_lock_sha256": digest(ROOT / "uv.lock"),
        },
        "protocol": config["screening"],
        "adaptation": config["adaptation"],
        "gates": config["deployment"],
        "limits": config["limits"],
        "sealed_test_opened": False,
        "historical_results_are_new_measurements": False,
    }
    save(path, plan)
    save(REPORT / "models.json", models)
    save(
        REPORT / "download_manifest.json",
        {"frozen_at": plan["frozen_at"], "downloads": [], "reused": existing},
    )
    for directory in (
        "hardware",
        "screening",
        "interactive_replay",
        "adaptation",
        "deployment",
        "feedback_verification",
    ):
        (REPORT / directory).mkdir(parents=True, exist_ok=True)
    return plan


def check_budget(
    plan: dict, *, session_seconds: int = 0, input_tokens: int = 0, new_bytes: int = 0
) -> None:
    limits = plan["limits"]
    ledger_path = REPORT / "budget_ledger.json"
    ledger: dict[str, Any] = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {
            "gpu_session_seconds": 0,
            "training_input_tokens": 0,
            "new_artifact_bytes": 0,
            "sessions": [],
        }
    )
    if ledger["gpu_session_seconds"] + session_seconds > limits["gpu_session_wall_hours"] * 3600:
        raise RuntimeError("GPU wall-hour budget exceeded")
    if (
        ledger["training_input_tokens"] + input_tokens
        > limits["additional_nonpadding_training_input_tokens"]
    ):
        raise RuntimeError("training token budget exceeded")
    if ledger["new_artifact_bytes"] + new_bytes > limits["new_research_artifact_bytes"]:
        raise RuntimeError("research artifact storage budget exceeded")
    if new_bytes > shutil.disk_usage(REPORT).free - 2 * 1024**3:
        raise RuntimeError("insufficient disk headroom")
    if session_seconds:
        observation = quota()
        if observation["active_jobs"]:
            raise RuntimeError("another Kaggle notebook is active")
        if observation["remaining"] is None:
            if ledger["sessions"] or session_seconds > 7200:
                raise RuntimeError(
                    "unavailable live quota permits only one initial session under two hours"
                )
        elif observation["remaining"] * 3600 < session_seconds:
            raise RuntimeError("insufficient observed Kaggle quota")
        if observation["renewal"] != plan["quota"]["renewal"]:
            raise RuntimeError("renewed allocation requires a new campaign decision")
        if session_seconds <= limits["finalization_reserve_seconds"]:
            raise RuntimeError("session does not leave finalization reserve")


def submit_screening(plan: dict) -> dict:
    revision = plan.get("plan_revision", 1)
    job_path = REPORT / f"screening/job-v{revision}.json"
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if job["plan_sha256"] != digest(REPORT / "plan.json"):
            raise ValueError("screening job belongs to another plan")
        job["observed_status"] = command("kaggle", "kernels", "status", job["reference"]).strip()
        save(job_path, job)
        return job
    check_budget(plan, session_seconds=7200)
    commit = command("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
    if command("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("commit scientific code before Kaggle submission")
    remote = command("git", "ls-remote", "origin", "refs/heads/research/small-model-prototype-r1")
    if remote.split()[0] != commit:
        raise RuntimeError("push the campaign branch before Kaggle submission")
    folder = ROOT / "artifacts/research/small_model_prototype_r1/submission/screening"
    folder.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "kaggle/small_model_prototype_r1/run_screening.py").read_text()
    (folder / "run.py").write_text(
        source.replace("__CHECKOUT_COMMIT__", commit).replace(
            "__PLAN_SHA__", digest(REPORT / "plan.json")
        )
    )
    reference = f"shlokbhakta/tabcomplete-small-model-prototype-r1-screening-v{revision}"
    save(
        folder / "kernel-metadata.json",
        {
            "id": reference,
            "title": reference.split("/")[1],
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["shlokbhakta/tabcomplete-model-data-r2-inputs"],
            "kernel_sources": [],
            "competition_sources": [],
        },
    )
    job = {
        "reference": reference,
        "plan_sha256": digest(REPORT / "plan.json"),
        "commit": commit,
        "submitted_at": datetime.now(UTC).isoformat(),
        "session_seconds_limit": 7200,
        "finalization_reserve_seconds": 900,
        "quota_before": quota(),
        "state": "submission_pending",
    }
    save(job_path, job)
    command("kaggle", "kernels", "push", "-p", str(folder), timeout=180)
    job["state"] = "submitted"
    save(job_path, job)
    return job


def submit_line_repair(plan: dict) -> dict:
    job_path = REPORT / "screening/line-repair-job.json"
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if job["plan_sha256"] != digest(REPORT / "plan.json"):
            raise ValueError("line repair belongs to another plan")
        job["observed_status"] = command("kaggle", "kernels", "status", job["reference"]).strip()
        save(job_path, job)
        return job
    if plan.get("plan_revision") != 3:
        raise ValueError("line repair requires registered plan revision 3")
    check_budget(plan, session_seconds=3600)
    commit = command("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
    if command("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("commit scientific code before Kaggle submission")
    remote = command("git", "ls-remote", "origin", "refs/heads/research/small-model-prototype-r1")
    if remote.split()[0] != commit:
        raise RuntimeError("push the campaign branch before Kaggle submission")
    folder = ROOT / "artifacts/research/small_model_prototype_r1/submission/line-repair"
    folder.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "kaggle/small_model_prototype_r1/run_line_repair.py").read_text()
    (folder / "run.py").write_text(
        source.replace("__CHECKOUT_COMMIT__", commit).replace(
            "__PLAN_SHA__", digest(REPORT / "plan.json")
        )
    )
    reference = "shlokbhakta/tabcomplete-small-model-prototype-r1-line-repair"
    save(
        folder / "kernel-metadata.json",
        {
            "id": reference,
            "title": reference.split("/")[1],
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["shlokbhakta/tabcomplete-model-data-r2-inputs"],
            "kernel_sources": [],
            "competition_sources": [],
        },
    )
    job = {
        "reference": reference,
        "plan_sha256": digest(REPORT / "plan.json"),
        "commit": commit,
        "submitted_at": datetime.now(UTC).isoformat(),
        "session_seconds_limit": 3600,
        "finalization_reserve_seconds": 900,
        "quota_before": quota(),
        "state": "submission_pending",
    }
    save(job_path, job)
    command("kaggle", "kernels", "push", "-p", str(folder), timeout=180)
    job["state"] = "submitted"
    save(job_path, job)
    return job


def submit_adaptation(plan: dict) -> dict:
    selection_path = REPORT / "selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection.get("locked_before_adaptation"):
        raise ValueError("finalist decision must be locked before adaptation")
    finalists = selection["finalists"]
    if finalists[0] != "q25-coder" or len(finalists) > 2 or len(set(finalists)) != len(finalists):
        raise ValueError("invalid finalists")
    if any(alias not in ALLOWED or alias == "p12-control" for alias in finalists):
        raise ValueError("non-allowlisted adaptation finalist")
    job_path = REPORT / "adaptation/job.json"
    fixture_suite = REPORT / "adaptation/fixture_suite-v2.json"
    fixture_suite_sha = digest(fixture_suite) if fixture_suite.exists() else None
    attempt = 1
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if job["plan_sha256"] != digest(REPORT / "plan.json"):
            raise ValueError("adaptation belongs to another plan")
        job["observed_status"] = command("kaggle", "kernels", "status", job["reference"]).strip()
        save(job_path, job)
        retry_completed_diagnostic = (
            "COMPLETE" in job["observed_status"].upper()
            and fixture_suite_sha is not None
            and job.get("fixture_suite_sha256") != fixture_suite_sha
            and json.loads((REPORT / "adaptation/diagnostic-v2.json").read_text())["status"]
            == "partial"
        )
        if "ERROR" not in job["observed_status"].upper() and not retry_completed_diagnostic:
            return job
        attempt = job.get("attempt", 1) + 1
        if attempt > 3:
            raise RuntimeError("adaptation retry limit reached; inspect failed job")
        save(REPORT / f"adaptation/job-v{attempt - 1}.json", job)
        job_path.unlink()
    check_budget(plan, session_seconds=28_800, input_tokens=4_000_000)
    commit = command("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
    if command("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("commit scientific code before Kaggle submission")
    remote = command("git", "ls-remote", "origin", "refs/heads/research/small-model-prototype-r1")
    if remote.split()[0] != commit:
        raise RuntimeError("push the campaign branch before Kaggle submission")
    folder = ROOT / "artifacts/research/small_model_prototype_r1/submission/adaptation"
    folder.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "kaggle/small_model_prototype_r1/run_adaptation.py").read_text()
    (folder / "run.py").write_text(
        source.replace("__CHECKOUT_COMMIT__", commit)
        .replace("__PLAN_SHA__", digest(REPORT / "plan.json"))
        .replace("__SELECTION_SHA__", digest(selection_path))
        .replace("__FIXTURE_SUITE_SHA__", fixture_suite_sha or "")
        .replace("__FINALISTS__", json.dumps(finalists))
    )
    reference = "shlokbhakta/tabcomplete-small-model-prototype-r1-adaptation"
    save(
        folder / "kernel-metadata.json",
        {
            "id": reference,
            "title": reference.split("/")[1],
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
        },
    )
    job = {
        "reference": reference,
        "plan_sha256": digest(REPORT / "plan.json"),
        "selection_sha256": digest(selection_path),
        "fixture_suite_sha256": fixture_suite_sha,
        "finalists": finalists,
        "commit": commit,
        "submitted_at": datetime.now(UTC).isoformat(),
        "session_seconds_limit": 28_800,
        "finalization_reserve_seconds": 1_800,
        "quota_before": quota(),
        "state": "submission_pending",
        "attempt": attempt,
    }
    save(job_path, job)
    command("kaggle", "kernels", "push", "-p", str(folder), timeout=180)
    job["state"] = "submitted"
    save(job_path, job)
    return job


def submit_selected_stage(plan: dict, stage: str) -> dict:
    if stage not in ("heldout", "conversion"):
        raise ValueError("unknown selected-model stage")
    decision_path = REPORT / "deployment/selection.json"
    decision = json.loads(decision_path.read_text())
    finalists = json.loads((REPORT / "selection.json").read_text())["finalists"]
    if (
        not decision.get("locked_before_heldout")
        or decision["plan_sha256"] != digest(REPORT / "plan.json")
        or decision["selected_alias"] not in ("q25-coder", "q3-base")
        or decision["selected_alias"] not in finalists
    ):
        raise ValueError("invalid development-locked deployment decision")
    progress_path = REPORT / "adaptation/progress.json"
    progress = json.loads(progress_path.read_text())
    alias = decision["selected_alias"]
    weight_sha = decision["adapted_weight_sha256"]
    if (
        progress["status"] != "complete"
        or progress["plan_sha256"] != decision["plan_sha256"]
        or progress["models"][alias]["main"]["training"]["inference_weight_sha256"]
        != weight_sha
    ):
        raise ValueError("adaptation result does not match selected checkpoint")
    if stage == "conversion":
        heldout_path = REPORT / "deployment/heldout-result.json"
        heldout = json.loads(heldout_path.read_text())
        if heldout["status"] != "complete" or heldout["selection_sha256"] != digest(decision_path):
            raise ValueError("complete the one locked held-out evaluation before export")
    job_path = REPORT / f"deployment/{stage}-job.json"
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if (
            job["plan_sha256"] != digest(REPORT / "plan.json")
            or job["selection_sha256"] != digest(decision_path)
        ):
            raise ValueError("selected-model job fingerprint changed")
        job["observed_status"] = command("kaggle", "kernels", "status", job["reference"]).strip()
        save(job_path, job)
        return job
    check_budget(plan, session_seconds=7200 if stage == "heldout" else 0,
                 new_bytes=2 * 1024**3 if stage == "conversion" else 0)
    commit = command("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
    if command("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("commit scientific code before Kaggle submission")
    remote = command("git", "ls-remote", "origin", "refs/heads/research/small-model-prototype-r1")
    if remote.split()[0] != commit:
        raise RuntimeError("push the campaign branch before Kaggle submission")
    folder = ROOT / f"artifacts/research/small_model_prototype_r1/submission/{stage}"
    folder.mkdir(parents=True, exist_ok=True)
    source_name = "run_selected_heldout.py" if stage == "heldout" else "run_selected_conversion.py"
    source = (ROOT / "kaggle/small_model_prototype_r1" / source_name).read_text()
    (folder / "run.py").write_text(
        source.replace("__CHECKOUT_COMMIT__", commit)
        .replace("__PLAN_SHA__", digest(REPORT / "plan.json"))
        .replace("__SELECTION_SHA__", digest(decision_path))
        .replace("__SELECTED_ALIAS__", alias)
        .replace("__ADAPTED_WEIGHT_SHA__", weight_sha)
    )
    reference = f"shlokbhakta/tabcomplete-small-model-prototype-r1-selected-{stage}"
    metadata: dict[str, Any] = {
        "id": reference, "title": reference.split("/")[1], "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": stage == "heldout", "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": ["shlokbhakta/tabcomplete-small-model-prototype-r1-adaptation"],
        "competition_sources": [],
    }
    if stage == "heldout":
        metadata["machine_shape"] = "NvidiaTeslaT4"
    save(folder / "kernel-metadata.json", metadata)
    job = {
        "reference": reference, "plan_sha256": digest(REPORT / "plan.json"),
        "selection_sha256": digest(decision_path), "selected_alias": alias,
        "adapted_weight_sha256": weight_sha, "commit": commit,
        "submitted_at": datetime.now(UTC).isoformat(),
        "session_seconds_limit": 7200, "finalization_reserve_seconds": 900,
        "quota_before": quota() if stage == "heldout" else None,
        "state": "submission_pending",
    }
    save(job_path, job)
    command("kaggle", "kernels", "push", "-p", str(folder), timeout=180)
    job["state"] = "submitted"
    save(job_path, job)
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=("auto", "heldout", "conversion"), default="auto")
    args = parser.parse_args()
    plan = freeze(args.config)
    if args.execute:
        if args.stage != "auto":
            job = submit_selected_stage(plan, args.stage)
        elif (REPORT / "selection.json").exists():
            job = submit_adaptation(plan)
        elif plan.get("plan_revision") == 3:
            job = submit_line_repair(plan)
        else:
            job = submit_screening(plan)
        print(
            json.dumps(
                {
                    "plan_sha256": digest(REPORT / "plan.json"),
                    "job": job["reference"],
                    "state": job["state"],
                }
            )
        )
    else:
        print(json.dumps({"plan_sha256": digest(REPORT / "plan.json"), "frozen": True}))


if __name__ == "__main__":
    main()
