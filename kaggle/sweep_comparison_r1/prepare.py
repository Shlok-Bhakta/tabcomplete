"""Prepare or submit one private Sweep benchmark using existing Kaggle helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle/one_line_gpu_pilot_r1"))
from build_pilot import _job_statuses_verified, live_quota  # noqa: E402

SPEC = importlib.util.spec_from_file_location("sweep_worker", Path(__file__).with_name("run.py"))
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
DATASET_ID = "shlokbhakta/tabcomplete-sweep-comparison-inputs-r1"
KERNEL_ID = "shlokbhakta/tabcomplete-sweep-comparison-r1"
LINE_SHA = "2eb55e7db35957007572cb15db2d27cd597b25322e85ae776ff4e723ddec2ead"


def cli(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, timeout=900)
    if result.returncode:
        raise RuntimeError("Sweep preparation command failed; no automatic retry")
    return result.stdout.strip()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def checked_quota(expected_renewal: str) -> dict:
    value = live_quota()
    if (
        not _job_statuses_verified(value) or value.get("active_jobs")
        or value.get("renewal") != expected_renewal
        or value.get("remaining") is None or float(value["remaining"]) < 4
    ):
        raise RuntimeError("Sweep authenticated quota/job/renewal gate failed")
    return value


def validate_plan(plan: dict, runner: Path, fixtures: Path) -> None:
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    digest = hashlib.sha256((json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode()).hexdigest()
    if plan.get("schema") != "sweep-comparison-plan-v1" or plan.get("plan_sha256") != digest:
        raise ValueError("expected an intact frozen Sweep comparison plan")
    if (plan["code"]["runner_sha256"] != worker.digest(runner)
            or plan["comparison"]["next_edit"]["fixture_input_sha256"] != worker.digest(fixtures)):
        raise ValueError("Sweep frozen source or fixture identity mismatch")


def prepare(
    plan: Path, line_suite: Path, next_edit_fixtures: Path, output: Path, q4_artifact: Path
) -> None:
    if output.exists():
        raise FileExistsError("Sweep bundle already exists; inspect before resuming")
    if worker.digest(line_suite) != LINE_SHA:
        raise ValueError("corrected 180-case line fixture identity changed")
    if (q4_artifact.stat().st_size != worker.Q4_BYTES
            or worker.digest(q4_artifact) != worker.Q4_SHA256):
        raise ValueError("canonical Q4 artifact identity mismatch")
    commit = cli(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    runner = ROOT / "scripts/run_sweep_comparison.py"
    committed_runner = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", commit + ":scripts/run_sweep_comparison.py"]
    )
    if committed_runner != runner.read_bytes():
        raise ValueError("Sweep runner must be committed before preparing GPU inputs")
    remote = cli(["git", "-C", str(ROOT), "ls-remote", "origin",
                  "refs/heads/prototype/product-r2"]).split()[0]
    if remote != commit:
        raise ValueError("Sweep worker source commit must be pushed")
    plan_value = json.loads(plan.read_text())
    # The serving runner validates its full model/prompt/runtime/fixture plan.
    # This check also catches selecting an unrelated JSON file accidentally.
    validate_plan(plan_value, runner, next_edit_fixtures)

    temporary = output.with_name(output.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError("incomplete Sweep preparation already exists")
    dataset = temporary / "dataset"
    kernel = temporary / "kernel"
    dataset.mkdir(parents=True, mode=0o700)
    kernel.mkdir(mode=0o700)
    shutil.copyfile(plan, dataset / "plan.json")
    shutil.copyfile(line_suite, dataset / "causal_line_v1-r3.jsonl")
    shutil.copyfile(next_edit_fixtures, dataset / "next_edit_inputs.jsonl")
    # Hardlink avoids a second local weight copy; Kaggle receives a private dataset.
    (dataset / worker.Q4_FILE).hardlink_to(q4_artifact)
    spec = {
        "schema": "sweep-comparison-kaggle-input-v1", "commit": commit,
        "model_revision": worker.MODEL_REVISION, "model_sha256": worker.MODEL_SHA256,
        "runtime_revision": worker.RUNTIME_REVISION, "session_seconds": 7200,
        "reserve_seconds": 1200, "training_enabled": False,
        "runner_sha256": worker.digest(runner),
        "files": {name: worker.digest(dataset / name)
                  for name in ("plan.json", "causal_line_v1-r3.jsonl",
                               "next_edit_inputs.jsonl", worker.Q4_FILE)},
        "runner_arguments": [
            "--plan", "{input}/plan.json", "--runtime", "{runtime}",
            "--artifact-dir", "{scratch}", "--artifact-budget-root", "/kaggle/temp",
            "--output", "{output}/results", "--line-suite", "{input}/causal_line_v1-r3.jsonl",
            "--gpu-layers", "99",
            "--next-edit-fixtures", "{input}/next_edit_inputs.jsonl",
        ],
        "runner_modes": ["download", "quantize", "quality", "next-edit"],
    }
    worker.validate_spec(spec)
    write(dataset / "sweep-worker-spec.json", spec)
    write(dataset / "dataset-metadata.json", {
        "id": DATASET_ID, "title": "TabComplete Sweep comparison inputs R1",
        "licenses": [{"name": "other"}],
        "description": "Private benchmark fixtures, frozen metadata and the exact Apache-2.0 "
                       "Sweep Q8-to-Q4 derivative used locally. No personal editor data, "
                       "teacher keys or training examples. Not a public model release.",
    })
    shutil.copyfile(Path(__file__).with_name("run.py"), kernel / "run.py")
    write(kernel / "kernel-metadata.json", {
        "id": KERNEL_ID, "title": "TabComplete Sweep comparison R1",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
    })
    write(temporary / "bundle-manifest.json", {
        "commit": commit, "plan_sha256": worker.digest(plan),
        "input_spec_sha256": worker.digest(dataset / "sweep-worker-spec.json"),
        "worker_sha256": worker.digest(kernel / "run.py"),
        "state": "prepared", "renewal": "2026-10-03T00:00:00",
    })
    temporary.rename(output)


def submit(bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    state_path = bundle / "submission-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"state": "prepared"}
    if state["state"] in {"submission_started", "kernel_pushed"}:
        raise RuntimeError("Sweep GPU submission already attempted; reconcile, never retry blindly")
    if (
        worker.digest(bundle / "dataset/plan.json") != manifest["plan_sha256"]
        or worker.digest(bundle / "dataset/sweep-worker-spec.json") != manifest["input_spec_sha256"]
        or worker.digest(bundle / "kernel/run.py") != manifest["worker_sha256"]
    ):
        raise ValueError("Sweep staged bundle was modified")
    spec = json.loads((bundle / "dataset/sweep-worker-spec.json").read_text())
    worker.validate_spec(spec)
    for name, sha in spec["files"].items():
        if worker.digest(bundle / "dataset" / name) != sha:
            raise ValueError("Sweep staged dataset input was modified")
    quota = checked_quota(manifest["renewal"])
    if state["state"] == "prepared":
        datasets = json.loads(cli(["kaggle", "datasets", "list", "--mine", "--page-size", "100",
                                   "--format", "json"]))
        if any(row.get("ref") == DATASET_ID for row in datasets):
            raise RuntimeError("Sweep dataset already exists; reconcile publication state")
        cli(["kaggle", "datasets", "create", "-p", str(bundle / "dataset"), "-t"])
        state = {"state": "dataset_created"}
        write(state_path, state)
    quota = checked_quota(manifest["renewal"])
    kernels = json.loads(cli(["kaggle", "kernels", "list", "--mine", "--page-size", "100",
                              "--format", "json"]))
    if any(row.get("ref") == KERNEL_ID for row in kernels):
        raise RuntimeError("Sweep kernel already exists; reconcile instead of duplicating")
    write(state_path, {"state": "submission_started", "quota_before": quota,
                       "submitted_at": datetime.now(UTC).isoformat()})
    cli(["kaggle", "kernels", "push", "-p", str(bundle / "kernel"),
         "--timeout", "7200", "--accelerator", "NvidiaTeslaT4"])
    write(state_path, {"state": "kernel_pushed", "kernel_id": KERNEL_ID,
                       "quota_before": quota, "submitted_at": datetime.now(UTC).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--line-suite", type=Path)
    parser.add_argument("--next-edit-fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q4-artifact", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit(args.output)
    elif args.plan and args.line_suite and args.next_edit_fixtures and args.q4_artifact:
        prepare(args.plan, args.line_suite, args.next_edit_fixtures, args.output, args.q4_artifact)
    else:
        parser.error("preparation needs --plan, --line-suite, "
                     "--next-edit-fixtures and --q4-artifact")
