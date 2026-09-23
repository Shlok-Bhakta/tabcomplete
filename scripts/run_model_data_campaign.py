"""Bounded R2 stages, immutable preregistration, and explicit Kaggle accounting."""

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
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/model_data_r2"
ARTIFACTS = ROOT / "artifacts/research/model_data_r2"


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(args, timeout=60):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"command {args[0]} failed with exit {result.returncode}")
    return result.stdout


def observe_quota():
    try:
        rows = json.loads(command(["kaggle", "quota", "--format", "json"]))
        gpu = next(r for r in rows if r["resource"] == "GPU")
        quota = {
            "used_account_hours": float(gpu["used"].rstrip("h")),
            "remaining_account_hours": float(gpu["remaining"].rstrip("h")),
            "renewal": gpu["refreshAt"],
            "units": "account quota-hours",
        }
    except Exception:
        quota = {"used_account_hours": None, "remaining_account_hours": None, "renewal": None}
    listed = csv.DictReader(
        io.StringIO(command(["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"]))
    )
    statuses = []
    for row in listed:
        try:
            status = command(["kaggle", "kernels", "status", row["ref"]]).strip()
        except RuntimeError:
            status = "unknown: status request failed"
        statuses.append({"ref": row["ref"], "status": status, "last_run": row.get("lastRunTime")})
    quota.update(
        observed_at=now(),
        source="authenticated kaggle CLI quota and kernel status",
        jobs=statuses,
        active_jobs=[r for r in statuses if any(s in r["status"] for s in ("RUNNING", "QUEUED"))],
    )
    write_json(REPORT / "quota" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + ".json"), quota)
    return quota


def freeze(config_path):
    config = yaml.safe_load(config_path.read_text())
    target = REPORT / "preregistered_plan.json"
    if target.exists():
        plan = json.loads(target.read_text())
        if plan["configuration_sha256"] != digest(config_path):
            raise ValueError("Frozen plan differs: register a documented new plan revision")
        return plan
    quota = observe_quota()
    models = {}
    for alias, spec in config["models"].items():
        if "hub" not in spec:
            continue
        info = httpx.get(
            f"https://huggingface.co/api/models/{spec['hub']}/revision/{spec['revision']}",
            timeout=30,
        )
        info.raise_for_status()
        data = info.json()
        cfg = httpx.get(
            f"https://huggingface.co/{spec['hub']}/resolve/{spec['revision']}/config.json",
            follow_redirects=True,
            timeout=30,
        )
        cfg.raise_for_status()
        fields = cfg.json()
        models[alias] = {
            "id": spec["hub"],
            "revision": data["sha"],
            "license": data.get("cardData", {}).get("license"),
            "hub_total_parameters_including_unused_components": data.get("safetensors", {}).get(
                "total"
            ),
            "text_vocabulary_size": fields.get("text_config", fields).get("vocab_size"),
            "config": fields,
        }
    parent = Path(
        os.environ.get(
            "TABCOMPLETE_R2_PARENT",
            str(
                ROOT.parent / "tabcomplete/outputs/kaggle/code_cpt_train_v3/code_cpt_run/main/final"
            ),
        )
    )
    assert digest(parent / "model.safetensors") == config["models"]["q35-p12"]["model_sha256"]
    assert digest(parent / "tokenizer.json") == config["models"]["q35-p12"]["tokenizer_sha256"]
    assert (
        digest(ROOT / config["evaluation"]["causal"]["path"])
        == config["evaluation"]["causal"]["sha256"]
    )
    plan = {
        "schema_version": 1,
        "frozen_at": now(),
        "configuration_sha256": digest(config_path),
        "configuration": config,
        "resolved_models": models,
        "quota_at_freeze": quota,
        "integration": {
            "research": "9ab49df73a2cff9c8ba340f93f96f4ec1dc58271",
            "observability": "5513296edfb77d4ab2763a293e93969e6a769557",
        },
        "outcomes_inspected_before_freeze": "historical R1 only, no R2 generation or training",
        "context_runtime_inventory": "No separate context-runtime worktree or verified tiling found",
        "thinkpad": "authorized existing alias timed out, not measured",
        "parent_bytes_verified": True,
        "sealed_test_opened": False,
    }
    write_json(target, plan)
    write_json(
        REPORT / "environment/local.json",
        {
            "observed_at": now(),
            "platform": platform.platform(),
            "cpu": command(["lscpu"]),
            "memory": command(["free", "-b"]),
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "python": sys.version,
            "local_cuda_assumed": False,
            "inference_threads": 4,
            "llama_cpp_revision": config["deployment"]["revision"],
        },
    )
    return plan


def submit_baseline(plan):
    state_path = REPORT / "baseline_evaluations/job.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != digest(REPORT / "preregistered_plan.json"):
            raise ValueError("job belongs to another plan")
        print(
            json.dumps(
                {
                    "existing_job": state["reference"],
                    "status": command(["kaggle", "kernels", "status", state["reference"]]).strip(),
                }
            )
        )
        return
    quota = observe_quota()
    if quota["active_jobs"]:
        raise RuntimeError("another notebook allocation is active")
    seconds = 7200
    if (
        quota["remaining_account_hours"] is not None
        and seconds / 3600 > quota["remaining_account_hours"]
    ):
        raise RuntimeError("insufficient verified quota")
    revision = command(["git", "rev-parse", "HEAD"]).strip()
    if command(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise RuntimeError("commit frozen code before submitting")
    folder = ARTIFACTS / "submissions/baseline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(
        (ROOT / "kaggle/model_data_r2/run_baseline.py")
        .read_text()
        .replace("__CHECKOUT_COMMIT__", revision)
    )
    reference = "shlokbhakta/tabcomplete-model-data-r2-baseline"
    write_json(
        folder / "kernel-metadata.json",
        {
            "id": reference,
            "title": "tabcomplete-model-data-r2-baseline",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["shlokbhakta/tabcomplete-code-cpt-parents-r1"],
            "kernel_sources": [],
            "competition_sources": [],
        },
    )
    state = {
        "reference": reference,
        "submitted_at": now(),
        "deadline_seconds": seconds,
        "reserved_wall_hours": 2,
        "plan_sha256": digest(REPORT / "preregistered_plan.json"),
        "git_sha": revision,
        "quota_before": quota,
        "status": "submission_pending",
    }
    write_json(state_path, state)
    result = command(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(folder),
            "--timeout",
            str(seconds),
            "--accelerator",
            "NvidiaTeslaT4",
        ],
        timeout=180,
    )
    state.update(status="submitted", response=result.strip())
    write_json(state_path, state)
    print(json.dumps({"reference": reference, "status": "submitted", "deadline_seconds": seconds}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/model_data_r2.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=["freeze", "baseline"], default="baseline")
    args = parser.parse_args()
    plan = freeze(args.config)
    if args.execute and args.stage == "baseline":
        submit_baseline(plan)
    else:
        print(
            json.dumps(
                {
                    "plan_sha256": digest(REPORT / "preregistered_plan.json"),
                    "frozen": True,
                    "execution_requested": args.execute,
                }
            )
        )


if __name__ == "__main__":
    main()
